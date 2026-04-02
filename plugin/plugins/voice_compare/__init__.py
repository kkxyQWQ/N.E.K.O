"""
voice_compare — N.E.K.O. 声音比对插件
======================================
算法：MFCC（Mel 频率倒谱系数）+ 余弦相似度
依赖：numpy, scipy（均为纯 Python，无需 GPU / 大型模型）

Entry Points
------------
compare   : 直接比对两段 Base64-WAV，返回是否同一人
enroll    : 注册主人声纹，持久化保存到 PluginStore
verify    : 与已注册声纹比对
status    : 查询插件当前状态
clear     : 清除已注册声纹
threshold : 修改全局判定阈值

UI
--
访问 /plugin/voice_compare/ui/ 打开网页录音界面
"""

import io
import base64
import threading
from typing import Any, Optional

import numpy as np

from plugin.sdk.plugin import (
    NekoPluginBase,
    PluginStore,
    Ok, Err, SdkError,
    neko_plugin, plugin_entry, lifecycle,
)


# ─────────────────────────────────────────────────────────────
#  纯算法工具（无副作用，方便单元测试）
# ─────────────────────────────────────────────────────────────

class AudioTooShortError(Exception):
    pass

class AudioDecodeError(Exception):
    pass


def decode_wav_b64(b64_str: str) -> tuple[np.ndarray, int]:
    """
    把 Base64 编码的 WAV 字节流解码成 float32 数组。
    返回 (samples: float32 ndarray, sample_rate: int)
    全程在内存中处理，不写磁盘。
    """
    try:
        raw = base64.b64decode(b64_str)
    except Exception as e:
        raise AudioDecodeError(f"Base64 解码失败: {e}") from e

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(raw))
    except Exception as e:
        raise AudioDecodeError(f"WAV 解析失败: {e}") from e

    # 多声道 → 取第 0 声道
    if data.ndim > 1:
        data = data[:, 0]

    # 整数类型 → float32，归一化到 [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return data, int(sr)


def check_duration(samples: np.ndarray, sr: int,
                   min_sec: float = 0.5, label: str = "音频") -> None:
    """时长不足则抛出 AudioTooShortError"""
    duration = len(samples) / sr
    if duration < min_sec:
        raise AudioTooShortError(
            f"{label}太短（{duration:.2f}s），请录制至少 {min_sec}s"
        )


def build_mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    """构建梅尔滤波器组，形状 (n_mels, n_fft//2+1)"""
    def hz2mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel2hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz2mel(0)
    mel_max = hz2mel(sr / 2.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            fb[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def extract_mfcc(
    samples: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 26,
    n_fft:  int = 512,
    hop:    int = 256,
) -> np.ndarray:
    """
    提取 MFCC，返回形状 (n_frames, n_mfcc)。
    纯 NumPy + SciPy，无第三方音频库依赖。
    """
    from scipy.signal import get_window
    from scipy.fft import rfft

    # 1. 分帧（stride trick，零拷贝）
    if len(samples) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[: len(samples)] = samples
        frames = pad[np.newaxis, :]
    else:
        n_frames = 1 + (len(samples) - n_fft) // hop
        shape   = (n_frames, n_fft)
        strides = (samples.strides[0] * hop, samples.strides[0])
        frames  = np.lib.stride_tricks.as_strided(samples, shape, strides).copy()

    # 2. 加汉宁窗
    frames *= get_window("hann", n_fft).astype(np.float32)

    # 3. 功率谱（rfft 只算正频率，效率是 fft 的两倍）
    power = np.abs(rfft(frames, axis=1)) ** 2   # (n_frames, n_fft//2+1)

    # 4. 梅尔滤波
    fb        = build_mel_filterbank(n_fft, sr, n_mels)
    mel_spec  = power @ fb.T                     # (n_frames, n_mels)
    mel_spec  = np.where(mel_spec > 1e-10, np.log(mel_spec), 0.0)

    # 5. DCT → MFCC（手动实现，不需要 scipy.fft.dct）
    n_m   = mel_spec.shape[1]
    idx   = np.arange(n_m, dtype=np.float32)
    dct   = np.cos(
        np.pi / n_m * np.outer(np.arange(n_mfcc, dtype=np.float32), idx + 0.5)
    ).astype(np.float32)                         # (n_mfcc, n_mels)
    mfcc  = mel_spec @ dct.T                     # (n_frames, n_mfcc)
    return mfcc


def mfcc_to_embedding(mfcc: np.ndarray) -> np.ndarray:
    """
    把 MFCC 矩阵压缩为一维特征向量：
    拼接 [均值, 标准差] 后 L2 归一化 → 余弦相似度可直接用点积计算。
    """
    vec  = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


def audio_to_embedding(
    samples: np.ndarray, sr: int,
    n_mfcc: int = 13, n_mels: int = 26,
    n_fft: int = 512, hop: int = 256,
) -> np.ndarray:
    """一步完成：原始音频 → 归一化声纹向量"""
    mfcc = extract_mfcc(samples, sr, n_mfcc, n_mels, n_fft, hop)
    return mfcc_to_embedding(mfcc)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，两向量已 L2 归一化时等于点积"""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def similarity_to_confidence(sim: float, threshold: float) -> str:
    if sim >= max(threshold + 0.2, 0.75):
        return "高"
    if sim >= threshold:
        return "中"
    if sim >= threshold - 0.15:
        return "低"
    return "极低"


# ─────────────────────────────────────────────────────────────
#  插件主类
# ─────────────────────────────────────────────────────────────

@neko_plugin
class VoiceComparePlugin(NekoPluginBase):
    """
    声音比对插件

    三种使用方式：
      A) 直接比对   compare(audio1, audio2)
      B) 注册+验证  enroll(audio) → verify(audio)
      C) 网页 UI    /plugin/voice_compare/ui/
    """

    # 声学超参（固定，无需用户配置）
    N_MFCC  = 13
    N_MELS  = 26
    N_FFT   = 512
    HOP     = 256
    MIN_SEC = 0.5   # 最短有效音频（秒）

    # 默认阈值
    DEFAULT_THRESHOLD = 0.50

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self.store  = PluginStore(ctx)

        # 已注册声纹（受 _lock 保护）
        self._lock:      threading.Lock        = threading.Lock()
        self._embedding: Optional[np.ndarray]  = None

        # 判定阈值（可调）
        self._threshold: float = self.DEFAULT_THRESHOLD

    # ── 内部工具 ──────────────────────────────────────────────

    def _b64_to_embedding(self, b64: str, label: str = "音频") -> np.ndarray:
        """
        Base64-WAV → 声纹特征向量
        出错一律 raise SdkError，由调用方转成 Err(...)
        """
        if not b64 or not isinstance(b64, str):
            raise SdkError(f"{label}：需要 Base64 字符串")

        try:
            samples, sr = decode_wav_b64(b64)
        except AudioDecodeError as e:
            raise SdkError(f"{label}解码失败：{e}") from e

        try:
            check_duration(samples, sr, self.MIN_SEC, label)
        except AudioTooShortError as e:
            raise SdkError(str(e)) from e

        try:
            emb = audio_to_embedding(
                samples, sr,
                self.N_MFCC, self.N_MELS, self.N_FFT, self.HOP,
            )
        except Exception as e:
            raise SdkError(f"{label}特征提取失败：{e}") from e

        return emb

    def _make_result(self, sim: float, thr: float) -> dict:
        """构造统一的比对结果字段"""
        is_same    = sim >= thr
        confidence = similarity_to_confidence(sim, thr)
        return {
            "is_same":    is_same,
            "result":     "是" if is_same else "不是",
            "similarity": round(sim, 4),
            "threshold":  thr,
            "confidence": confidence,
        }

    # ── 生命周期 ──────────────────────────────────────────────

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        self.logger.info("voice_compare: 启动中…")

        # 挂载静态 UI
        self.register_static_ui("static")

        # 从 PluginStore 恢复阈值
        saved_thr = await self.store.get("threshold")
        if saved_thr is not None:
            try:
                self._threshold = float(saved_thr)
                self.logger.info(f"voice_compare: 恢复阈值 {self._threshold}")
            except (TypeError, ValueError):
                pass  # 存储数据损坏，使用默认值

        # 从 PluginStore 恢复声纹
        saved_emb = await self.store.get("embedding")
        if saved_emb is not None:
            try:
                with self._lock:
                    self._embedding = np.array(saved_emb, dtype=np.float32)
                self.logger.info(
                    f"voice_compare: 恢复声纹特征（维度={len(self._embedding)}）"
                )
            except Exception:
                self.logger.warning("voice_compare: 声纹数据损坏，已忽略")

        enrolled = self._embedding is not None
        return Ok({
            "status":    "ready",
            "enrolled":  enrolled,
            "threshold": self._threshold,
            "ui":        "/plugin/voice_compare/ui/",
            "algorithm": "MFCC-13 + Cosine Similarity",
        })

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        self.logger.info("voice_compare: 关闭，持久化状态…")

        # 保存阈值
        await self.store.set("threshold", self._threshold)

        # 保存声纹（若已注册）
        with self._lock:
            emb = self._embedding

        if emb is not None:
            await self.store.set("embedding", emb.tolist())
            self.logger.info("voice_compare: 声纹已持久化")

        return Ok({"status": "stopped"})

    # ── Entry Points ──────────────────────────────────────────

    @plugin_entry(
        id="compare",
        name="声音比对",
        description=(
            "直接比对两段音频，判断是否为同一人，无需提前注册。"
            "音频为 Base64 编码的 WAV 格式（16kHz 单声道）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "audio1": {
                    "type": "string",
                    "description": "第一段音频（Base64 WAV）",
                },
                "audio2": {
                    "type": "string",
                    "description": "第二段音频（Base64 WAV）",
                },
                "threshold": {
                    "type": "number",
                    "description": "本次比对阈值（0~1），不填则用全局设置",
                },
            },
            "required": ["audio1", "audio2"],
        },
        llm_result_fields=["result", "similarity", "confidence"],
    )
    async def compare(
        self,
        audio1: str,
        audio2: str,
        threshold: Optional[float] = None,
        **_,
    ):
        thr = self._threshold if threshold is None else float(threshold)
        if not 0.0 <= thr <= 1.0:
            return Err(SdkError("threshold 必须在 0.0 ~ 1.0 之间"))

        try:
            emb1 = self._b64_to_embedding(audio1, "音频1")
            emb2 = self._b64_to_embedding(audio2, "音频2")
        except SdkError as e:
            return Err(e)

        sim = cosine_similarity(emb1, emb2)
        result = self._make_result(sim, thr)
        self.logger.info(
            f"compare: {'同一人' if result['is_same'] else '不同人'} "
            f"sim={sim:.4f} thr={thr}"
        )
        return Ok(result)

    # ----------------------------------------------------------

    @plugin_entry(
        id="enroll",
        name="注册声纹",
        description=(
            "录一段声音作为主人声纹，持久化保存，重启后不丢失。"
            "建议录制 3 秒以上的清晰语音。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "audio": {
                    "type": "string",
                    "description": "注册音频（Base64 WAV）",
                },
            },
            "required": ["audio"],
        },
        llm_result_fields=["success", "message"],
    )
    async def enroll(self, audio: str, **_):
        try:
            emb = self._b64_to_embedding(audio, "注册音频")
        except SdkError as e:
            return Err(e)

        with self._lock:
            self._embedding = emb

        # 立即持久化，无需等到 shutdown
        await self.store.set("embedding", emb.tolist())

        self.logger.info(f"enroll: 声纹注册成功，维度={len(emb)}")
        return Ok({
            "success":     True,
            "message":     "声纹注册成功，已自动保存",
            "feature_dim": len(emb),
        })

    # ----------------------------------------------------------

    @plugin_entry(
        id="verify",
        name="验证声纹",
        description="将一段音频与已注册的主人声纹比对，判断是否为同一人。请先调用 enroll 注册。",
        input_schema={
            "type": "object",
            "properties": {
                "audio": {
                    "type": "string",
                    "description": "待验证音频（Base64 WAV）",
                },
            },
            "required": ["audio"],
        },
        llm_result_fields=["result", "similarity", "confidence", "message"],
    )
    async def verify(self, audio: str, **_):
        # 线程安全地读取已注册声纹
        with self._lock:
            ref = self._embedding.copy() if self._embedding is not None else None

        if ref is None:
            return Err(SdkError("尚未注册声纹，请先调用 enroll"))

        try:
            emb = self._b64_to_embedding(audio, "验证音频")
        except SdkError as e:
            return Err(e)

        sim    = cosine_similarity(ref, emb)
        result = self._make_result(sim, self._threshold)
        result["message"] = (
            f"验证通过，相似度 {sim:.1%}，置信度{result['confidence']}"
            if result["is_same"]
            else f"验证未通过，相似度 {sim:.1%}，低于阈值 {self._threshold}"
        )

        self.logger.info(
            f"verify: {'通过' if result['is_same'] else '未通过'} "
            f"sim={sim:.4f} thr={self._threshold}"
        )
        return Ok(result)

    # ----------------------------------------------------------

    @plugin_entry(
        id="status",
        name="查询状态",
        description="查询插件当前状态：是否已注册声纹、当前阈值、算法信息等。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["enrolled", "threshold", "message"],
    )
    async def status(self, **_):
        with self._lock:
            enrolled  = self._embedding is not None
            feat_dim  = len(self._embedding) if enrolled else 0

        msg = (
            f"已注册声纹（特征维度 {feat_dim}），当前阈值 {self._threshold}"
            if enrolled
            else "尚未注册声纹，请先调用 enroll"
        )
        return Ok({
            "enrolled":    enrolled,
            "feature_dim": feat_dim,
            "threshold":   self._threshold,
            "algorithm":   "MFCC-13 + Cosine Similarity",
            "ui":          "/plugin/voice_compare/ui/",
            "message":     msg,
        })

    # ----------------------------------------------------------

    @plugin_entry(
        id="clear",
        name="清除声纹",
        description="清除已注册的主人声纹（不可恢复，需重新 enroll）。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["success", "message"],
    )
    async def clear(self, **_):
        with self._lock:
            self._embedding = None

        await self.store.delete("embedding")

        self.logger.info("clear: 声纹数据已清除")
        return Ok({
            "success": True,
            "message": "声纹已清除，请重新注册",
        })

    # ----------------------------------------------------------

    @plugin_entry(
        id="threshold",
        name="设置阈值",
        description=(
            "调整声纹判定的相似度阈值（0~1）。"
            "值越高判定越严格。推荐范围：0.40 ~ 0.65，默认 0.50。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "value": {
                    "type":        "number",
                    "description": "新阈值（0.0 ~ 1.0）",
                    "minimum":     0.0,
                    "maximum":     1.0,
                },
            },
            "required": ["value"],
        },
        llm_result_fields=["success", "threshold", "message"],
    )
    async def set_threshold(self, value: float, **_):
        if not 0.0 <= value <= 1.0:
            return Err(SdkError("阈值必须在 0.0 ~ 1.0 之间"))

        self._threshold = float(value)
        await self.store.set("threshold", self._threshold)

        self.logger.info(f"threshold: 已更新为 {self._threshold}")
        return Ok({
            "success":   True,
            "threshold": self._threshold,
            "message":   f"阈值已设为 {self._threshold}，立即生效并已保存",
        })
