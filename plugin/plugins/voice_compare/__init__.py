"""
voice_compare — N.E.K.O. 声音比对插件
======================================
算法：MFCC + 余弦相似度，无需任何外部模型
依赖：numpy, scipy

设计逻辑
--------
用户跟猫娘说"帮我比对声音" →
猫娘调用 open_ui 入口点 →
插件返回 WebUI 地址，猫娘告诉用户去 UI 操作 →
用户在网页上录音，网页直接调 /api/plugin/voice_compare/xxx 接口

入口点
------
open_ui   : 主入口，猫娘调用后引导用户打开 WebUI
compare   : 比对两段 Base64-WAV（WebUI 调用）
enroll    : 注册声纹（WebUI 调用）
verify    : 验证声纹（WebUI 调用）
status    : 查询状态
clear     : 清除声纹
threshold : 设置阈值
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
#  音频算法（纯函数）
# ─────────────────────────────────────────────────────────────

class _AudioError(Exception):
    pass


def _decode_wav(b64: str) -> tuple:
    """Base64-WAV → (float32 samples, sample_rate)"""
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise _AudioError(f"Base64 解码失败: {e}") from e
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(raw))
    except Exception as e:
        raise _AudioError(f"WAV 解析失败: {e}") from e

    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    return data, int(sr)


def _mel_fb(n_fft: int, sr: int, n_mels: int = 26) -> np.ndarray:
    def hz2mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel2hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts  = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2.0), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb   = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m-1], bins[m], bins[m+1]
        if mid > lo: fb[m-1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid: fb[m-1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def _get_embedding(samples: np.ndarray, sr: int,
                   n_mfcc=13, n_mels=26, n_fft=512, hop=256) -> np.ndarray:
    from scipy.signal import get_window
    from scipy.fft import rfft

    if len(samples) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[:len(samples)] = samples
        frames = pad[np.newaxis, :]
    else:
        n_fr   = 1 + (len(samples) - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            samples, shape=(n_fr, n_fft),
            strides=(samples.strides[0] * hop, samples.strides[0]),
        ).copy()

    frames *= get_window("hann", n_fft).astype(np.float32)
    power   = np.abs(rfft(frames, axis=1)) ** 2
    mel     = power @ _mel_fb(n_fft, sr, n_mels).T
    mel     = np.where(mel > 1e-10, np.log(mel), 0.0)

    idx  = np.arange(n_mels, dtype=np.float32)
    dct  = np.cos(np.pi / n_mels *
                  np.outer(np.arange(n_mfcc, dtype=np.float32), idx + 0.5))
    mfcc = mel @ dct.T

    vec  = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _confidence(sim: float, thr: float) -> str:
    if sim >= max(thr + 0.2, 0.75): return "高"
    if sim >= thr:                   return "中"
    if sim >= thr - 0.15:            return "低"
    return "极低"


# ─────────────────────────────────────────────────────────────
#  插件主类
# ─────────────────────────────────────────────────────────────

@neko_plugin
class VoiceComparePlugin(NekoPluginBase):

    N_MFCC = 13;  N_MELS = 26;  N_FFT = 512;  HOP = 256
    MIN_SEC = 0.5
    DEFAULT_THRESHOLD = 0.50

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        # PluginStore 不需要传 ctx，由基类自动注入
        self.store  = PluginStore()

        self._lock:      threading.Lock       = threading.Lock()
        self._embedding: Optional[np.ndarray] = None
        self._threshold: float                = self.DEFAULT_THRESHOLD

    # ── 内部工具 ─────────────────────────────────────────────

    def _to_emb(self, b64: str, label: str) -> np.ndarray:
        if not b64 or not isinstance(b64, str):
            raise SdkError(f"{label}：需要 Base64 字符串")
        try:
            samples, sr = _decode_wav(b64)
        except _AudioError as e:
            raise SdkError(f"{label}解码失败：{e}") from e
        if len(samples) < sr * self.MIN_SEC:
            raise SdkError(f"{label}太短，请录制至少 {self.MIN_SEC}s")
        try:
            return _get_embedding(samples, sr,
                                  self.N_MFCC, self.N_MELS, self.N_FFT, self.HOP)
        except Exception as e:
            raise SdkError(f"{label}特征提取失败：{e}") from e

    def _result(self, sim: float, thr: float) -> dict:
        is_same = sim >= thr
        return {
            "is_same":    is_same,
            "result":     "是" if is_same else "不是",
            "similarity": round(sim, 4),
            "threshold":  thr,
            "confidence": _confidence(sim, thr),
        }

    # ── 生命周期（同步，不用 async，避免 SDK await Ok 报错）─

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        self.logger.info("voice_compare: 启动")
        self.register_static_ui("static")

        # 从 PluginStore 恢复持久化数据
        try:
            thr = await self.store.get("threshold")
            if thr is not None:
                self._threshold = float(thr)
                self.logger.info(f"voice_compare: 恢复阈值 {self._threshold}")
        except Exception:
            pass

        try:
            emb = await self.store.get("embedding")
            if emb is not None:
                with self._lock:
                    self._embedding = np.array(emb, dtype=np.float32)
                self.logger.info("voice_compare: 恢复声纹特征")
        except Exception:
            pass

        # 注意：lifecycle 直接 return 字典，不要包 Ok()
        return {
            "status":    "ready",
            "enrolled":  self._embedding is not None,
            "threshold": self._threshold,
            "ui":        "/plugin/voice_compare/ui/",
        }

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        self.logger.info("voice_compare: 关闭")
        try:
            await self.store.set("threshold", self._threshold)
            with self._lock:
                emb = self._embedding
            if emb is not None:
                await self.store.set("embedding", emb.tolist())
                self.logger.info("voice_compare: 声纹已持久化")
        except Exception:
            pass
        return {"status": "stopped"}

    # ── Entry Points ─────────────────────────────────────────

    @plugin_entry(
        id="open_ui",
        name="打开声音比对界面",
        description=(
            "打开声音比对的网页操作界面。"
            "当用户说「比对声音」「声纹识别」「打开声音比对」等时调用。"
            "调用后将 ui_url 告知用户，引导其在网页上完成录音和比对。"
        ),
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["ui_url", "message"],
    )
    def open_ui(self, **_):
        """主入口：猫娘调用后告诉用户去 WebUI 操作"""
        enrolled = self._embedding is not None
        return Ok({
            "ui_url":   "/plugin/voice_compare/ui/",
            "enrolled": enrolled,
            "message": (
                "声音比对界面已就绪，请点击链接打开。"
                + ("已有注册声纹，可直接验证。"
                   if enrolled
                   else "还没有注册声纹，建议先在「注册声纹」页录一段主人声音。")
            ),
        })

    @plugin_entry(
        id="compare",
        name="声音比对",
        description="比对两段音频是否为同一人（由 WebUI 调用）",
        input_schema={
            "type": "object",
            "properties": {
                "audio1":    {"type": "string"},
                "audio2":    {"type": "string"},
                "threshold": {"type": "number"},
            },
            "required": ["audio1", "audio2"],
        },
        llm_result_fields=["result", "similarity", "confidence"],
    )
    def compare(self, audio1: str, audio2: str,
                threshold: Optional[float] = None, **_):
        thr = self._threshold if threshold is None else float(threshold)
        if not 0.0 <= thr <= 1.0:
            return Err(SdkError("threshold 须在 0~1 之间"))
        try:
            e1 = self._to_emb(audio1, "音频1")
            e2 = self._to_emb(audio2, "音频2")
        except SdkError as e:
            return Err(e)
        sim = _cosine(e1, e2)
        res = self._result(sim, thr)
        self.logger.info(f"compare: {'同一人' if res['is_same'] else '不同人'} sim={sim:.4f}")
        return Ok(res)

    @plugin_entry(
        id="enroll",
        name="注册声纹",
        description="注册主人声纹并持久化（由 WebUI 调用）",
        input_schema={
            "type": "object",
            "properties": {"audio": {"type": "string"}},
            "required": ["audio"],
        },
        llm_result_fields=["success", "message"],
    )
    async def enroll(self, audio: str, **_):
        try:
            emb = self._to_emb(audio, "注册音频")
        except SdkError as e:
            return Err(e)
        with self._lock:
            self._embedding = emb
        await self.store.set("embedding", emb.tolist())
        self.logger.info(f"enroll: 成功，维度={len(emb)}")
        return Ok({"success": True, "message": "声纹注册成功", "feature_dim": len(emb)})

    @plugin_entry(
        id="verify",
        name="验证声纹",
        description="与已注册声纹比对（由 WebUI 调用）",
        input_schema={
            "type": "object",
            "properties": {"audio": {"type": "string"}},
            "required": ["audio"],
        },
        llm_result_fields=["result", "similarity", "confidence", "message"],
    )
    def verify(self, audio: str, **_):
        with self._lock:
            ref = self._embedding.copy() if self._embedding is not None else None
        if ref is None:
            return Err(SdkError("尚未注册声纹，请先在 WebUI 中注册"))
        try:
            emb = self._to_emb(audio, "验证音频")
        except SdkError as e:
            return Err(e)
        sim = _cosine(ref, emb)
        res = self._result(sim, self._threshold)
        res["message"] = (
            f"验证通过，相似度 {sim:.1%}，置信度{res['confidence']}"
            if res["is_same"] else
            f"验证未通过，相似度 {sim:.1%}"
        )
        self.logger.info(f"verify: {'通过' if res['is_same'] else '未通过'} sim={sim:.4f}")
        return Ok(res)

    @plugin_entry(
        id="status",
        name="查询状态",
        description="查询声纹注册状态和当前阈值",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["enrolled", "threshold", "message"],
    )
    def status(self, **_):
        with self._lock:
            enrolled = self._embedding is not None
            dim      = len(self._embedding) if enrolled else 0
        return Ok({
            "enrolled":    enrolled,
            "feature_dim": dim,
            "threshold":   self._threshold,
            "ui_url":      "/plugin/voice_compare/ui/",
            "message":     f"已注册声纹（维度 {dim}），阈值 {self._threshold}"
                           if enrolled else "尚未注册声纹",
        })

    @plugin_entry(
        id="clear",
        name="清除声纹",
        description="清除已注册的主人声纹",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["success", "message"],
    )
    async def clear(self, **_):
        with self._lock:
            self._embedding = None
        await self.store.delete("embedding")
        self.logger.info("clear: 声纹已清除")
        return Ok({"success": True, "message": "声纹已清除，请重新注册"})

    @plugin_entry(
        id="threshold",
        name="设置阈值",
        description="调整判定阈值（0~1，越高越严格，默认 0.5）",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["value"],
        },
        llm_result_fields=["success", "threshold"],
    )
    async def set_threshold(self, value: float, **_):
        v = float(value)
        if not 0.0 <= v <= 1.0:
            return Err(SdkError("阈值须在 0~1 之间"))
        self._threshold = v
        await self.store.set("threshold", v)
        return Ok({"success": True, "threshold": v})
