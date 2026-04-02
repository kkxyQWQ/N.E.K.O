"""
N.E.K.O. 声纹比对插件（轻量版）
基于 MFCC 特征提取的简单声纹识别
无需下载大型模型，直接使用！

优化版本 - 遵循SDK最佳实践
"""

from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err, SdkError, PluginStore,
)
from typing import Any, Optional
import threading
import base64
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import get_window


@neko_plugin
class VoiceprintPlugin(NekoPluginBase):
    """声纹比对插件（轻量版）- 基于MFCC特征，无需大型模型"""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger

        # 配置参数
        self.voiceprint_threshold = 0.5
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_fft = 512
        self.hop_length = 256

        # 缓存
        self._embedding1: Optional[np.ndarray] = None
        self._audio1: Optional[np.ndarray] = None

        # 线程安全锁（SDK最佳实践：保护共享状态）
        self._lock = threading.Lock()

        # 持久化存储
        self.store = PluginStore(ctx)

        # 缓存路径（使用 data_path 获取插件数据目录）
        self.cache_path = str(self.data_path("cache.wav"))
        self.reference_audio_path = str(self.data_path("reference.wav"))

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """音频预处理"""
        if len(audio) == 0:
            return audio
        return audio

    def _frame_audio(self, audio: np.ndarray) -> np.ndarray:
        """分帧处理"""
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        frames = np.zeros((n_frames, self.n_fft))

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end <= len(audio):
                frames[i] = audio[start:end]

        return frames

    def _apply_window(self, frames: np.ndarray) -> np.ndarray:
        """加汉宁窗"""
        window = get_window('hann', self.n_fft)
        return frames * window

    def _compute_mel_filterbank(self, n_mels: int = 26) -> np.ndarray:
        """计算梅尔滤波器组"""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        low_freq_mel = hz_to_mel(0)
        high_freq_mel = hz_to_mel(self.sample_rate / 2)

        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        fft_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        filters = np.zeros((n_mels, self.n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = fft_bins[m - 1]
            f_m = fft_bins[m]
            f_m_plus = fft_bins[m + 1]

            for k in range(f_m_minus, f_m):
                filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filters

    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """计算MFCC特征"""
        frames = self._frame_audio(audio)
        windowed = self._apply_window(frames)
        spec = np.abs(fft(windowed, axis=1))[:, :self.n_fft // 2 + 1]
        power_spec = spec ** 2
        mel_filters = self._compute_mel_filterbank()
        mel_spec = np.dot(power_spec, mel_filters.T)
        mel_spec = np.where(mel_spec > 0, np.log(mel_spec), 0)

        n_mels = mel_spec.shape[1]
        dct_matrix = np.zeros((self.n_mfcc, n_mels))
        for i in range(self.n_mfcc):
            dct_matrix[i] = np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels)
        mfcc = np.dot(mel_spec, dct_matrix.T)

        return mfcc

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """提取声纹特征向量"""
        mfcc = self._compute_mfcc(audio)
        mean_mfcc = np.mean(mfcc, axis=0)
        std_mfcc = np.std(mfcc, axis=0)
        embedding = np.concatenate([mean_mfcc, std_mfcc])

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _decode_audio(self, audio_base64: str, save_path: str) -> bool:
        """解码音频并保存"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with open(save_path, 'wb') as f:
                f.write(audio_bytes)
            return True
        except Exception as e:
            # 使用 exception() 记录完整堆栈信息
            self.logger.exception(f"解码音频失败: {e}")
            return False

    def _load_wav(self, filename: str) -> tuple:
        """加载WAV音频"""
        sr, audio = wavfile.read(filename)

        if len(audio.shape) > 1:
            audio = audio[:, 0]

        audio = audio.astype(np.float32)
        if audio.max() != 0:
            audio = audio / audio.max()

        return audio, sr

    def _get_confidence(self, similarity: float, threshold: float) -> str:
        """获取置信度等级"""
        if similarity > 0.7:
            return "高"
        elif similarity >= threshold:
            return "中"
        return "低"

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        """插件启动"""
        self.logger.info("VoiceprintPlugin(轻量版) 启动中...")

        # 注册静态UI目录
        self.register_static_ui("static")

        # 加载持久化的阈值设置
        stored_threshold = await self.store.get("threshold")
        if stored_threshold is not None:
            self.voiceprint_threshold = float(stored_threshold)
            self.logger.info(f"从存储加载阈值: {self.voiceprint_threshold}")

        # 加载持久化的声纹特征
        stored_embedding = await self.store.get("embedding")
        if stored_embedding is not None:
            with self._lock:
                self._embedding1 = np.array(stored_embedding)
            self.logger.info("已加载持久化的声纹特征")

        return Ok({
            "status": "ready",
            "model": "MFCC_Lightweight",
            "threshold": self.voiceprint_threshold,
            "voice_enrolled": self._embedding1 is not None,
            "ui_url": "/plugin/voiceprint/ui/",
            "usage": "访问 UI 使用录音比对功能"
        })

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        """插件关闭 - 保存状态"""
        self.logger.info("VoiceprintPlugin 关闭中...")

        # 保存当前阈值
        await self.store.set("threshold", self.voiceprint_threshold)

        # 保存声纹特征（如果已注册）
        with self._lock:
            if self._embedding1 is not None:
                embedding_list = self._embedding1.tolist()
                await self.store.set("embedding", embedding_list)
                self.logger.info("声纹特征已持久化")

        return Ok({"status": "stopped"})

    @plugin_entry(
        id="enroll",
        name="注册声纹",
        description="将一段音频注册为声纹样本（只需注册一次，会持久化保存）",
        input_schema={
            "type": "object",
            "properties": {
                "audio": {
                    "type": "string",
                    "description": "音频数据（Base64编码的WAV格式）"
                }
            },
            "required": ["audio"]
        },
        llm_result_fields=["success", "message"]
    )
    async def enroll_voiceprint(self, audio: str, **_):
        """注册声纹样本"""
        # 验证输入
        if not audio or not isinstance(audio, str):
            return Err(SdkError("无效的音频数据"))

        # 解码并保存音频
        if not self._decode_audio(audio, self.reference_audio_path):
            return Err(SdkError("音频解码失败，请确保是有效的WAV格式"))

        # 加载音频
        try:
            audio_data, sample_rate = self._load_wav(self.reference_audio_path)
        except Exception as e:
            # 使用 exception() 记录完整堆栈信息
            self.logger.exception(f"加载音频失败: {e}")
            return Err(SdkError(f"加载音频失败: {str(e)}"))

        # 验证音频长度
        if len(audio_data) < self.sample_rate * 0.5:  # 少于0.5秒
            return Err(SdkError(f"音频太短，请提供至少0.5秒的音频"))

        # 预处理
        audio_data = self._preprocess_audio(audio_data)

        # 提取声纹特征
        new_embedding = self._extract_embedding(audio_data)

        # 使用锁保护共享状态（线程安全）
        with self._lock:
            self._audio1 = audio_data
            self._embedding1 = new_embedding
            # 持久化保存
            await self.store.set("embedding", new_embedding.tolist())

        self.logger.info(f"声纹注册成功！特征维度: {len(new_embedding)}")

        return Ok({
            "success": True,
            "message": "声纹注册成功，已自动保存",
            "sample_rate": sample_rate,
            "audio_duration": round(len(audio_data) / sample_rate, 2),
            "feature_dim": len(new_embedding)
        })

    @plugin_entry(
        id="verify",
        name="验证声纹",
        description="将待验证的音频与已注册的声纹进行比对，判断是否为同一人",
        input_schema={
            "type": "object",
            "properties": {
                "audio": {
                    "type": "string",
                    "description": "待验证的音频（Base64编码的WAV格式）"
                }
            },
            "required": ["audio"]
        },
        llm_result_fields=["result", "similarity", "confidence"]
    )
    async def verify_voiceprint(self, audio: str, **_):
        """验证声纹"""
        # 检查是否已注册声纹（线程安全读取）
        with self._lock:
            if self._embedding1 is None:
                return Err(SdkError("尚未注册声纹，请先调用 enroll 注册"))
            stored_embedding = self._embedding1.copy()

        # 验证输入
        if not audio or not isinstance(audio, str):
            return Err(SdkError("无效的音频数据"))

        # 解码并保存音频
        if not self._decode_audio(audio, self.cache_path):
            return Err(SdkError("音频解码失败"))

        # 加载音频
        try:
            audio_data, sample_rate = self._load_wav(self.cache_path)
        except Exception as e:
            self.logger.exception(f"加载音频失败: {e}")
            return Err(SdkError(f"加载音频失败: {str(e)}"))

        # 验证音频长度
        if len(audio_data) < self.sample_rate * 0.5:  # 少于0.5秒
            return Err(SdkError(f"音频太短，请提供至少0.5秒的音频"))

        # 预处理
        audio_data = self._preprocess_audio(audio_data)

        # 提取声纹特征
        embedding2 = self._extract_embedding(audio_data)

        # 计算相似度（使用锁外复制的stored_embedding）
        similarity = self._cosine_similarity(stored_embedding, embedding2)
        is_same_person = similarity >= self.voiceprint_threshold

        self.logger.info(
            f"声纹验证: {'是同一人' if is_same_person else '不是同一人'} "
            f"(相似度={similarity:.4f}, 阈值={self.voiceprint_threshold})"
        )

        result_text = "是" if is_same_person else "不是"

        return Ok({
            "result": result_text,
            "similarity": round(similarity, 4),
            "threshold": self.voiceprint_threshold,
            "confidence": self._get_confidence(similarity, self.voiceprint_threshold),
            "sample_rate": sample_rate
        })

    @plugin_entry(
        id="compare",
        name="声纹比对",
        description="比对两段音频，判断是否为同一人（无需提前注册）",
        input_schema={
            "type": "object",
            "properties": {
                "audio1": {
                    "type": "string",
                    "description": "第一段音频（Base64编码的WAV格式）"
                },
                "audio2": {
                    "type": "string",
                    "description": "第二段音频（Base64编码的WAV格式）"
                },
                "threshold": {
                    "type": "number",
                    "description": "相似度阈值 (0.0-1.0)",
                    "default": 0.5
                }
            },
            "required": ["audio1", "audio2"]
        },
        llm_result_fields=["result", "similarity", "confidence"]
    )
    async def compare_voice(self, audio1: str, audio2: str, threshold: float = 0.5, **_):
        """直接比对两段音频"""
        # 验证输入
        if not audio1 or not audio2:
            return Err(SdkError("请提供两段音频数据"))

        if not (0 <= threshold <= 1):
            return Err(SdkError("阈值必须在 0.0 到 1.0 之间"))

        # 解码音频1
        if not self._decode_audio(audio1, self.reference_audio_path):
            return Err(SdkError("音频1解码失败"))

        # 解码音频2
        if not self._decode_audio(audio2, self.cache_path):
            return Err(SdkError("音频2解码失败"))

        # 加载音频
        try:
            audio_data1, sr1 = self._load_wav(self.reference_audio_path)
            audio_data2, sr2 = self._load_wav(self.cache_path)
        except Exception as e:
            self.logger.exception(f"加载音频失败: {e}")
            return Err(SdkError(f"加载音频失败: {str(e)}"))

        # 验证音频长度
        if len(audio_data1) < self.sample_rate * 0.5:
            return Err(SdkError("音频1太短，请提供至少0.5秒的音频"))
        if len(audio_data2) < self.sample_rate * 0.5:
            return Err(SdkError("音频2太短，请提供至少0.5秒的音频"))

        # 预处理
        audio_data1 = self._preprocess_audio(audio_data1)
        audio_data2 = self._preprocess_audio(audio_data2)

        # 提取声纹特征
        embedding1 = self._extract_embedding(audio_data1)
        embedding2 = self._extract_embedding(audio_data2)

        # 计算相似度
        similarity = self._cosine_similarity(embedding1, embedding2)
        is_same_person = similarity >= threshold

        self.logger.info(
            f"声纹比对: {'是同一人' if is_same_person else '不是同一人'} "
            f"(相似度={similarity:.4f}, 阈值={threshold})"
        )

        result_text = "是" if is_same_person else "不是"

        return Ok({
            "result": result_text,
            "similarity": round(similarity, 4),
            "threshold": threshold,
            "confidence": self._get_confidence(similarity, threshold),
            "audio1_duration": round(len(audio_data1) / sr1, 2),
            "audio2_duration": round(len(audio_data2) / sr2, 2)
        })

    @plugin_entry(
        id="set_threshold",
        name="设置阈值",
        description="设置声纹识别的相似度阈值，会自动保存",
        input_schema={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "阈值 (0.0-1.0)，越低越严格",
                    "default": 0.5
                }
            }
        },
        llm_result_fields=["success", "threshold"]
    )
    async def set_threshold(self, threshold: float = 0.5, **_):
        """设置阈值"""
        if not (0 <= threshold <= 1):
            return Err(SdkError("阈值必须在 0.0 到 1.0 之间"))

        self.voiceprint_threshold = threshold

        # 持久化保存
        await self.store.set("threshold", threshold)

        self.logger.info(f"声纹识别阈值已设置为: {threshold}")

        return Ok({
            "success": True,
            "threshold": threshold,
            "message": f"阈值已设置为 {threshold}，已自动保存"
        })

    @plugin_entry(
        id="get_status",
        name="状态查询",
        description="查询插件当前状态",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    async def get_status(self, **_):
        """获取状态"""
        return Ok({
            "model": "MFCC_Lightweight",
            "no_model_required": True,
            "voice_enrolled": self._embedding1 is not None,
            "threshold": self.voiceprint_threshold,
            "sample_rate": self.sample_rate,
            "feature_dim": len(self._embedding1) if self._embedding1 is not None else 0,
            "ui_url": "/plugin/voiceprint/ui/"
        })

    @plugin_entry(
        id="clear_enrollment",
        name="清除注册",
        description="清除已注册的声纹数据",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    async def clear_enrollment(self, **_):
        """清除注册的声纹"""
        # 使用锁保护共享状态（线程安全）
        with self._lock:
            self._embedding1 = None
            self._audio1 = None

        # 删除持久化数据
        await self.store.delete("embedding")

        self.logger.info("已清除注册的声纹数据")

        return Ok({
            "success": True,
            "message": "已清除注册的声纹数据"
        })
