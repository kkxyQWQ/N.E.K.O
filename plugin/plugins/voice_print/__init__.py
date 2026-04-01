"""
N.E.K.O. 声纹比对插件
输入两段音频，判断是否为同一个人
"""

from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err,
)
from typing import Any, List
import numpy as np


@neko_plugin
class VoiceprintPlugin(NekoPluginBase):
    """声纹比对插件 - 简单比对两段音频是否为同一人"""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self.feature_dim = 13  # MFCC特征维度

    def _extract_features(self, audio_data: bytes) -> List[float]:
        """
        从音频数据中提取声纹特征
        简化实现：基于音频信号的统计特征
        """
        # 将字节转换为numpy数组（假设16位PCM格式）
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # 转换为浮点数并归一化
        audio_float = audio_array.astype(np.float32) / 32768.0

        # 提取统计特征作为声纹特征向量
        features = []

        # 时域特征
        features.append(float(np.mean(audio_float)))           # 均值
        features.append(float(np.std(audio_float)))              # 标准差
        features.append(float(np.max(audio_float)))              # 最大值
        features.append(float(np.min(audio_float)))              # 最小值
        features.append(float(np.mean(np.abs(audio_float))))    # 平均绝对值
        features.append(float(np.std(np.abs(audio_float))))      # 绝对值标准差

        # 频域特征（简化的频谱分析）
        fft = np.fft.fft(audio_float)
        magnitude = np.abs(fft[:len(fft)//2])

        # 频谱统计特征
        features.append(float(np.mean(magnitude)))              # 频谱均值
        features.append(float(np.std(magnitude)))               # 频谱标准差
        features.append(float(np.max(magnitude)))              # 频谱最大值

        # 频段能量
        n_bands = 4
        band_size = len(magnitude) // n_bands
        for i in range(n_bands):
            band = magnitude[i*band_size:(i+1)*band_size]
            features.append(float(np.mean(band)))               # 各频段能量

        # 确保特征维度一致
        while len(features) < self.feature_dim:
            features.append(0.0)
        features = features[:self.feature_dim]

        return features

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    @lifecycle(id="startup")
    def on_startup(self, **_):
        """插件启动"""
        self.logger.info("VoiceprintPlugin started!")
        return Ok({"status": "ready"})

    @plugin_entry(
        id="compare",
        name="声纹比对",
        description="比对两段音频，判断是否为同一个人。是则输出'是'，否则输出'不是'",
        input_schema={
            "type": "object",
            "properties": {
                "audio1": {
                    "type": "string",
                    "description": "第一段音频（Base64编码）"
                },
                "audio2": {
                    "type": "string",
                    "description": "第二段音频（Base64编码）"
                },
                "threshold": {
                    "type": "number",
                    "description": "相似度阈值，低于此值判定为不同人",
                    "default": 0.7
                }
            },
            "required": ["audio1", "audio2"]
        }
    )
    async def compare_voice(self, audio1: str, audio2: str, threshold: float = 0.7, **_):
        """
        比对两段音频是否为同一人
        返回：是 / 不是
        """
        import base64

        try:
            # 解码两段音频
            audio1_bytes = base64.b64decode(audio1)
            audio2_bytes = base64.b64decode(audio2)

            # 检查音频长度
            if len(audio1_bytes) < 1000:
                return Err({"error": "第一段音频太短"})
            if len(audio2_bytes) < 1000:
                return Err({"error": "第二段音频太短"})

            # 提取特征
            features1 = self._extract_features(audio1_bytes)
            features2 = self._extract_features(audio2_bytes)

            # 计算相似度
            similarity = self._cosine_similarity(features1, features2)

            # 判定结果
            is_same_person = similarity >= threshold

            self.logger.info(f"Voice compare: similarity={similarity:.4f}, threshold={threshold}, result={'是' if is_same_person else '不是'}")

            return Ok({
                "result": "是" if is_same_person else "不是",
                "similarity": round(similarity, 4),
                "threshold": threshold,
                "confidence": "高" if is_same_person else "低" if similarity < 0.5 else "中"
            })

        except Exception as e:
            self.logger.error(f"Compare failed: {e}")
            return Err({"error": str(e)})
