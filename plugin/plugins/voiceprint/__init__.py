from plugin.sdk.plugin import (
    NekoPluginBase, neko_plugin, plugin_entry, lifecycle,
    Ok, Err,
)
from typing import Any, List
import math


@neko_plugin
class VoiceprintPlugin(NekoPluginBase):
    """声纹比对插件 - 简单比对两段音频是否为同一人"""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger

    def _bytes_to_samples(self, audio_data: bytes) -> List[float]:
        """将字节数据转换为采样点"""
        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            if i + 1 < len(audio_data):
                value = audio_data[i] | (audio_data[i + 1] << 8)
                if value >= 32768:
                    value -= 65536
                samples.append(value / 32768.0)
        return samples

    def _mean(self, data: List[float]) -> float:
        """计算平均值"""
        return sum(data) / len(data) if data else 0.0

    def _std(self, data: List[float]) -> float:
        """计算标准差"""
        if not data:
            return 0.0
        m = self._mean(data)
        variance = sum((x - m) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    def _extract_features(self, audio_data: bytes) -> List[float]:
        """从音频数据中提取声纹特征"""
        samples = self._bytes_to_samples(audio_data)

        if len(samples) < 10:
            return [0.0] * 13

        features = []

        # 时域特征
        features.append(self._mean(samples))
        features.append(self._std(samples))
        features.append(max(samples))
        features.append(min(samples))

        abs_values = [abs(x) for x in samples]
        features.append(self._mean(abs_values))
        features.append(self._std(abs_values))

        # 简化的频域特征
        n = len(samples)
        magnitudes = []
        step = max(1, n // 128)

        for k in range(0, min(n, 128), 2):
            real = 0.0
            imag = 0.0
            for i in range(0, n, step):
                angle = 2 * math.pi * k * i / n
                real += samples[i] * math.cos(angle)
                imag += samples[i] * math.sin(angle)
            magnitudes.append(math.sqrt(real * real + imag * imag))

        if magnitudes:
            features.append(self._mean(magnitudes))
            features.append(self._std(magnitudes))
            features.append(max(magnitudes))

            n_bands = 4
            band_size = len(magnitudes) // n_bands
            for i in range(n_bands):
                start = i * band_size
                end = start + band_size
                band = magnitudes[start:end]
                features.append(self._mean(band) if band else 0.0)
        else:
            features.extend([0.0] * 7)

        while len(features) < 13:
            features.append(0.0)
        return features[:13]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

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
                    "description": "相似度阈值",
                    "default": 0.7
                }
            },
            "required": ["audio1", "audio2"]
        }
    )
    async def compare_voice(self, audio1: str, audio2: str, threshold: float = 0.7, **_):
        """比对两段音频是否为同一人"""
        import base64

        try:
            audio1_bytes = base64.b64decode(audio1)
            audio2_bytes = base64.b64decode(audio2)

            if len(audio1_bytes) < 1000:
                return Err({"error": "第一段音频太短"})
            if len(audio2_bytes) < 1000:
                return Err({"error": "第二段音频太短"})

            features1 = self._extract_features(audio1_bytes)
            features2 = self._extract_features(audio2_bytes)

            similarity = self._cosine_similarity(features1, features2)
            is_same = similarity >= threshold

            return Ok({
                "result": "是" if is_same else "不是",
                "similarity": round(similarity, 4),
                "threshold": threshold
            })

        except Exception as e:
            self.logger.error(f"Compare failed: {e}")
            return Err({"error": str(e)})
