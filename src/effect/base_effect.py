# src/effect/base_effect.py
from abc import ABC, abstractmethod
import numpy as np

class AudioEffect(ABC):
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        處理音訊數據並回傳處理後的數據。
        """
        pass