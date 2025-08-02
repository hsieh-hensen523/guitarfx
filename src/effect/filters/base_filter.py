# src/effects/base_filter.py
from abc import ABC, abstractmethod
import numpy as np

class AudioFilter(ABC):
    @abstractmethod
    def __init__(self, samplerate, order):
        # 抽象類別，確保子類別有這些共同屬性
        self.samplerate = samplerate
        self.order = order
        self.b = None
        self.a = None
        self.zi = None
        
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Processes audio data with the filter and updates the state.
        """
        pass