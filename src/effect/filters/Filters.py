# src/effects/filters.py

import numpy as np
from scipy.signal import butter, lfilter, lfiltic
from .base_filter import AudioFilter

class LowpassFilter(AudioFilter):
    def __init__(self, samplerate, cutoff_freq, order=5):
        super().__init__(samplerate, order)
        nyquist = 0.5 * samplerate
        normal_cutoff = min(cutoff_freq / nyquist, 0.99)
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
        self.zi = lfiltic(self.b, self.a, y=[], x=[])
        
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        filtered_data, self.zi = lfilter(self.b, self.a, audio_data, zi=self.zi)
        return filtered_data

class HighpassFilter(AudioFilter):
    def __init__(self, samplerate, cutoff_freq, order=5):
        super().__init__(samplerate, order)
        nyquist = 0.5 * samplerate
        normal_cutoff = max(cutoff_freq / nyquist, 0.001)
        self.b, self.a = butter(order, normal_cutoff, btype='high', analog=False)
        self.zi = lfiltic(self.b, self.a, y=[], x=[])
        
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        filtered_data, self.zi = lfilter(self.b, self.a, audio_data, zi=self.zi)
        return filtered_data

class BandpassFilter(AudioFilter):
    def __init__(self, samplerate, lowcut, highcut, order=5):
        super().__init__(samplerate, order)
        nyquist = 0.5 * samplerate
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.99)
        self.b, self.a = butter(order, [low, high], btype='band', analog=False)
        self.zi = lfiltic(self.b, self.a, y=[], x=[])
        
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        filtered_data, self.zi = lfilter(self.b, self.a, audio_data, zi=self.zi)
        return filtered_data