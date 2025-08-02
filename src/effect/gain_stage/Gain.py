# src/effect/gain_stage/Gain.py
import numpy as np
from ..base_effect import AudioEffect

class Gain(AudioEffect):
    def __init__(self, gain_factor=1.0):
        self.gain_factor = gain_factor

    def set_parameters(self, gain_factor):
        self.gain_factor = gain_factor

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        return audio_data * self.gain_factor