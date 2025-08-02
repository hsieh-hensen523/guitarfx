import numpy as np
from scipy.signal import butter, lfilter

def get_rms(data):
    """
    計算音訊數據的 RMS (Root Mean Square) 值，代表音量大小。
    """
    if len(data.tobytes()) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(data)))

def is_pop_noise(fft_magnitude, sample_rate, threshold_ratio=0.2, min_energy=0.01):
    # freqs = np.fft.rfftfreq(len(fft_magnitude)*2 - 1, 1/sample_rate)
    total_energy = np.sum(fft_magnitude)
    if total_energy < min_energy:
        return False
    # speech_band = (freqs >= 100) & (freqs <= 3500)
    # # speech_energy = np.sum(fft_magnitude[speech_band])
    # # # if speech_energy > 50:
    # # #     print(f"Total energy: {total_energy}, Speech energy: {speech_energy}")
    return total_energy > 1000

def is_transient_noise(data, threshold=0.3):
    diff = np.abs(np.diff(data))
    peak = np.max(diff)
    return peak > threshold

def spectral_flatness(fft_magnitude):
    geo_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
    arith_mean = np.mean(fft_magnitude)
    return geo_mean / (arith_mean + 1e-10)
