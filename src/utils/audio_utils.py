import numpy as np
from scipy.signal import butter, lfilter

def get_rms(data):
    """
    計算音訊數據的 RMS (Root Mean Square) 值，代表音量大小。
    """
    if len(data.tobytes()) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(data)))

def is_pop_noise(fft_magnitude, sample_rate, prev_energy=0.0, min_energy=0.01):
    # freqs = np.fft.rfftfreq(len(fft_magnitude)*2 - 1, 1/sample_rate)
    total_energy = np.sum(fft_magnitude)
    if total_energy < min_energy:
        return False, prev_energy
    if total_energy - prev_energy > 500:
        print(f"[爆音偵測] 檢測到爆音: 總能量={total_energy:.4f}, 前一能量={prev_energy:.4f}")
        return total_energy - prev_energy > 500, prev_energy
    return False, total_energy

def is_transient_noise(data, threshold=0.3):
    diff = np.abs(np.diff(data))
    peak = np.max(diff)
    return peak > threshold

def spectral_flatness(fft_magnitude):
    geo_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
    arith_mean = np.mean(fft_magnitude)
    return geo_mean / (arith_mean + 1e-10)
