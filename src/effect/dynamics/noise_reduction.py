# src/effect/dynamics/noise_reduction.py

from ..base_effect import AudioEffect
import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import windows

class NoiseReduction(AudioEffect):
    def __init__(self, samplerate, chunk_size, learning_frames=20, overlap_ratio=0.5):
        # 初始化常數
        self.samplerate = samplerate
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        
        # 為了重疊相加，我們需要處理的幀長度是 chunk_size + overlap_size
        self.overlap_size = self.chunk_size
        self.full_frame_size = 2 * self.chunk_size
        self.fft_size = self.full_frame_size

        # 降噪相關屬性
        self.noise_profile = np.zeros(self.fft_size // 2 + 1)
        self.learning_frames = learning_frames
        self.frames_learned = 0
        
        # 窗函數長度要與處理的幀長度一致
        self.window = windows.hann(self.full_frame_size)
        
        # 重疊緩衝區
        self.overlap_buffer = np.zeros(self.overlap_size)
        self.prev_processed_overlap = np.zeros(self.overlap_size)

    def process(self, audio_data, is_speech):
        # 如果還在學習階段，且沒有語音，則更新雜訊模型
        if not is_speech and self.frames_learned < self.learning_frames:
            # 這裡的學習邏輯也要用 full_frame
            full_frame = np.concatenate((self.overlap_buffer, audio_data))
            windowed_full_frame = full_frame * self.window
            fft_magnitude = np.abs(rfft(windowed_full_frame))
            
            # 簡單的平均化來學習雜訊
            self.noise_profile = (self.noise_profile * self.frames_learned + fft_magnitude) / (self.frames_learned + 1)
            self.frames_learned += 1
            print(f"[降噪] 正在學習雜訊模型... ({self.frames_learned}/{self.learning_frames})")

            # 在學習階段，我們直接返回原始的 audio_data
            return audio_data

        # 如果已經學習完畢，則應用降噪
        if self.frames_learned >= self.learning_frames:
            # 1. 準備輸入數據：將 audio_data 與上一幀的重疊緩衝區拼接
            full_frame = np.concatenate((self.overlap_buffer, audio_data))
            
            # 2. 應用窗函數
            windowed_data = full_frame * self.window
            
            # 3. 轉換到頻域
            fft_data = rfft(windowed_data)
            fft_magnitude = np.abs(fft_data)
            fft_phase = np.angle(fft_data)
            
            # 4. 頻譜減法
            # over_subtraction_factor = 1.2
            # snr = fft_magnitude / (self.noise_profile + 1e-10)
            # gain_mask = np.maximum(snr - over_subtraction_factor, 0) / (snr - over_subtraction_factor + 1e-10)
            # gain_mask = np.clip(gain_mask, 0, 1)
# 
            # processed_magnitude = fft_magnitude * gain_mask
            
            # Wiener Filter 增益掩碼
            alpha = 1.0  # 過度減法因子
            snr_posterior = fft_magnitude**2 / (self.noise_profile**2 + 1e-10)
            gain_mask_wiener = snr_posterior / (snr_posterior + alpha)
            processed_magnitude = fft_magnitude * gain_mask_wiener
            processed_fft = processed_magnitude * np.exp(1j * fft_phase)
            
            # 5. 轉換回時域
            processed_full_frame = irfft(processed_fft)
            
            # 6. 重疊相加
            # 建立一個新的輸出緩衝區，其長度為 chunk_size
            window_a = self.window[:self.chunk_size]
            window_b = self.window[self.chunk_size:]
            window_sum = window_a + window_b
            output_data = (
                processed_full_frame[:self.chunk_size] * window_a
                + self.prev_processed_overlap * window_b
            ) / window_sum
                        
            # 將當前幀處理後的非重疊區域直接放入輸出數據的後半部分
            self.prev_processed_overlap = processed_full_frame[self.chunk_size:]

            # 7. 更新緩衝區
            # 將處理後幀的後半部分（重疊區）存入緩衝區
            self.overlap_buffer = audio_data.copy()
            
            return output_data
        else:
            return audio_data
