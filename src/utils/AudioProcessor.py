# src/utils/AudioProcessor.py

import pyaudio
import numpy as np
from utils.audio_utils import  get_rms, is_pop_noise, spectral_flatness
from effect import BandpassFilter, NoiseReduction, ConvolutionIR
import queue
import webrtcvad
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
ir_path = os.path.join(current_dir, "..", "..", "irs", "DYN-57", "OD-M212-P50-DYN-57-P05-10.wav")

class AudioProcessor:
    def __init__(self, samplerate, lowcut, highcut, order=5, chunk_size=1024, vad_mode=0, gain=1.0):
        # 初始化常數
        self.samplerate = samplerate
        self.gain = gain
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.chunk_size = chunk_size

        # 語音偵測相關狀態
        self.vad = webrtcvad.Vad(vad_mode)
        self.speech_frame_count = 0
        self.speech_started = False
        self.SPEECH_FRAME_THRESHOLD = 5  # 你可以設定一個適合的值
        
        # 降躁處理
        self.noise = NoiseReduction(
            samplerate=self.samplerate,
            chunk_size=self.chunk_size,
            learning_frames=10  # 你可以根據需要調整學習幀數
        )
        
        # 增加音箱模擬效果
        self.convolution_ir = ConvolutionIR(
            samplerate=self.samplerate,
            ir_path=ir_path
        )

        # 爆音偵測相關狀態
        self.pop_noise_silence_counter = 0
        self.MAX_VOLUME_RMS = 0.05  # 你需要根據測試調整這個值
        self.POP_SILENCE_FRAMES = 10 # 爆音後靜音的幀數
        
        # 濾波器
        self.bandpass_filter = BandpassFilter(
            samplerate=self.samplerate,
            lowcut=self.lowcut,
            highcut=self.highcut,
            order=self.order
        )
        
        # 資料隊列 (如果需要在其他地方存取)
        self.audio_queue = queue.Queue()

    def audio_callback(self, in_data, frame_count, time_info, status):
        # 1. 數據轉換與前置濾波
        int16_array = np.frombuffer(in_data, dtype=np.int16)
        float32_array = int16_array.astype(np.float32) / 32768.0

        # 這裡可以保留 processed_array 作為原始數據，供FFT分析
        processed_array = float32_array

        # 應用 bandpass 濾波器
        filtered_array = self.bandpass_filter.process(float32_array)

        # 2. 語音偵測 (VAD)
        # VAD判斷需要16bit的數據，所以這裡先轉換一次
        vad_check_array = np.clip(filtered_array * 32768.0, -32768, 32767).astype(np.int16)
        vad_check_bytes = vad_check_array.tobytes()
        is_speech_in_this_frame = self.vad.is_speech(vad_check_bytes, self.samplerate)

        if is_speech_in_this_frame:
            self.speech_frame_count += 1
            if self.speech_frame_count >= self.SPEECH_FRAME_THRESHOLD and not self.speech_started:
                self.speech_started = True
                print("[語音偵測] 有人在說話")
        else:
            self.speech_frame_count = max(0, self.speech_frame_count - 1)
            if self.speech_frame_count == 0 and self.speech_started:
                self.speech_started = False
                print("[語音偵測] 語音結束")


        # 3. 應用降噪
        is_speech = self.speech_started or is_speech_in_this_frame
        filtered_array = self.noise.process(filtered_array, is_speech)

        # 將處理後的音訊數據放入隊列
        self.audio_queue.put(filtered_array.copy())

        # 3. 爆音偵測與靜音狀態
        fft_magnitude = np.abs(np.fft.rfft(processed_array))
        rms = get_rms(filtered_array)

        if (
            is_pop_noise(fft_magnitude, self.samplerate)
            and rms > self.MAX_VOLUME_RMS
        ):
            print(f"[爆音偵測] RMS={rms:.4f}")
            self.pop_noise_silence_counter = self.POP_SILENCE_FRAMES
            print("→ 進入爆音靜音狀態")

        # 4. 決策輸出
        # 優先檢查爆音靜音狀態
        if self.pop_noise_silence_counter > 0:
            self.pop_noise_silence_counter -= 1
            silent_array = np.zeros_like(int16_array, dtype=np.int16)
            print("輸出靜音數據 (爆音)")
            return (silent_array.tobytes(), pyaudio.paContinue)

        # 接著檢查是否有語音，並決定是否應用增益或靜音
        if self.speech_started:
            # 如果有語音，應用增益並正常輸出
            # gained_array = self.convolution_ir.process(filtered_array * self.gain)
            gained_array = filtered_array * self.gain
            output_int16_array = np.clip(gained_array * 32768.0, -32768, 32767).astype(np.int16)
            # print("輸出處理後的音訊數據")
            return (output_int16_array.tobytes(), pyaudio.paContinue)
        else:
            # 如果沒有語音，回傳靜音數據 (這就是Noise Gate)
            # 即使 filtered_array 還有微小雜音，這裡也直接靜音
            silent_array = np.zeros_like(int16_array, dtype=np.int16)
            print("輸出靜音數據 (無語音)")
            return (silent_array.tobytes(), pyaudio.paContinue)
    
    