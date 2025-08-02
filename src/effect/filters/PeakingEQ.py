# src/effect/filters/PeakingEQ.py

from ..base_effect import AudioEffect
import numpy as np

class PeakingEQ(AudioEffect):
    def __init__(self, samplerate, gain_db=0.0, center_freq=1000.0, Q=1.0):
        super().__init__()
        self.samplerate = samplerate
        self.gain_db = gain_db
        self.center_freq = center_freq
        self.Q = Q
        
        # 濾波器的係數
        self.b = np.array([0.0, 0.0, 0.0])
        self.a = np.array([0.0, 0.0, 0.0])
        
        # 延遲緩衝區
        self.x_delay = np.zeros(2)
        self.y_delay = np.zeros(2)
        
        self._calculate_coefficients()

    def _calculate_coefficients(self):
        """
        根據參數計算二階濾波器的係數 (biquad filter)。
        """
        A = 10**(self.gain_db / 40)
        omega = 2 * np.pi * self.center_freq / self.samplerate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2 * self.Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
        
        # 儲存標準化的係數
        self.b = np.array([b0/a0, b1/a0, b2/a0])
        self.a = np.array([1.0, a1/a0, a2/a0])
        
    def process(self, audio_data):
        """
        應用 IIR 濾波器到音訊數據。
        """
        output_data = np.zeros_like(audio_data)
        
        # 這是直接形式 II 濾波器的實現
        for i in range(len(audio_data)):
            # 計算當前輸出
            output = (self.b[0] * audio_data[i] + 
                      self.b[1] * self.x_delay[0] + 
                      self.b[2] * self.x_delay[1] -
                      self.a[1] * self.y_delay[0] -
                      self.a[2] * self.y_delay[1])
            
            output_data[i] = output
            
            # 更新延遲緩衝區
            self.x_delay[1] = self.x_delay[0]
            self.x_delay[0] = audio_data[i]
            self.y_delay[1] = self.y_delay[0]
            self.y_delay[0] = output
            
        return output_data

    def set_parameters(self, gain_db=None, center_freq=None, Q=None):
        """
        提供一個方法來動態修改濾波器參數。
        """
        if gain_db is not None:
            self.gain_db = gain_db
        if center_freq is not None:
            self.center_freq = center_freq
        if Q is not None:
            self.Q = Q
        
        self._calculate_coefficients()

# 輔助函式：將頻寬 (Hz) 轉換為 Q 值
def bw_to_q(bw, fc, samplerate):
    """
    將頻寬轉換為 Q 值。
    bw: 頻寬 (Hz)
    fc: 中心頻率 (Hz)
    samplerate: 採樣率
    """
    return fc / (samplerate * np.sin(np.pi * bw / samplerate))


# Example of how to use this class (for testing purposes, not part of GUI)
if __name__ == "__main__":
    import os
    import soundfile as sf
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    current_script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    music_dir = os.path.join(project_root, 'music')

    input_audio_file = os.path.join(music_dir, 'funky_guitar.wav')
    processed_music_dir = os.path.join(project_root, 'processed_audio')
    output_audio_file = os.path.join(processed_music_dir, 'processed_peakingeq_auto_comp.wav')

    print(f"Input file path: {input_audio_file}")
    print(f"Output file path: {output_audio_file}")

    print("Testing Peaking EQ:")
    
    try:
        data, samplerate = sf.read(input_audio_file, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono
        print(f"Successfully loaded audio from {input_audio_file} with sample rate {samplerate}")

        # --- Test with auto gain compensation ---
        # low=60-250HZ
        # middle=250-4000HZ
        # treble=4000-20000HZ
        print("\n--- Testing with Auto Gain Compensation ---")
        clipper_auto = PeakingEQ(samplerate=samplerate, gain_db=10.0, center_freq=200, Q=1.0)
        processed_audio_auto = clipper_auto.process(data)
        sf.write(output_audio_file, processed_audio_auto, samplerate)
        print("Processed audio with auto compensation saved to processed_peakingeq_auto_comp.wav")

        # # --- Test with manual postgain ---
        # print("\n--- Testing with Manual Postgain ---")
        # clipper_manual = Distortion(samplerate=samplerate, threshold=0.5, pregain=50.0, postgain=1.0, # postgain=1.0 will make it quiet
        #                             filter_cutoff_freq=5000, auto_gain_compensation=False)
        # processed_audio_manual = clipper_manual.process(data)
        # sf.write('processed_clipped_manual_low_postgain.wav', processed_audio_manual, samplerate)
        # print("Processed audio with manual low postgain saved to processed_clipped_manual_low_postgain.wav")

        # clipper_manual_boost = Distortion(samplerate=samplerate, threshold=0.5, pregain=50.0, postgain=50.0, # postgain=50.0 to compensate
        #                                   filter_cutoff_freq=5000, auto_gain_compensation=False)
        # processed_audio_manual_boost = clipper_manual_boost.process(data)
        # sf.write('processed_clipped_manual_high_postgain.wav', processed_audio_manual_boost, samplerate)
        # print("Processed audio with manual high postgain saved to processed_clipped_manual_high_postgain.wav")


    except FileNotFoundError:
        print(f"Error: File not found at {input_audio_file}. Please check the path and file existence.")
    except Exception as e:
        print(f"An error occurred: {e}")