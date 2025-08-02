# src/effect/filters/ThreeBandEQ.py

from ..base_effect import AudioEffect
from .PeakingEQ import PeakingEQ

class ThreeBandEQ(AudioEffect):
    def __init__(self, samplerate, low_gain_db=120, mid_gain_db=1000, high_gain_db=8000):
        self.low_eq = PeakingEQ(samplerate, center_freq=low_gain_db)
        self.mid_eq = PeakingEQ(samplerate, center_freq=mid_gain_db)
        self.high_eq = PeakingEQ(samplerate, center_freq=high_gain_db)

    def process(self, audio_data):
        output_data = audio_data.copy()

        for i in range(len(output_data)):
            # 1. 低頻 EQ 處理
            x_in_low = output_data[i]
            y_out_low = (self.low_eq.b[0] * x_in_low + 
                         self.low_eq.b[1] * self.low_eq.x_delay[0] + 
                         self.low_eq.b[2] * self.low_eq.x_delay[1] -
                         self.low_eq.a[1] * self.low_eq.y_delay[0] -
                         self.low_eq.a[2] * self.low_eq.y_delay[1])
            # 更新低頻 EQ 的延遲緩衝區
            self.low_eq.x_delay[1] = self.low_eq.x_delay[0]
            self.low_eq.x_delay[0] = x_in_low
            self.low_eq.y_delay[1] = self.low_eq.y_delay[0]
            self.low_eq.y_delay[0] = y_out_low

            # 2. 中頻 EQ 處理，輸入是 y_out_low
            x_in_mid = y_out_low
            y_out_mid = (self.mid_eq.b[0] * x_in_mid + 
                         self.mid_eq.b[1] * self.mid_eq.x_delay[0] + 
                         self.mid_eq.b[2] * self.mid_eq.x_delay[1] -
                         self.mid_eq.a[1] * self.mid_eq.y_delay[0] -
                         self.mid_eq.a[2] * self.mid_eq.y_delay[1])
            # 更新中頻 EQ 的延遲緩衝區
            self.mid_eq.x_delay[1] = self.mid_eq.x_delay[0]
            self.mid_eq.x_delay[0] = x_in_mid
            self.mid_eq.y_delay[1] = self.mid_eq.y_delay[0]
            self.mid_eq.y_delay[0] = y_out_mid

            # 3. 高頻 EQ 處理，輸入是 y_out_mid
            x_in_high = y_out_mid
            y_out_high = (self.high_eq.b[0] * x_in_high + 
                          self.high_eq.b[1] * self.high_eq.x_delay[0] + 
                          self.high_eq.b[2] * self.high_eq.x_delay[1] -
                          self.high_eq.a[1] * self.high_eq.y_delay[0] -
                          self.high_eq.a[2] * self.high_eq.y_delay[1])
            # 更新高頻 EQ 的延遲緩衝區
            self.high_eq.x_delay[1] = self.high_eq.x_delay[0]
            self.high_eq.x_delay[0] = x_in_high
            self.high_eq.y_delay[1] = self.high_eq.y_delay[0]
            self.high_eq.y_delay[0] = y_out_high

            # 將最終輸出寫入
            output_data[i] = y_out_high

        return output_data

    def set_parameters(self, low_gain_db=0.0, mid_gain_db=0.0, high_gain_db=0.0):
        # 簡化參數設定
        self.low_eq.set_parameters(gain_db=low_gain_db)
        self.mid_eq.set_parameters(gain_db=mid_gain_db)
        self.high_eq.set_parameters(gain_db=high_gain_db)
        
        
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
    output_audio_file = os.path.join(processed_music_dir, 'processed_threebandeq_auto_comp.wav')

    print(f"Input file path: {input_audio_file}")
    print(f"Output file path: {output_audio_file}")

    print("Testing Three Band EQ:")
    
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
        clipper_auto = ThreeBandEQ(samplerate=samplerate)
        clipper_auto.set_gains(low_gain_db=-10.0, mid_gain_db=5.0, high_gain_db=10.0)
        processed_audio_auto = clipper_auto.process(data)
        sf.write(output_audio_file, processed_audio_auto, samplerate)
        print("Processed audio with auto compensation saved to processed_threebandeq_auto_comp.wav")

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