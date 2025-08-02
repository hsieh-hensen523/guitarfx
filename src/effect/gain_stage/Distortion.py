# effects/Distortion.py

from ..filters.Filters import HighpassFilter
from ..base_effect import AudioEffect
import soundfile as sf
import numpy as np
import sys
import os

class Distortion(AudioEffect):
    """
    A class for applying a hard clipping (distortion) effect to audio data.
    Now includes optional automatic gain compensation based on RMS.
    """
    def __init__(self, samplerate, threshold=0.5, pregain=5.0, postgain=3.0, 
                 filter_cutoff_freq=3000, filter_order=5, auto_gain_compensation=False):
        """
        Initializes the HardClippingEffect.

        Args:
            samplerate (int): The sample rate of the audio data.
            threshold (float): The clipping threshold (0.01 to 1.0).
            pregain (float): Gain applied before clipping.
            postgain (float): Gain applied after clipping.
            filter_cutoff_freq (float): Cutoff frequency for an optional post-clipping low-pass filter.
            filter_order (int): Order of the low-pass filter.
            auto_gain_compensation (bool): If True, automatically adjusts postgain to match input RMS.
        """
        if not isinstance(samplerate, int) or samplerate <= 0:
            raise ValueError("Samplerate must be a positive integer.")
            
        self.samplerate = samplerate
        self.threshold = threshold
        self.pregain = pregain
        self.postgain = postgain
        self.filter_cutoff_freq = filter_cutoff_freq
        self.filter_order = filter_order
        self.auto_gain_compensation = auto_gain_compensation
        self.lfilter_state = None  # Initialize filter state
        self.highpass_filter = HighpassFilter(samplerate, cutoff_freq=filter_cutoff_freq, order=filter_order) if filter_cutoff_freq else None
        
        # Validate initial parameters
        self._validate_params()

    def _validate_params(self):
        """Internal method to validate parameters."""
        if not (0.0 < self.threshold <= 1.0):
            print("Warning: Clipping threshold must be between 0.0 and 1.0 (exclusive of 0.0). Adjusting to 0.5.", file=sys.stderr)
            self.threshold = 0.5
        if self.pregain <= 0:
            print("Warning: Pregain must be greater than 0. Adjusting to 5.0.", file=sys.stderr)
            self.pregain = 5.0
        if self.postgain <= 0:
            print("Warning: Postgain must be greater than 0. Adjusting to 3.0.", file=sys.stderr)
            self.postgain = 3.0
        if self.filter_cutoff_freq is not None and not (0 < self.filter_cutoff_freq < self.samplerate / 2):
             print(f"Warning: Filter cutoff frequency {self.filter_cutoff_freq} Hz is out of valid range (0 to {self.samplerate/2} Hz). Disabling filter.", file=sys.stderr)
             self.filter_cutoff_freq = None # Disable filter if invalid

    def set_parameters(self, threshold=None, pregain=None, postgain=None, filter_cutoff_freq=None, filter_order=None, auto_gain_compensation=None):
        """
        Updates the effect parameters.
        Only updates parameters that are explicitly provided.
        """
        if threshold is not None:
            self.threshold = threshold
        if pregain is not None:
            self.pregain = pregain
        if postgain is not None:
            self.postgain = postgain
        if filter_cutoff_freq is not None:
            self.filter_cutoff_freq = filter_cutoff_freq
        if filter_order is not None:
            self.filter_order = filter_order
        if auto_gain_compensation is not None:
            self.auto_gain_compensation = auto_gain_compensation
        self._validate_params() # Re-validate after update

    def process(self, audio_data):
        """
        Applies the hard clipping effect to the audio data.

        Args:
            audio_data (np.ndarray): Input audio data (1D array for mono).

        Returns:
            np.ndarray: Processed audio data.
        """
        if self.samplerate == 0:
            print("Error: Samplerate is 0 in Distortion. Cannot process.", file=sys.stderr)
            return audio_data # Return original data if samplerate is invalid

        # Calculate input RMS for auto gain compensation
        input_rms = np.sqrt(np.mean(audio_data**2))
        
        gained_data = audio_data * self.pregain
        clipped_data = np.copy(gained_data)
        clipped_data[clipped_data > self.threshold] = self.threshold
        clipped_data[clipped_data < -self.threshold] = -self.threshold
        
        # Calculate RMS of the clipped data (before final scaling)
        clipped_rms = np.sqrt(np.mean(clipped_data**2))

        # Determine the final scaling factor
        final_scale_factor = 1.0 # Default if no auto compensation or if input is silent

        if self.auto_gain_compensation and input_rms > 1e-9: # Avoid division by zero for silence
            """Calculate compensation needed to bring clipped_data's RMS to input_rms
            Note: We are already dividing by self.pregain, so the compensation
            needs to be applied to the 'clipped_data / self.pregain' part.
            The goal is (clipped_data / self.pregain) * compensation_factor_auto = target_rms_level
            Where target_rms_level is proportional to input_rms
            
            The current approach: `processed_data = clipped_data / self.pregain * self.postgain`
            If we want output_rms ~= input_rms, then:
            output_rms = RMS(clipped_data / self.pregain * compensation_factor)
            output_rms = RMS(clipped_data) / self.pregain * compensation_factor
            So, input_rms = clipped_rms / self.pregain * compensation_factor
            compensation_factor = (input_rms * self.pregain) / clipped_rms"""
            
            if clipped_rms > 1e-9: # Avoid division by zero if clipped data is silent
                # This compensation factor is applied on top of the 1/pregain scaling
                compensation_factor_auto = (input_rms * self.pregain) / clipped_rms
                # We need to be careful not to make the signal too loud.
                # A common practice is to limit this compensation factor or clamp the final signal.
                # Let's cap the auto compensation factor to prevent excessive boosting of very quiet distorted signals
                compensation_factor_auto = min(compensation_factor_auto, 100.0) # Cap at 100x boost
                final_scale_factor = compensation_factor_auto
            else:
                final_scale_factor = 0.0 # If clipped data is silent, output silent
        else:
            final_scale_factor = self.postgain # Use manual postgain

        processed_data = clipped_data / self.pregain * final_scale_factor
        processed_data = np.clip(processed_data, -1.0, 1.0) # Final clamping to ensure range

        if self.filter_cutoff_freq is not None:
            processed_data = self.highpass_filter.process(processed_data)

        return processed_data

# Example of how to use this class (for testing purposes, not part of GUI)
if __name__ == "__main__":
    # Dummy input audio file for testing
    # 獲取當前腳本的目錄
    current_script_dir = os.path.dirname(__file__)

    # 向上走一級目錄 (your_project_root)
    parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))

    # 構建到 music 資料夾的路徑
    music_dir = os.path.join(parent_dir, 'music')

    # 構建完整的輸入和輸出檔案路徑
    input_audio_file = os.path.join(music_dir, 'funky_guitar.wav')
    output_hard_clipped_file = os.path.join(music_dir, 'output_hard_clipped.wav')

    print(f"Input file path: {input_audio_file}")
    print(f"Output file path: {output_hard_clipped_file}")

    print("Testing Distortion:")
    
    try:
        data, samplerate = sf.read(input_audio_file, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono
        print(f"Successfully loaded audio from {input_audio_file} with sample rate {samplerate}")

        # --- Test with auto gain compensation ---
        print("\n--- Testing with Auto Gain Compensation ---")
        clipper_auto = Distortion(samplerate=samplerate, threshold=0.6, pregain=50.0, 
                                  filter_cutoff_freq=12000, auto_gain_compensation=True)
        processed_audio_auto = clipper_auto.process(data)
        sf.write('processed_distortion_auto_comp.wav', processed_audio_auto, samplerate)
        print("Processed audio with auto compensation saved to processed_distortion_auto_comp.wav")

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