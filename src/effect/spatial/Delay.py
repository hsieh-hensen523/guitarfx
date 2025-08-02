# effects/Delay.py
import numpy as np
import sys
from ..base_effect import AudioEffect


class Delay(AudioEffect):
    """
    Delay/Echo effect class, maintains internal state (delay buffer).
    """
    def __init__(self, samplerate, max_delay_time_s=2.0, delay_time_ms=300, feedback=0.5, dry_wet_mix=0.5):
        
        if not isinstance(samplerate, int) or samplerate <= 0:
            raise ValueError("Samplerate must be a positive integer.")
        self.samplerate = samplerate
        self.max_delay_samples = int(samplerate * max_delay_time_s)
        self.delay_buffer = np.zeros(self.max_delay_samples, dtype=np.float32)
        self.delay_buffer_pos = 0
        self.delay_time_ms = delay_time_ms
        self.feedback = feedback
        self.dry_wet_mix = dry_wet_mix
        
        self._validate_params()
        
    def _validate_params(self):
        """Internal method to validate parameters."""
        if self.delay_time_ms <= 0:
            print("Warning: Delay time must be greater than 0. Using default 300 ms.", file=sys.stderr)
            self.delay_time_ms = 300
        if not (0 <= self.feedback < 1.0):
            print("Warning: Feedback value must be between 0 (inclusive) and 1 (exclusive). Using default 0.5.", file=sys.stderr)
            self.feedback = 0.5
        if not (0 <= self.dry_wet_mix <= 1.0):
            print("Warning: Dry/Wet mix must be between 0 and 1. Using default 0.5.", file=sys.stderr)
            self.dry_wet_mix = 0.5
            
    def set_parameters(self, delay_time_ms=None, feedback=None, dry_wet_mix=None):
        """
        Updates the effect parameters.
        Only updates parameters that are explicitly provided.
        """
        if delay_time_ms is not None:
            self.delay_time_ms = delay_time_ms
        if feedback is not None:
            self.feedback = feedback
        if dry_wet_mix is not None:
            self.dry_wet_mix = dry_wet_mix
        self._validate_params()
        
        
    def process(self, audio_data):
        """
        Applies delay/echo effect.
        Args:
            audio_data (np.ndarray): Input audio data, should be mono (1D array).
        Returns:
            np.ndarray: Processed audio data.
        """

        delay_samples = int(self.samplerate * (self.delay_time_ms / 1000.0))
        
        if delay_samples >= self.max_delay_samples:
            print(f"Warning: Delay time too long ({self.delay_time_ms}ms), exceeds buffer capacity. Limiting delay to max buffer size.", file=sys.stderr)
            delay_samples = self.max_delay_samples - 1
            if delay_samples < 0: delay_samples = 0

        processed_audio_data = np.zeros_like(audio_data)
        
        for i in range(len(audio_data)):
            read_pos = (self.delay_buffer_pos - delay_samples + self.max_delay_samples) % self.max_delay_samples
            
            delayed_sample = self.delay_buffer[read_pos]
            current_input_sample = audio_data[i]
            
            # Sum current input and feedback for writing to delay buffer
            sample_to_write = current_input_sample + delayed_sample * self.feedback
            
            # Write to delay buffer
            self.delay_buffer[self.delay_buffer_pos] = sample_to_write

            # Advance write position
            self.delay_buffer_pos = (self.delay_buffer_pos + 1) % self.max_delay_samples

            # Mix dry and wet signals for output
            processed_audio_data[i] = (current_input_sample * (1 - self.dry_wet_mix) + 
                                       delayed_sample * self.dry_wet_mix)
        
        return processed_audio_data
    
    
# Example of how to use this class (for testing purposes)
if __name__ == "__main__":
    import os
    import soundfile as sf

    current_script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(os.path.join(current_script_dir, os.pardir),os.pardir))
    music_dir = os.path.join(parent_dir, 'music')
    processed_dir = os.path.join(parent_dir, 'processed_audio')

    input_audio_file = os.path.join(music_dir, 'funky_guitar.wav')

    print(f"Input file path: {input_audio_file}")
    
    try:
        data, samplerate = sf.read(input_audio_file, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono
        print(f"Successfully loaded audio from {input_audio_file} with sample rate {samplerate}")

        # --- Test with auto gain compensation ---
        print("\n--- Testing Delay with Auto Gain Compensation ---")
        delay_auto = Delay(samplerate=samplerate, max_delay_time_s=1, delay_time_ms=10, feedback=0.3, dry_wet_mix=0.5)
        processed_audio_auto = delay_auto.process(data)
        sf.write(os.path.join(processed_dir, 'processed_delay_auto_comp.wav'), processed_audio_auto, samplerate)
        print("Processed audio with auto compensation saved to processed_delay_auto_comp.wav")

        # # --- Test with manual postgain ---
        # print("\n--- Testing SoftClipping with Manual Postgain ---")
        # overdrive_manual = SoftClipping(samplerate=samplerate, pregain=20.0, postgain=0.1, # postgain=0.1 will make it quiet
        #                                 auto_gain_compensation=False)
        # processed_audio_manual = overdrive_manual.process(data)
        # sf.write('processed_overdrive_manual_low_postgain.wav', processed_audio_manual, samplerate)
        # print("Processed audio with manual low postgain saved to processed_overdrive_manual_low_postgain.wav")

        # overdrive_manual_boost = SoftClipping(samplerate=samplerate, pregain=20.0, postgain=0.8, # postgain=0.8 to compensate
        #                                       auto_gain_compensation=False)
        # processed_audio_manual_boost = overdrive_manual_boost.process(data)
        # sf.write('processed_overdrive_manual_high_postgain.wav', processed_audio_manual_boost, samplerate)
        # print("Processed audio with manual high postgain saved to processed_overdrive_manual_high_postgain.wav")


    except FileNotFoundError:
        print(f"Error: File not found at {input_audio_file}. Please check the path and file existence.")
    except Exception as e:
        print(f"An error occurred: {e}")