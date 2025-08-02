# effects/Overdrive.py

import numpy as np
import sys # for stderr output

class Overdrive:
    """
    A class for applying a soft clipping (overdrive) effect using the tanh function.
    Now includes optional automatic gain compensation based on RMS.
    """
    def __init__(self, samplerate, pregain=5.0, postgain=0.5, auto_gain_compensation=False):
        """
        Initializes the Overdrive effect.

        Args:
            samplerate (int): The sample rate of the audio data.
            pregain (float): Gain applied before the tanh function.
            postgain (float): Gain applied after the tanh function (used if auto_gain_compensation is False).
            auto_gain_compensation (bool): If True, automatically adjusts postgain to match input RMS.
        """
        if not isinstance(samplerate, int) or samplerate <= 0:
            raise ValueError("Samplerate must be a positive integer.")

        self.samplerate = samplerate
        self.pregain = pregain
        self.postgain = postgain # This will be overridden if auto_gain_compensation is True
        self.auto_gain_compensation = auto_gain_compensation
        
        self._validate_params()

    def _validate_params(self):
        """Internal method to validate parameters."""
        if self.pregain <= 0:
            print("Warning: Pregain must be greater than 0. Adjusting to 5.0.", file=sys.stderr)
            self.pregain = 5.0
        if self.postgain <= 0: # Only validate if not in auto mode, or if it's explicitly set
            print("Warning: Postgain must be greater than 0. Adjusting to 0.5.", file=sys.stderr)
            self.postgain = 0.5

    def set_parameters(self, pregain=None, postgain=None, auto_gain_compensation=None):
        """
        Updates the effect parameters.
        Only updates parameters that are explicitly provided.
        """
        if pregain is not None:
            self.pregain = pregain
        if postgain is not None:
            self.postgain = postgain
        if auto_gain_compensation is not None:
            self.auto_gain_compensation = auto_gain_compensation
        self._validate_params() # Re-validate after update

    def process(self, audio_data):
        """
        Applies the soft clipping effect to the audio data.

        Args:
            audio_data (np.ndarray): Input audio data (1D array for mono).

        Returns:
            np.ndarray: Processed audio data.
        """
        if self.samplerate == 0:
            print("Error: Samplerate is 0 in SoftClipping. Cannot process.", file=sys.stderr)
            return audio_data

        # Calculate input RMS for auto gain compensation
        input_rms = np.sqrt(np.mean(audio_data**2))
        
        gained_data = audio_data * self.pregain
        processed_data = np.tanh(gained_data) # Apply tanh function
        
        # Calculate RMS of the tanh'd data (before final scaling)
        processed_rms = np.sqrt(np.mean(processed_data**2))

        # Determine the final scaling factor
        final_scale_factor = 1.0 # Default if no auto compensation or if input is silent

        if self.auto_gain_compensation and input_rms > 1e-9: # Avoid division by zero for silence
            if processed_rms > 1e-9: # Avoid division by zero if processed data is silent
                # Compensation factor to bring processed_rms to input_rms
                compensation_factor_auto = input_rms / processed_rms
                # Cap the auto compensation factor to prevent excessive boosting of very quiet distorted signals
                compensation_factor_auto = min(compensation_factor_auto, 100.0) # Cap at 100x boost
                final_scale_factor = compensation_factor_auto
            else:
                final_scale_factor = 0.0 # If processed data is silent, output silent
        else:
            final_scale_factor = self.postgain # Use manual postgain

        processed_data = processed_data * final_scale_factor
        processed_data = np.clip(processed_data, -1.0, 1.0) # Final clamping to ensure range
        
        return processed_data

# Example of how to use this class (for testing purposes)
if __name__ == "__main__":
    import os
    import soundfile as sf

    current_script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    music_dir = os.path.join(parent_dir, 'music')

    input_audio_file = os.path.join(music_dir, 'funky_guitar.wav')

    print(f"Input file path: {input_audio_file}")
    
    try:
        data, samplerate = sf.read(input_audio_file, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1) # Convert to mono
        print(f"Successfully loaded audio from {input_audio_file} with sample rate {samplerate}")

        # --- Test with auto gain compensation ---
        print("\n--- Testing SoftClipping with Auto Gain Compensation ---")
        overdrive_auto = Overdrive(samplerate=samplerate, pregain=15.0, auto_gain_compensation=True)
        processed_audio_auto = overdrive_auto.process(data)
        sf.write('processed_overdrive_auto_comp.wav', processed_audio_auto, samplerate)
        print("Processed audio with auto compensation saved to processed_overdrive_auto_comp.wav")

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
