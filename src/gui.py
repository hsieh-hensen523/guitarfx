import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font 
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading 

# Assuming these are your effect classes and utilities
# 假設這是你的效果器類別和實用工具
from effect import Delay, Distortion, Overdrive, PeakingEQ
from utils.audio_utils import get_rms

# --- GUI Application Class ---

class AudioProcessorGUI:
    """
    A simple Tkinter GUI for applying and playing audio effects.
    一個簡單的 Tkinter GUI，用於應用和播放音訊效果。
    """
    def __init__(self, master):
        self.master = master
        master.title("Python Audio Effect Processor") 
        master.geometry("600x650") 
        master.resizable(False, False) 

        self.default_font = font.Font(family="Arial", size=10) 
        
        s = ttk.Style()
        s.theme_use('clam') 
        s.configure('.', font=self.default_font)
        s.configure('TLabel', font=self.default_font)
        s.configure('TButton', font=self.default_font)
        s.configure('TRadiobutton', font=self.default_font)
        s.configure('TScale', font=self.default_font)
        s.configure('TLabelframe.Label', font=self.default_font)

        self.original_audio_data = None
        self.samplerate = None
        self.processed_audio_data = None
        self.current_playback_stream = None 

        # The Delay instance is created inside _run_processing_in_thread,
        # so this line is redundant and can be removed.
        # Delay 實例在 _run_processing_in_thread 中創建，
        # 因此這一行是多餘的，可以移除。
        # self.delay_effect_instance = Delay(samplerate=44100) 

        self.effect_configs = {
            'none': {
                'label': 'No Effect', 
                'params': []
            },
            'hard_clipping': {
                'label': 'Hard Clipping (Distortion)',
                'func': Distortion,
                'params': [
                    {'id': 'threshold', 'label': 'Threshold', 'min': 0.01, 'max': 1.0, 'res': 0.01, 'default': 0.5},
                    {'id': 'pregain', 'label': 'Pregain', 'min': 1.0, 'max': 20.0, 'res': 0.1, 'default': 5.0},
                    {'id': 'postgain', 'label': 'Postgain', 'min': 0.1, 'max': 5.0, 'res': 0.01, 'default': 3.0},
                    {'id': 'filter_cutoff_freq', 'label': 'Filter Cutoff (Hz)', 'min': 500, 'max': 10000, 'res': 100, 'default': 3000}
                ]
            },
            'soft_clipping_tanh': {
                'label': 'Soft Clipping (Overdrive)', 
                'func': Overdrive,
                'params': [
                    {'id': 'pregain', 'label': 'Pregain', 'min': 1.0, 'max': 50.0, 'res': 0.1, 'default': 5.0},
                    {'id': 'postgain', 'label': 'Postgain', 'min': 0.1, 'max': 2.0, 'res': 0.01, 'default': 0.5}
                ]
            },
            'peaking_eq': {
                'label': 'Peaking Equalizer (EQ)',
                'func': PeakingEQ,
                'params': [
                    {'id': 'center_freq', 'label': 'Center Freq (Hz)', 'min': 50, 'max': 10000, 'res': 10, 'default': 1000},
                    {'id': 'gain_db', 'label': 'Gain (dB)', 'min': -15, 'max': 15, 'res': 0.1, 'default': 6.0},
                    {'id': 'Q', 'label': 'Q Value (Bandwidth)', 'min': 0.5, 'max': 10.0, 'res': 0.1, 'default': 1.0}
                ]
            },
            'delay_echo': {
                'label': 'Delay/Echo',
                'func': Delay, 
                'params': [
                    {'id': 'delay_time_ms', 'label': 'Delay Time (ms)', 'min': 10, 'max': 1000, 'res': 1, 'default': 300},
                    {'id': 'feedback', 'label': 'Feedback', 'min': 0.0, 'max': 0.95, 'res': 0.01, 'default': 0.6},
                    {'id': 'dry_wet_mix', 'label': 'Dry/Wet Mix', 'min': 0.0, 'max': 1.0, 'res': 0.01, 'default': 0.7}
                ]
            }
        }
        self.current_effect_var = tk.StringVar(value='none')
        self.current_params = {} 
        self.param_labels = {} # Using a separate dictionary for labels for cleaner code

        self._create_widgets()
        self._update_parameters_ui() 

    def _create_widgets(self):
        file_frame = ttk.LabelFrame(self.master, text="Audio File", padding="10")
        file_frame.pack(padx=10, pady=10, fill="x")

        self.file_path_label = ttk.Label(file_frame, text="No file selected")
        self.file_path_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Button(file_frame, text="Select File", command=self._load_audio_file).pack(side="right")

        effect_frame = ttk.LabelFrame(self.master, text="Select Effect", padding="10")
        effect_frame.pack(padx=10, pady=10, fill="x")

        for effect_id, config in self.effect_configs.items():
            ttk.Radiobutton(effect_frame, text=config['label'], variable=self.current_effect_var,
                            value=effect_id, command=self._update_parameters_ui).pack(anchor="w", pady=2)
        
        self.params_frame = ttk.LabelFrame(self.master, text="Effect Parameters", padding="10")
        self.params_frame.pack(padx=10, pady=10, fill="x")

        self.process_button = ttk.Button(self.master, text="Process and Play", command=self._process_audio, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.playback_frame = ttk.Frame(self.master)
        self.playback_frame.pack(pady=5)
        ttk.Button(self.playback_frame, text="Play Original Audio", command=self._play_original_audio, state=tk.DISABLED).pack(side="left", padx=5)
        ttk.Button(self.playback_frame, text="Play Processed Audio", command=self._play_processed_audio, state=tk.DISABLED).pack(side="left", padx=5)
        ttk.Button(self.playback_frame, text="Stop Playback", command=self._stop_audio, state=tk.DISABLED).pack(side="left", padx=5)

    def _load_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                data, samplerate = sf.read(file_path, dtype='float32')
                
                if data.ndim > 1:
                    data = data.mean(axis=1)
                    messagebox.showinfo("Audio Loaded", "Stereo audio detected, converted to mono.")
                
                self.original_audio_data = data
                self.samplerate = samplerate
                self.file_path_label.config(text=f"File: {file_path.split('/')[-1]} ({samplerate} Hz)")
                self.process_button.config(state=tk.NORMAL)
                self.playback_frame.winfo_children()[0].config(state=tk.NORMAL) 
                
                # The Delay instance is now handled correctly in the processing thread,
                # so this line is also redundant.
                # Delay 實例現在已在處理執行緒中正確處理，因此此行也是多餘的。
                # self.delay_effect_instance = Delay(samplerate=self.samplerate) 
                
                messagebox.showinfo("Audio Loaded", "Audio file loaded successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file: {e}")
                self.original_audio_data = None
                self.samplerate = None
                self.file_path_label.config(text="No file selected")
                self.process_button.config(state=tk.DISABLED)
                self.playback_frame.winfo_children()[0].config(state=tk.DISABLED)

    def _update_parameters_ui(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        selected_effect_id = self.current_effect_var.get()
        config = self.effect_configs[selected_effect_id]
        
        self.current_params = {} 
        self.param_labels = {}

        if not config['params']:
            ttk.Label(self.params_frame, text="This effect has no adjustable parameters.").pack(pady=10)
        else:
            for param in config['params']:
                frame = ttk.Frame(self.params_frame)
                frame.pack(fill="x", pady=5)

                ttk.Label(frame, text=f"{param['label']}:").pack(side="left", padx=(0, 10))
                
                param_var = tk.DoubleVar(value=param['default'])
                self.current_params[param['id']] = param_var 

                value_label = ttk.Label(frame, text=f"{param['default']:.2f}")
                value_label.pack(side="left", padx=(10, 0))
                self.param_labels[param['id']] = value_label # Store label in a separate dictionary for clarity
                
                slider = ttk.Scale(frame, from_=param['min'], to=param['max'],
                                    orient="horizontal", variable=param_var,
                                    command=lambda val, pv=param_var, id=param['id']: self._update_param_value_label(pv, id))
                slider.set(param['default']) 
                slider.pack(side="left", fill="x", expand=True)

    def _update_param_value_label(self, param_var, param_id):
        # Now we retrieve the label from the new, separate dictionary
        # 現在我們從新的、獨立的字典中檢索標籤
        value_label = self.param_labels[param_id]
        value_label.config(text=f"{param_var.get():.2f}")

    def _process_audio(self):
        if self.original_audio_data is None or self.samplerate is None:
            messagebox.showwarning("Warning", "Please load an audio file first!")
            return

        self.process_button.config(state=tk.DISABLED, text="Processing...")
        self.master.update_idletasks() 

        processing_thread = threading.Thread(target=self._run_processing_in_thread)
        processing_thread.start()

    def _run_processing_in_thread(self):
        """
        Executes audio processing in a separate thread to prevent the GUI from freezing.
        This version instantiates the effect and then calls its set_parameters
        method with a dynamically built parameter dictionary.
        """
        if self.original_audio_data is None or self.samplerate is None:
            messagebox.showwarning("Warning", "Please load an audio file first!")
            self.process_button.config(state=tk.NORMAL, text="Process and Play")
            return

        self.process_button.config(state=tk.DISABLED, text="Processing...")
        self.master.update_idletasks()

        try:
            selected_effect_id = self.current_effect_var.get()

            if selected_effect_id == 'none':
                self.processed_audio_data = self.original_audio_data.copy()
                messagebox.showinfo("Processing Complete", "Audio processing finished! (No effect applied)")
                self.playback_frame.winfo_children()[1].config(state=tk.NORMAL)
                self.playback_frame.winfo_children()[2].config(state=tk.NORMAL)
                return

            effect_class = self.effect_configs[selected_effect_id]['func']
            effect_instance = effect_class(samplerate=self.samplerate)

            params_to_pass = {}
            for param_config in self.effect_configs[selected_effect_id]['params']:
                param_id = param_config['id']
                params_to_pass[param_id] = self.current_params[param_id].get()
            
            if hasattr(effect_instance, 'set_parameters') and callable(getattr(effect_instance, 'set_parameters')):
                effect_instance.set_parameters(**params_to_pass)
            else:
                for param_id, value in params_to_pass.items():
                    setattr(effect_instance, param_id, value)

            self.processed_audio_data = effect_instance.process(self.original_audio_data)

            messagebox.showinfo("Processing Complete", "Audio processing finished!")
            self.playback_frame.winfo_children()[1].config(state=tk.NORMAL)
            self.playback_frame.winfo_children()[2].config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during audio processing: {e}")
            self.processed_audio_data = None
        finally:
            self.process_button.config(state=tk.NORMAL, text="Process and Play")

    def _play_audio_data(self, audio_data):
        if audio_data is None or self.samplerate is None:
            messagebox.showwarning("Warning", "No audio data to play.")
            return
        
        self._stop_audio() 

        self.current_playback_stream = sd.play(audio_data, self.samplerate)

    def _play_original_audio(self):
        self._play_audio_data(self.original_audio_data)

    def _play_processed_audio(self):
        self._play_audio_data(self.processed_audio_data)

    def _stop_audio(self):
        sd.stop()
        print("Audio playback stopped.")
        self.current_playback_stream = None 

# --- Main Program Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessorGUI(root)
    root.mainloop()

