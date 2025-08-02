import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font 
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading 
import uuid
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Assuming these are your effect classes and utilities
# 假設這是你的效果器類別和實用工具
from effect import *
from utils.audio_utils import get_rms

# --- GUI Application Class ---

class AudioProcessorGUI:
    """
    A simple Tkinter GUI for applying and playing audio effects.
    Now with a drag-and-drop effect chain.
    """
    def __init__(self, master):
        self.master = master
        master.title("Python Audio Effect Processor")
        master.geometry("1000x800")
        master.resizable(True, True)

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

        self.effect_configs = {
            'hard_clipping': {
                'label': 'Hard Clipping (Distortion)',
                'func': Distortion,
                'params': [
                    {'id': 'threshold', 'label': 'Threshold', 'min': 0.01, 'max': 1.0, 'res': 0.01, 'default': 0.5},
                    {'id': 'pregain', 'label': 'Pregain', 'min': 1.0, 'max': 20.0, 'res': 0.1, 'default': 5.0},
                    {'id': 'postgain', 'label': 'Postgain', 'min': 0.1, 'max': 5.0, 'res': 0.01, 'default': 3.0},
                    {'id': 'filter_cutoff_freq', 'label': 'LPF Cutoff (Hz)', 'min': 500, 'max': 10000, 'res': 100, 'default': 3000}
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
            'threeband_eq': {
                'label': 'Three Band Equalizer (EQ)',
                'func': ThreeBandEQ,
                'params': [
                    {'id': 'low_gain_db', 'label': 'Low Gain (dB)', 'min': -15, 'max': 15, 'res': 0.1, 'default': 0.0},
                    {'id': 'mid_gain_db', 'label': 'Mid Gain (dB)', 'min': -15, 'max': 15, 'res': 0.1, 'default': 0.0},
                    {'id': 'high_gain_db', 'label': 'High Gain (dB)', 'min': -15, 'max': 15, 'res': 0.1, 'default': 0.0}
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
        
        # New data structure to hold the effect chain
        # 新的數據結構來儲存效果器鏈
        self.effect_chain = []
        self.param_vars = {} # Stores tk.DoubleVar instances by effect UUID
        self.drag_item_index = None # For drag-and-drop functionality
        self.selected_chain_item = None # Stores the UUID of the selected effect in the chain
        self.sine_wave_samplerate = 44100 # 標準正弦波的取樣率
        self.sine_wave_freq = 440 # 正弦波頻率 (440 Hz)

        self._create_widgets()
        # 初始繪製，顯示單一正弦波
        self._update_sine_wave_plot()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Top: File Selection and Master Gain ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=5)

        file_frame = ttk.LabelFrame(top_frame, text="Audio File", padding="10")
        file_frame.pack(side="left", padx=(0, 10), fill="x", expand=True)
        self.file_path_label = ttk.Label(file_frame, text="No file selected")
        self.file_path_label.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(file_frame, text="Select File", command=self._load_audio_file).pack(side="right")

        master_gain_frame = ttk.LabelFrame(top_frame, text="Master Gain", padding="10")
        master_gain_frame.pack(side="right", padx=(10, 0), fill="x", expand=True)
        self.gain_var = tk.DoubleVar(value=1.0)
        self.gain_label = ttk.Label(master_gain_frame, text=f"Gain: {self.gain_var.get():.2f}")
        self.gain_label.pack(side="left", padx=(0, 10))
        self.gain_slider = ttk.Scale(master_gain_frame, from_=0.0, to=5.0, orient="horizontal", variable=self.gain_var,
                                     command=lambda val: self.gain_label.config(text=f"Gain: {float(val):.2f}"))
        self.gain_slider.pack(side="left", fill="x", expand=True)
        
        # Waveform Plot Frame
        # 波形圖面板
        plot_frame = ttk.LabelFrame(top_frame, text="Waveform Display", padding="10")
        plot_frame.pack(side="top", pady=10, fill="both", expand=True)

        self.fig = Figure(figsize=(8, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.grid(False)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # --- Middle: Effect Selection and Chain ---
        chain_frame = ttk.LabelFrame(main_frame, text="Effect Chain", padding="10")
        chain_frame.pack(pady=10, fill="both", expand=True)
        
        middle_frame = ttk.Frame(chain_frame)
        middle_frame.pack(fill="both", expand=True)

        # Available Effects Listbox
        available_effects_frame = ttk.LabelFrame(middle_frame, text="Available Effects", padding="5")
        available_effects_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.available_listbox = tk.Listbox(available_effects_frame, selectmode=tk.SINGLE, height=15)
        self.available_listbox.pack(side="left", fill="both", expand=True)
        available_scrollbar = ttk.Scrollbar(available_effects_frame, orient=tk.VERTICAL, command=self.available_listbox.yview)
        available_scrollbar.pack(side="right", fill=tk.Y)
        self.available_listbox.config(yscrollcommand=available_scrollbar.set)
        
        # Populate available listbox
        for effect_id, config in self.effect_configs.items():
            self.available_listbox.insert(tk.END, config['label'])
        
        # Buttons Frame
        button_frame = ttk.Frame(middle_frame, width=50)
        button_frame.pack(side="left", fill="y", padx=10)
        ttk.Button(button_frame, text="Add >", command=self._add_effect).pack(pady=(100, 5), fill="x")
        ttk.Button(button_frame, text="< Remove", command=self._remove_effect).pack(pady=5, fill="x")

        # Effect Chain Listbox
        chain_listbox_frame = ttk.LabelFrame(middle_frame, text="Chain (Drag to reorder)", padding="5")
        chain_listbox_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.chain_listbox = tk.Listbox(chain_listbox_frame, selectmode=tk.SINGLE, height=15)
        self.chain_listbox.pack(side="left", fill="both", expand=True)
        chain_scrollbar = ttk.Scrollbar(chain_listbox_frame, orient=tk.VERTICAL, command=self.chain_listbox.yview)
        chain_scrollbar.pack(side="right", fill=tk.Y)
        self.chain_listbox.config(yscrollcommand=chain_scrollbar.set)

        # Bind events for drag-and-drop and parameter display
        self.chain_listbox.bind("<Button-1>", self._on_drag_start)
        self.chain_listbox.bind("<B1-Motion>", self._on_drag_motion)
        self.chain_listbox.bind("<ButtonRelease-1>", self._on_drag_stop)
        self.chain_listbox.bind("<<ListboxSelect>>", self._on_chain_select)
        
        self.params_frame = ttk.LabelFrame(main_frame, text="Effect Parameters", padding="10")
        self.params_frame.pack(padx=10, pady=10, fill="x")

        # --- Bottom: Processing and Playback Controls ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="x", pady=5)

        self.process_button = ttk.Button(bottom_frame, text="Process and Play", command=self._process_audio, state=tk.DISABLED)
        self.process_button.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.playback_frame = ttk.Frame(bottom_frame)
        self.playback_frame.pack(side="left", fill="x", expand=True)
        ttk.Button(self.playback_frame, text="Play Original Audio", command=self._play_original_audio, state=tk.DISABLED).pack(side="left", padx=2)
        ttk.Button(self.playback_frame, text="Play Processed Audio", command=self._play_processed_audio, state=tk.DISABLED).pack(side="left", padx=2)
        ttk.Button(self.playback_frame, text="Stop Playback", command=self._stop_audio, state=tk.DISABLED).pack(side="left", padx=2)
        
        
    def _draw_waveform(self, original_data, processed_data=None, title="Waveform Visualization"):
        """
        Draws the waveform(s) on the matplotlib canvas.
        在 matplotlib 畫布上繪製波形。
        """
        self.ax.clear()  # Clear previous plot
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.grid(True)

        if original_data is not None and self.samplerate is not None:
            time_array = np.arange(len(original_data)) / self.samplerate
            self.ax.plot(time_array, original_data, label='Original Audio', color='gray', alpha=0.5)
            
            if processed_data is not None:
                self.ax.plot(time_array, processed_data, label='Processed Audio', color='blue')
                self.ax.legend()
        
        self.fig.canvas.draw_idle()

    def _add_effect(self):
        selected_index = self.available_listbox.curselection()
        if not selected_index:
            return
        
        effect_label = self.available_listbox.get(selected_index[0])
        effect_id = next((k for k, v in self.effect_configs.items() if v['label'] == effect_label), None)
        
        if effect_id:
            # Create a unique ID for this instance
            instance_id = str(uuid.uuid4())
            
            # Create a dictionary to store this instance's parameters
            params_for_instance = {}
            for param_config in self.effect_configs[effect_id]['params']:
                param_id = param_config['id']
                param_var = tk.DoubleVar(value=param_config['default'])
                params_for_instance[param_id] = param_var
            
            # Store the parameters in our main dictionary
            self.param_vars[instance_id] = params_for_instance
            
            # Add the effect to the chain
            self.effect_chain.append({
                'uuid': instance_id,
                'id': effect_id,
                'label': effect_label
            })
            
            # Update the listbox
            self.chain_listbox.insert(tk.END, effect_label)
            
            # Update the sine wave plot to reflect the new chain
            self._update_sine_wave_plot()

    def _remove_effect(self):
        selected_index = self.chain_listbox.curselection()
        if not selected_index:
            return
        
        index_to_remove = selected_index[0]
        item_to_remove = self.effect_chain.pop(index_to_remove)
        self.chain_listbox.delete(index_to_remove)
        
        # Clean up the associated parameter variables
        del self.param_vars[item_to_remove['uuid']]
        
        # Clear the parameter panel if the removed item was selected
        if self.selected_chain_item == item_to_remove['uuid']:
            self._clear_params_ui()
            self.selected_chain_item = None
            
        # Update the sine wave plot to reflect the new chain
        self._update_sine_wave_plot()

    def _on_drag_start(self, event):
        self.drag_item_index = self.chain_listbox.nearest(event.y)
        
    def _on_drag_motion(self, event):
        if self.drag_item_index is not None:
            new_index = self.chain_listbox.nearest(event.y)
            if new_index != self.drag_item_index:
                # Get the item from the listbox and our data structure
                item_text = self.chain_listbox.get(self.drag_item_index)
                item_data = self.effect_chain.pop(self.drag_item_index)

                # Delete the old item and insert the new one
                self.chain_listbox.delete(self.drag_item_index)
                self.chain_listbox.insert(new_index, item_text)
                
                # Insert the data item at the new position
                self.effect_chain.insert(new_index, item_data)

                # Update the drag index
                self.drag_item_index = new_index
                
    def _on_drag_stop(self, event):
        self.drag_item_index = None
        self._update_sine_wave_plot()

    def _on_chain_select(self, event):
        selected_index = self.chain_listbox.curselection()
        if not selected_index:
            self._clear_params_ui()
            self.selected_chain_item = None
            return

        index = selected_index[0]
        effect_data = self.effect_chain[index]
        self.selected_chain_item = effect_data['uuid']
        
        self._update_parameters_ui(effect_data['id'], effect_data['uuid'])

    def _update_parameters_ui(self, effect_id, uuid):
        self._clear_params_ui()
        
        config = self.effect_configs[effect_id]
        self.params_frame.config(text=f"Effect Parameters: {config['label']}")
        
        params_vars = self.param_vars[uuid]
        
        if not config['params']:
            ttk.Label(self.params_frame, text="This effect has no adjustable parameters.").pack(pady=10)
        else:
            for param in config['params']:
                frame = ttk.Frame(self.params_frame)
                frame.pack(fill="x", pady=5)

                ttk.Label(frame, text=f"{param['label']}:").pack(side="left", padx=(0, 10))
                
                param_var = params_vars[param['id']]
                
                value_label = ttk.Label(frame, text=f"{param_var.get():.2f}")
                value_label.pack(side="left", padx=(10, 0))
                
                slider = ttk.Scale(frame, from_=param['min'], to=param['max'],
                                   orient="horizontal", variable=param_var,
                                   command=lambda val, label=value_label: label.config(text=f"{float(val):.2f}"))
                slider.set(param_var.get())
                slider.bind("<ButtonRelease-1>", lambda event: self._update_sine_wave_plot())
                slider.pack(side="left", fill="x", expand=True)
                
        self.master.update_idletasks()

    def _clear_params_ui(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.params_frame.config(text="Effect Parameters")
        
    def _update_sine_wave_plot(self):
        """
        Generates a sine wave, processes it through the current effect chain,
        and updates the visualization plot.
        生成一個正弦波，通過當前的效果器鏈處理，並更新視覺化圖形。
        """
        # Generate a single cycle of a sine wave
        duration = 1.0 / self.sine_wave_freq
        t = np.linspace(0, duration, int(self.sine_wave_samplerate * duration), endpoint=False)
        original_sine_wave = np.sin(2 * np.pi * self.sine_wave_freq * t).astype(np.float32)
        
        processed_sine_wave = original_sine_wave.copy()

        for effect_data in self.effect_chain:
            effect_id = effect_data['id']
            effect_class = self.effect_configs[effect_id]['func']
            effect_instance = effect_class(samplerate=self.sine_wave_samplerate)
            
            params_to_pass = {}
            for param_id, param_var in self.param_vars[effect_data['uuid']].items():
                params_to_pass[param_id] = param_var.get()
            
            if hasattr(effect_instance, 'set_parameters'):
                effect_instance.set_parameters(**params_to_pass)
            
            processed_sine_wave = effect_instance.process(processed_sine_wave)

        # Apply master gain to the processed sine wave
        processed_sine_wave *= self.gain_var.get()

        # Update the plot with the new sine wave data
        self._draw_waveform(original_sine_wave, processed_sine_wave, title="Sine Wave Effect Visualization")

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
                self._update_sine_wave_plot()
                messagebox.showinfo("Audio Loaded", "Audio file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load audio file: {e}")
                self.original_audio_data = None
                self.samplerate = None
                self.file_path_label.config(text="No file selected")
                self.process_button.config(state=tk.DISABLED)
                self.playback_frame.winfo_children()[0].config(state=tk.DISABLED)

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
        Executes audio processing in a separate thread.
        This version processes the entire effect chain.
        """
        if self.original_audio_data is None or self.samplerate is None:
            return

        try:
            # Start with a copy of the original data
            processed_data = self.original_audio_data.copy()

            if not self.effect_chain:
                # If chain is empty, just apply master gain
                self.processed_audio_data = processed_data * self.gain_var.get()
                messagebox.showinfo("Processing Complete", "Audio processing finished! (No effects in chain)")
            else:
                # Iterate through the effect chain and apply each effect in order
                for effect_data in self.effect_chain:
                    effect_id = effect_data['id']
                    effect_class = self.effect_configs[effect_id]['func']
                    effect_instance = effect_class(samplerate=self.samplerate)
                    
                    params_to_pass = {}
                    for param_id, param_var in self.param_vars[effect_data['uuid']].items():
                        params_to_pass[param_id] = param_var.get()
                    
                    if hasattr(effect_instance, 'set_parameters'):
                        effect_instance.set_parameters(**params_to_pass)
                    
                    processed_data = effect_instance.process(processed_data)

                # Finally, apply the master gain
                gain_value = self.gain_var.get()
                self.processed_audio_data = processed_data * gain_value

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
        sd.play(audio_data, self.samplerate)

    def _play_original_audio(self):
        self._play_audio_data(self.original_audio_data)

    def _play_processed_audio(self):
        self._play_audio_data(self.processed_audio_data)

    def _stop_audio(self):
        sd.stop()
        print("Audio playback stopped.")


# --- Main Program Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessorGUI(root)
    root.mainloop()

