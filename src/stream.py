from effect import ConvolutionIR
import pyaudio
import numpy as np
import time
import webrtcvad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue
from utils.AudioProcessor import AudioProcessor

# -- 設定音訊參數 --
RATE = 48000  
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# -- VAD 設定 --
vad = webrtcvad.Vad(0)
pop_noise_silence_counter = 0
POP_SILENCE_FRAMES = 10
speech_frame_count = 0
SPEECH_FRAME_THRESHOLD = 10
speech_started = False

# # -- 共享 queue --
# audio_queue = queue.Queue()

# -- 效果設定 --
MAX_VOLUME_RMS = 0.05
overdrive_effect = None

# -- 畫圖計數器 --
frame_counter = [0]

# ---------------------------- 視覺化 ----------------------------

def animate(i, line_time, line_freq, processor, rate, chunk):
    # 這裡的 processor 是一個 AudioProcessor 的實例
    if not processor.audio_queue.empty():
        try:
            # 從 processor 的 queue 中獲取數據
            data = processor.audio_queue.get_nowait()
            if len(data) < chunk:
                return line_time, line_freq
            
            line_time.set_ydata(data)
            
            fft_data = np.abs(np.fft.rfft(data))
            fft_data = fft_data / np.max(fft_data + 1e-10)
            line_freq.set_ydata(fft_data)
            
        except queue.Empty:
            # 如果隊列為空，則不做任何事
            return line_time, line_freq
            
    return line_time, line_freq

def init_plot(rate, chunk):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    t = np.arange(chunk) / rate
    f = np.fft.rfftfreq(chunk, d=1./rate)
    line_time, = ax1.plot(t, np.zeros(chunk))
    ax1.set_ylim(-1, 1)
    ax1.set_title("Time Domain")
    line_freq, = ax2.plot(f, np.zeros(len(f)))
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, rate / 2)
    ax2.set_title("Frequency Domain")
    return fig, line_time, line_freq

# ---------------------------- 主程式 ----------------------------

def run_audio_processor():
    global overdrive_effect
    p = pyaudio.PyAudio()
    input_device_index = 21
    output_device_index = 19

    print("--- 開啟音訊串流 ---")
    print(f"輸入裝置索引: {input_device_index}")
    print(f"輸出裝置索引: {output_device_index}")

    try:
        processor = AudioProcessor(
            samplerate=RATE, 
            lowcut=100, 
            highcut=12000, 
            order=5, 
            chunk_size=CHUNK_SIZE,
            gain=3.0,
        )
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            stream_callback=processor.audio_callback
        )
    except Exception as e:
        print(f"音訊裝置錯誤：{e}")
        p.terminate()
        return

    print("--- 音訊處理已啟動，Ctrl+C 結束 ---")
    stream.start_stream()

    fig, line_time, line_freq = init_plot(RATE, CHUNK_SIZE)
    ani = animation.FuncAnimation(
        fig, animate, fargs=(line_time, line_freq, processor, RATE, CHUNK_SIZE),
        interval=CHUNK_DURATION_MS, blit=False
    )
    plt.show()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n--- 中斷，關閉中 ---")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("程式結束。")

if __name__ == "__main__":
    run_audio_processor()
