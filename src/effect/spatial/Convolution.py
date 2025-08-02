# effects/cabinet_sim/Convolution.py

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import resample
import soundfile as sf
import os
import sys
from ..base_effect import AudioEffect

class ConvolutionIR(AudioEffect):
    """
    一個用於使用頻域卷積(FFT)來套用脈衝響應(IR)的類別。
    這是音箱箱體模擬的核心。
    """
    def __init__(self, ir_path, samplerate):
        """
        初始化類別，載入並處理 IR 檔案。

        Args:
            ir_path (str): 脈衝響應(IR)檔案的路徑。
            samplerate (int): 你的專案所使用的音訊採樣率。
        """
        print(f"載入 IR 檔案：{ir_path}")
        try:
            # 載入 IR 檔案並檢查採樣率
            ir_data, ir_sr = sf.read(ir_path)
            if ir_sr != samplerate:
                print(f"警告:IR 檔案採樣率 ({ir_sr} Hz) 與專案採樣率 ({samplerate} Hz) 不匹配。")
                print("這可能會導致聲音異常。請確保 IR 檔案的採樣率正確。")
                # 在實際應用中，你可能需要在這裡加入重採樣邏輯。
                ir_data = resample(ir_data, int(len(ir_data) * samplerate / ir_sr))

            # 確保 IR 數據是單聲道，如果不是則取平均
            if len(ir_data.shape) > 1:
                ir_data = np.mean(ir_data, axis=1)

            self.ir_data = ir_data
            self.samplerate = samplerate
            self.ir_length = len(ir_data)

            # 進行 IR 的 FFT 轉換，這是頻域卷積的第一步
            self.ir_fft = fft(self.ir_data, n=self.ir_length)
            
            print("IR 載入並處理完成。")
        except FileNotFoundError:
            print(f"錯誤：找不到 IR 檔案：{ir_path}")
            self.ir_data = None
            self.ir_fft = None

    def process(self, audio_data):
        """
        對輸入的音訊數據進行卷積處理。

        Args:
            audio_data (np.ndarray): 輸入的音訊數據（單聲道）。

        Returns:
            np.ndarray: 處理後的音訊數據。
        """
        if self.ir_data is None:
            print("錯誤:IR 未成功載入，無法進行處理。")
            return audio_data

        audio_length = len(audio_data)
        print(f"處理音訊長度: {audio_length}, IR 長度: {self.ir_length}")

        # 為了進行高效的頻域卷積，我們需要將兩個訊號填充到相同的長度，
        # 且這個長度通常是 2 的冪次方，這會讓 FFT 演算法最快。
        # 我們將長度設為 audio_length + ir_length - 1，然後向上取最接近的 2 的冪次方。
        convolution_length = audio_length + self.ir_length - 1
        n_fft = 2 ** int(np.ceil(np.log2(convolution_length)))
        print(f"卷積處理長度: {n_fft} (音訊長度 + IR 長度 - 1)")
        
        print(f"音訊長度: {audio_length}, IR 長度: {self.ir_length}, 卷積處理長度: {n_fft}")

        # 進行音訊和 IR 的 FFT 轉換
        audio_fft = fft(audio_data, n=n_fft)
        ir_fft = fft(self.ir_data, n=n_fft)
        print("audio_fft長度:", len(audio_fft), "ir_fft長度:", len(ir_fft))

        # 頻域卷積的核心：兩個訊號的頻譜進行逐點相乘
        convolved_fft = audio_fft * ir_fft

        # 使用 IFFT 將結果轉換回時域
        convolved_audio = ifft(convolved_fft)

        # 由於 FFT/IFFT 轉換的結果是複數，我們只需要實部
        convolved_audio = np.real(convolved_audio)

        # 修剪結果，使其長度與輸入音訊加上 IR 的長度一致
        convolved_audio = convolved_audio[:convolution_length]

        # 正規化音訊以防止爆音
        peak = np.max(np.abs(convolved_audio))
        if peak > 1.0:
            convolved_audio /= peak

        return convolved_audio

# ----------------- 測試程式碼 -----------------
if __name__ == "__main__":
    # 這裡放一個範例 IR 和音訊檔案路徑
    # 假設你已經將 IR 檔案放在 irs/free_IR_B/ 資料夾中
    # 獲取當前腳本的絕對路徑
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 透過兩個 '..' 回到專案根目錄
    project_root = os.path.join(current_script_dir, '..', '..')
    
    # 設定 IR 檔案的相對路徑
    # 假設你使用 irs/DYN-57/DYN-57_example.wav 作為測試 IR
    test_ir_path = os.path.join(project_root, 'irs', 'DYN-57', 'OD-M212-P50-DYN-57-P05-10.wav')
    
    # 設定測試音訊檔案的路徑
    # 假設你的測試音訊在 music/test_audio.wav
    test_audio_path = os.path.join(project_root, 'music', 'funky_guitar.wav')
    
    # 設定輸出音訊檔案的路徑
    output_audio_path = os.path.join(project_root, 'processed_audio', 'processed_convolution_audio.wav')

    # 如果 IR 檔案不存在，則建立一個簡單的測試 IR 
    if not os.path.exists(test_ir_path):
        print(f"IR 檔案 '{test_ir_path}' 不存在，建立一個簡單的測試 IR。")
        samplerate = 44100
        duration = 0.5  # 0.5 秒的簡單 IR
        t = np.linspace(0., duration, int(samplerate * duration))
        test_ir_data = np.sin(2 * np.pi * 100 * t) * np.exp(-t / 0.1)
        sf.write(test_ir_path, test_ir_data, samplerate)

    # 如果音訊檔案不存在，則建立一個簡單的測試音訊
    if not os.path.exists(test_audio_path):
        print(f"測試音訊檔案 '{test_audio_path}' 不存在，建立一個簡單的測試音訊。")
        samplerate = 44100
        duration = 5.0  # 5 秒的簡單音訊
        t = np.linspace(0., duration, int(samplerate * duration))
        test_audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz 的正弦波
        sf.write(test_audio_path, test_audio_data, samplerate)
    
    # 讀取測試音訊
    audio_data, samplerate = sf.read(test_audio_path)
    
    # 初始化 ConvolutionIR 類別
    convolver = ConvolutionIR(ir_path=test_ir_path, samplerate=samplerate)

    # 進行卷積處理
    print("\n開始進行卷積處理..")
    processed_audio = convolver.process(audio_data)

    # 將處理後的音訊存檔
    sf.write(output_audio_path, processed_audio, samplerate)
    print(f"\n處理完成!音訊已儲存到 '{output_audio_path}'")
