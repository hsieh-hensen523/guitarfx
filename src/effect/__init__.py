# src/effect/__init__.py

# Import and re-export the classes from their respective modules.
# 從各自的模組中匯入並重新匯出類別。
from .spatial.Delay import Delay
from .gain_stage.Distortion import Distortion
from .gain_stage.Overdrive import Overdrive # Assuming you have an Overdrive.py as well.
# from .filters.PeakingEQ import PeakingEQ # Assuming you have a PeakingEQ.py as well.
from .filters.Filters import LowpassFilter, HighpassFilter, BandpassFilter
from .dynamics.noise_reduction import NoiseReduction # Assuming you have a noise_reduction.py as well.
from .spatial.Convolution import ConvolutionIR # Assuming you have a Convolution.py as well.

# You can also use a list to define what gets imported with `from effect import *`
# 也可以使用一個列表來定義當執行 `from effect import *` 時會匯入什麼。
__all__ = [
    "Delay",
    "Distortion",
    "Overdrive",
    # "PeakingEQ",
    "LowpassFilter",
    "HighpassFilter",
    "BandpassFilter",
    "NoiseReduction",
    "Convolution"
]