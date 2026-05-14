"""ThermalDecoder: BMP thermography decoding and GUI."""

from thermal_decoder.constants import APP_VERSION as __version__

from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder.scale_detector import ScaleDetector
from thermal_decoder.thermal_decoder import ThermalDecoder

__all__ = [
    "__version__",
    "System_OCV_Vis_Temp_Error",
    "ScaleDetector",
    "ThermalDecoder",
]
