"""Shared constants and default thresholds."""

import os

APP_VERSION = "0.0.3-beta"

# Сертификат сборки (HMAC): при сборке и в приложении должен совпадать секрет.
# Переопределение: переменная окружения THERMAL_DECODER_CERT_SECRET.
CERT_FORMAT = "thermal_decoder_cert_v1"
CERT_VALIDITY_DAYS = 90
CERT_HMAC_SECRET = "thermal-decoder-cert-hmac-change-in-production"

STATE_FILENAME = "thermal_decoder_license_state.json"
CERT_DIST_FILENAME = "ThermalDecoder.cert"


def cert_hmac_secret() -> str:
    """Секрет подписи сертификата: env, иначе CERT_HMAC_SECRET."""
    return os.environ.get("THERMAL_DECODER_CERT_SECRET") or CERT_HMAC_SECRET

gaussian_ksize = 5
gaussian_sigma = 0

default_grid_step_px = 64

bgr_near_scale_thresh = 62.0
bgr_ambiguous_low = 18.0

# Пиксели под маской «текст/UI»: отсев только если далеко от палитры шкалы
text_mask_min_bgr_dist = 20.0

# В оверлее не рисовать больше столько маркеров INVALID (прореживание)
max_invalid_overlay_markers = 5000

hsv_white_v_min = 220
hsv_black_v_max = 45
hsv_chroma_max_for_extreme = 90

lab_l_white_min = 88.0
lab_l_black_max = 35.0
lab_ab_chroma_max_neutral = 25.0

invalid_cross_color_bgr = (0, 255, 0)
invalid_cross_thickness = 2
invalid_err_text = "ERR"

gradient_strict_multiplier = 6.0
gradient_soft_multiplier = 12.0
gradient_delta_e_floor_strict = 14.0
gradient_delta_e_floor_soft = 22.0

csv_chunk_rows = 4096

colormap_names = (
    "JET",
    "HOT",
    "INFERNO",
    "VIRIDIS",
    "TURBO",
    "PLASMA",
)
