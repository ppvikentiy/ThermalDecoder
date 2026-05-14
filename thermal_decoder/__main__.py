"""CLI: python -m thermal_decoder …"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder.thermal_decoder import ThermalDecoder


def _cmd_scan(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    out = Path(args.output)
    rect: tuple[int, int, int, int] | None = None
    if args.rect:
        parts = args.rect.replace(",", " ").split()
        if len(parts) != 4:
            print("Ожидается --rect X Y W H", file=sys.stderr)
            return 2
        rect = tuple(int(x) for x in parts)
    if not args.auto_scale and rect is None:
        print("Задайте --rect X Y W H или флаг --auto-scale", file=sys.stderr)
        return 2

    scale_rect = None if args.auto_scale else rect
    try:
        dec = ThermalDecoder()
        result = dec.scan_ocv(
            inp,
            output_dir=out,
            scale_rect=scale_rect,
            min_temp=args.min_temp,
            max_temp=args.max_temp,
            apply_blur=args.blur,
            auto_detect_scale=args.auto_scale,
            gradient_strict=not args.soft_gradient,
            overlay_mode=args.overlay,
            colormap_name=args.colormap.upper(),
            grid_step=args.grid_step,
            include_temp_matrix=not args.no_matrix,
        )
    except System_OCV_Vis_Temp_Error as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print("OK:", result.get("result_overlay"))
    print("   ", result.get("data_csv"))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="ThermalDecoder — сканирование BMP без GUI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="Полный пайплайн scan_ocv")
    s.add_argument("input", type=str, help="Путь к .bmp")
    s.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Папка для result_overlay.bmp и data.csv",
    )
    s.add_argument("--min-temp", type=float, default=0.0)
    s.add_argument("--max-temp", type=float, default=100.0)
    s.add_argument(
        "--rect",
        type=str,
        default="",
        help="Область шкалы: 'X Y W H' (иначе при --auto-scale или ошибка)",
    )
    s.add_argument(
        "--auto-scale",
        action="store_true",
        help="Искать шкалу автоматически (игнор --rect)",
    )
    s.add_argument("--blur", action="store_true")
    s.add_argument(
        "--soft-gradient",
        action="store_true",
        help="Мягкая проверка градиента шкалы",
    )
    s.add_argument(
        "--overlay",
        choices=("grid", "colormap", "both"),
        default="both",
    )
    s.add_argument("--colormap", type=str, default="JET")
    s.add_argument("--grid-step", type=int, default=64)
    s.add_argument(
        "--no-matrix",
        action="store_true",
        help="Не держать temp_matrix в ответе (экономия RAM)",
    )
    s.set_defaults(func=_cmd_scan)

    args = p.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
