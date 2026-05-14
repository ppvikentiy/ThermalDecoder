"""Генерация ThermalDecoder.cert при сборке (PyInstaller)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from thermal_decoder.constants import APP_VERSION, CERT_FORMAT, cert_hmac_secret
from thermal_decoder.cert_license import compute_signature


def build_certificate_payload() -> dict[str, str]:
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    issued_at = issued.isoformat()
    secret = cert_hmac_secret()
    sig = compute_signature(secret, APP_VERSION, issued_at)
    return {
        "format": CERT_FORMAT,
        "app_version": APP_VERSION,
        "issued_at": issued_at,
        "signature": sig,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write signed ThermalDecoder certificate JSON.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output path (e.g. dist/ThermalDecoder/ThermalDecoder.cert)",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_certificate_payload()
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
