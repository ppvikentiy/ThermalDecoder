"""Проверка файла сертификата сборки и локальное состояние «лицензия подтверждена»."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from thermal_decoder.constants import (
    APP_VERSION,
    CERT_FORMAT,
    CERT_VALIDITY_DAYS,
    STATE_FILENAME,
    cert_hmac_secret,
)


def _ensure_writable_dir(path: Path) -> Path | None:
    """
    Возвращает *path*, если в каталоге можно создавать файлы.
    Используется для выбора базовой папки хранения state на разных системах.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".probe"
        probe.write_bytes(b".")
        try:
            probe.unlink()
        except OSError:
            # Даже если удалить не получилось, главное — запись работает.
            pass
        return path
    except OSError:
        return None


def licensing_base_dir() -> Path:
    """
    Каталог для state:
    - рядом с exe (frozen), если можно писать;
    - иначе LocalAppData/TEMP в Windows;
    - в разработке — cwd, затем ~/.thermal_decoder, затем TEMP.
    """
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir)
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            candidates.append(Path(local_app) / "ThermalDecoder")
    else:
        candidates.append(Path.cwd())
        candidates.append(Path.home() / ".thermal_decoder")

    tmp_env = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp_env:
        candidates.append(Path(tmp_env) / "thermal_decoder")
    candidates.append(Path(tempfile.gettempdir()) / "thermal_decoder")

    for c in candidates:
        writable = _ensure_writable_dir(c)
        if writable:
            return writable
    # Fallback: даже если все кандидаты недоступны, возвращаем cwd.
    return Path.cwd()


def state_file_path() -> Path:
    return licensing_base_dir() / STATE_FILENAME


def canonical_signing_message(app_version: str, issued_at: str) -> bytes:
    return f"{app_version}\n{issued_at}".encode("utf-8")


def compute_signature(secret: str, app_version: str, issued_at: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        canonical_signing_message(app_version, issued_at),
        hashlib.sha256,
    ).hexdigest()
    return digest


def verify_signature(secret: str, app_version: str, issued_at: str, signature_hex: str) -> bool:
    try:
        expected = compute_signature(secret, app_version, issued_at)
    except Exception:
        return False
    try:
        return hmac.compare_digest(expected.lower(), signature_hex.strip().lower())
    except Exception:
        return False


def _parse_issued_at(issued_at: str) -> datetime | None:
    try:
        raw = issued_at.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def verify_cert_data(
    data: dict[str, Any],
    *,
    expected_app_version: str,
    secret: str,
    now: datetime | None = None,
) -> tuple[bool, str, datetime | None, datetime | None]:
    """
    Проверяет распарсенный JSON сертификата.
    Возвращает (ok, reason_ru, issued_utc, valid_until_utc).
    """
    now = now or datetime.now(timezone.utc)
    if data.get("format") != CERT_FORMAT:
        return False, "неверный формат файла", None, None
    app_ver = data.get("app_version")
    if not isinstance(app_ver, str) or not app_ver:
        return False, "нет поля app_version", None, None
    if app_ver != expected_app_version:
        return False, "сертификат для другой версии приложения", None, None
    issued_raw = data.get("issued_at")
    if not isinstance(issued_raw, str):
        return False, "нет поля issued_at", None, None
    sig = data.get("signature")
    if not isinstance(sig, str) or not sig:
        return False, "нет подписи", None, None
    issued = _parse_issued_at(issued_raw)
    if issued is None:
        return False, "некорректная дата issued_at", None, None
    if not verify_signature(secret, app_ver, issued_raw.strip(), sig):
        return False, "неверная подпись", issued, None
    valid_until = issued + timedelta(days=CERT_VALIDITY_DAYS)
    if now > valid_until:
        return False, "истёк срок действия сертификата", issued, valid_until
    return True, "", issued, valid_until


def load_cert_file(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"не удалось прочитать файл: {e}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "файл не является корректным JSON"
    if not isinstance(data, dict):
        return None, "ожидался объект JSON"
    return data, ""


def verify_cert_path(
    path: Path,
    *,
    expected_app_version: str | None = None,
    secret: str | None = None,
    now: datetime | None = None,
) -> tuple[bool, str, datetime | None, datetime | None]:
    exp = expected_app_version if expected_app_version is not None else APP_VERSION
    sec = secret if secret is not None else cert_hmac_secret()
    data, err = load_cert_file(path)
    if data is None:
        return False, err or "ошибка чтения", None, None
    return verify_cert_data(data, expected_app_version=exp, secret=sec, now=now)


@dataclass(frozen=True)
class SavedLicenseView:
    """Состояние для UI и отчёта: сохранённая проверка всё ещё валидна."""

    ok: bool
    reason: str
    cert_path: Path | None
    issued_utc: datetime | None
    valid_until_utc: datetime | None


def _load_state_raw() -> dict[str, Any] | None:
    p = state_file_path()
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def save_verified_cert_path(cert_path: Path) -> bool:
    """
    Сохраняет путь к сертификату; возвращает True при успехе, False при ошибке записи.
    """
    cert_path = cert_path.resolve()
    payload = {"cert_path": str(cert_path)}
    p = state_file_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def clear_license_state() -> bool:
    """Удаляет сохранённое состояние; False, если нет прав или файл не удалить."""
    p = state_file_path()
    try:
        if p.is_file():
            p.unlink()
        return True
    except OSError:
        return False


def evaluate_saved_license(now: datetime | None = None) -> SavedLicenseView:
    """Перечитывает сертификат с диска по state и снова проверяет подпись и срок."""
    now = now or datetime.now(timezone.utc)
    raw = _load_state_raw()
    if not raw:
        return SavedLicenseView(
            ok=False,
            reason="проверка не выполнялась",
            cert_path=None,
            issued_utc=None,
            valid_until_utc=None,
        )
    cp = raw.get("cert_path")
    if not isinstance(cp, str) or not cp:
        return SavedLicenseView(
            ok=False,
            reason="в состоянии нет пути к сертификату",
            cert_path=None,
            issued_utc=None,
            valid_until_utc=None,
        )
    path = Path(cp)
    ok, reason, issued, until = verify_cert_path(path, now=now)
    if ok:
        return SavedLicenseView(
            ok=True,
            reason="",
            cert_path=path,
            issued_utc=issued,
            valid_until_utc=until,
        )
    return SavedLicenseView(
        ok=False,
        reason=reason,
        cert_path=path,
        issued_utc=issued,
        valid_until_utc=until,
    )


def license_report_status_and_expiry(now: datetime | None = None) -> tuple[str, str]:
    """
    Две строки для технологического файла:
    статус (не проверялась / действительна / недействительна или истекла),
    срок (YYYY-MM-DD или —).
    """
    now = now or datetime.now(timezone.utc)
    v = evaluate_saved_license(now=now)
    if v.ok and v.valid_until_utc:
        return (
            "действительна",
            v.valid_until_utc.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        )
    if v.reason == "проверка не выполнялась":
        return "не проверялась", "—"
    return "недействительна или истекла", "—"
