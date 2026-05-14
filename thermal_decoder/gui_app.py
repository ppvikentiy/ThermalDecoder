"""Графический интерфейс ThermalDecoder на Tkinter: мастер из четырёх шагов без Qt."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import datetime
import os
import platform
import queue
import shutil
import socket
import sys
import tempfile
import threading
import zipfile
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from thermal_decoder.cert_license import (
    clear_license_state,
    evaluate_saved_license,
    license_report_status_and_expiry,
    save_verified_cert_path,
    verify_cert_path,
)
from thermal_decoder.constants import APP_VERSION, default_grid_step_px
from thermal_decoder.cv_io import imread_bgr, imwrite
from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder.io_export import _bw_fg_bg
from thermal_decoder.thermal_decoder import ThermalDecoder


def _instruction_txt_path() -> Path | None:
    """Путь к ИНСТРУКЦИЯ.txt: рядом с exe (сборка), каталог проекта, текущая папка."""
    name = "ИНСТРУКЦИЯ.txt"
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / name)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / name)
    candidates.append(Path(__file__).resolve().parent.parent / name)
    candidates.append(Path.cwd() / name)
    for p in candidates:
        if p.is_file():
            return p
    return None


def _gradient_strict_from_ui(label: str) -> bool:
    """True — строгая проверка градиента шкалы; False — мягкая (по выбору в комбобоксе)."""
    return label.startswith("Строг")


_STEP_TITLES = (
    "Шаг 1 из 4 — файл и папка вывода",
    "Шаг 2 из 4 — область шкалы и температура",
    "Шаг 3 из 4 — параметры анализа",
    "Шаг 4 из 4 — сохранение результатов",
)

_APP_NAME = "ThermalDecoder"

_UI_BG = "#f0f2f5"
_UI_LF_BG = "#ffffff"
_UI_STATUS_BG = "#e8eaef"
_UI_ACCENT = "#0b5cad"
_UI_ACCENT_HOVER = "#094a8f"


def _ui_font_tuple(size: int, weight: str = "normal") -> tuple[str | int, ...]:
    """Возвращает кортеж шрифта интерфейса: Segoe UI в Windows, иначе системный Tk."""
    if sys.platform == "win32":
        fam = "Segoe UI"
    else:
        fam = str(tkfont.nametofont("TkDefaultFont").actual("family"))
    if weight == "bold":
        return (fam, size, "bold")
    return (fam, size)


def _mono_font(size: int) -> tuple[str | int, ...]:
    """Моноширинный шрифт для полей с фиксированной шириной символа (логи, диалоги)."""
    if sys.platform == "win32":
        return ("Consolas", size)
    return ("TkFixedFont", size)


def _bind_ctrl_mousewheel(widget: tk.Canvas, callback) -> None:
    """Вешает Ctrl+колесо на canvas: в Windows/macOS — MouseWheel, в Linux — кнопки 4/5."""
    if sys.platform in ("win32", "darwin"):
        widget.bind("<Control-MouseWheel>", callback)
    else:

        def up(e: tk.Event) -> None:
            callback(SimpleNamespace(delta=120, x=e.x, y=e.y))

        def down(e: tk.Event) -> None:
            callback(SimpleNamespace(delta=-120, x=e.x, y=e.y))

        widget.bind("<Control-Button-4>", up)
        widget.bind("<Control-Button-5>", down)


def _harden_environment() -> None:
    """
    Подчищает чувствительные переменные окружения для старых систем и нестандартных окружений:
    1) убирает битые пути Tcl/Tk, чтобы Tkinter не падал при старте;
    2) гарантирует рабочий TEMP/TMP и PATH;
    3) выставляет безопасную кодировку ввода/вывода.
    """
    try:
        for key in ("TCL_LIBRARY", "TK_LIBRARY", "TKPATH", "TCLLIBPATH"):
            val = os.environ.get(key)
            if val and not Path(val).exists():
                os.environ.pop(key, None)
        tmp_dir = None
        try:
            tmp_dir = Path(tempfile.gettempdir())
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            tmp_dir = Path.cwd()
        for key in ("TEMP", "TMP"):
            val = os.environ.get(key)
            if not val or not Path(val).exists():
                os.environ[key] = str(tmp_dir)
        os.environ.setdefault("PATH", "")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        # Любая ошибка при санации окружения не должна блокировать запуск.
        pass


class ThermalDecoderApp:
    """Окно мастера: выбор файла, настройка ROI шкалы и запуск анализа в фоне."""

    @staticmethod
    def _safe_int(var: tk.IntVar, default: int = 0) -> int:
        """Безопасное чтение IntVar из Spinbox (пустая строка даёт TclError — возвращается default)."""
        try:
            return int(var.get())
        except (tk.TclError, ValueError, TypeError):
            return default

    def _get_scale_rect_values(self) -> tuple[int, int, int, int]:
        """Текущий прямоугольник ROI шкалы (x, y, w, h) в координатах изображения, с ограничением по кадру."""
        rx = self._safe_int(self.var_rx, 0)
        ry = self._safe_int(self.var_ry, 0)
        rw = max(1, self._safe_int(self.var_rw, 40))
        rh = max(1, self._safe_int(self.var_rh, 100))
        if self._bgr is not None:
            ih, iw = self._bgr.shape[:2]
            rx = max(0, min(rx, max(0, iw - 1)))
            ry = max(0, min(ry, max(0, ih - 1)))
            rw = max(1, min(rw, max(1, iw - rx)))
            rh = max(1, min(rh, max(1, ih - ry)))
        return rx, ry, rw, rh

    def _setup_styles(self) -> None:
        """Настраивает тему ttk (clam), цвета и стили кнопок, подсказок и статус-бара."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        ui = _ui_font_tuple(10)
        ui_sm = _ui_font_tuple(9)
        hint = _ui_font_tuple(9)
        title = _ui_font_tuple(13, "bold")

        self.root.configure(bg=_UI_BG)

        style.configure("TFrame", background=_UI_BG)
        style.configure("TLabel", font=ui, foreground="#1f2937")
        style.configure(
            "Title.TLabel",
            background=_UI_BG,
            foreground="#0f172a",
            font=title,
        )
        style.configure(
            "Hint.TLabel",
            background=_UI_BG,
            foreground="#4b5563",
            font=hint,
        )
        style.configure(
            "MicroHint.TLabel",
            background=_UI_LF_BG,
            foreground="#6b7280",
            font=_ui_font_tuple(8),
        )

        style.configure(
            "TLabelframe",
            background=_UI_LF_BG,
            foreground="#111827",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=_UI_LF_BG,
            foreground="#111827",
            font=_ui_font_tuple(10, "bold"),
        )

        style.configure("TButton", font=ui, padding=(12, 6))
        style.configure(
            "Accent.TButton",
            font=ui,
            padding=(12, 6),
            background=_UI_ACCENT,
            foreground="#ffffff",
        )
        style.map(
            "Accent.TButton",
            background=[("active", _UI_ACCENT_HOVER), ("disabled", "#9ca3af")],
            foreground=[("disabled", "#f3f4f6")],
        )

        style.configure("TEntry", font=ui, fieldbackground=_UI_LF_BG)
        style.configure(
            "TSpinbox", font=ui, fieldbackground=_UI_LF_BG, padding=(4, 2)
        )
        style.configure(
            "TCombobox", font=ui, fieldbackground=_UI_LF_BG, padding=(4, 2)
        )
        style.configure("TCheckbutton", font=ui)
        style.configure("Horizontal.TSeparator", background="#d1d5db")

        style.configure("StatusBar.TFrame", background=_UI_STATUS_BG)
        style.configure(
            "Status.TLabel",
            background=_UI_STATUS_BG,
            foreground="#374151",
            font=ui_sm,
            padding=(12, 8),
        )

    def __init__(self) -> None:
        """Создаёт окно, переменные состояния, четыре шага мастера и цикл опроса очереди фоновых результатов."""
        _harden_environment()
        self.root = tk.Tk()
        self.root.title(
            f"ThermalDecoder — анализ термограмм (BMP) — {APP_VERSION}"
        )
        # Подбираем размер окна под экран (важно для устройств с высоким DPI,
        # где фиксированная высота 720px приводила к скрытой панели навигации).
        screen_w = max(1, self.root.winfo_screenwidth())
        screen_h = max(1, self.root.winfo_screenheight())
        base_w, base_h = 960, 720
        margin_w, margin_h = 80, 120  # запас под рамки/панель задач
        min_w_floor, min_h_floor = 760, 560
        min_w = min(base_w, max(min_w_floor, screen_w - margin_w))
        min_h = min(base_h, max(min_h_floor, screen_h - margin_h))
        # Не выходим за пределы экрана (на случай экстремально маленьких экранов).
        min_w = min(min_w, max(300, screen_w - 20))
        min_h = min(min_h, max(300, screen_h - 20))
        self.root.minsize(min_w, min_h)
        self.root.geometry(f"{min_w}x{min_h}")
        self._setup_styles()

        menubar = tk.Menu(self.root, tearoff=0)
        menubar.add_command(label="О системе", command=self._show_about_system)
        menubar.add_command(label="О программе", command=self._show_about_program)
        menubar.add_command(label="Лицензия", command=self._show_license)
        menubar.add_command(label="Инструкция", command=self._show_instruction)
        menubar.add_command(label="Сертификат", command=self._show_certificate_dialog)
        self.root.config(menu=menubar)

        self.image_path: str | None = None
        self._bgr: np.ndarray | None = None
        self._temp_matrix: np.ndarray | None = None
        self._disp_scale = 1.0
        self._disp_off_x = 0
        self._disp_off_y = 0
        self._photo: ImageTk.PhotoImage | None = None
        self._drag_anchor: tuple[int, int] | None = None

        self._queue: queue.Queue = queue.Queue()
        self._step = 1

        self.var_blur = tk.BooleanVar(value=False)
        self.var_outdir = tk.StringVar(value="")
        self.var_image_label = tk.StringVar(value="Файл не выбран")

        self.var_rx = tk.IntVar(value=0)
        self.var_ry = tk.IntVar(value=0)
        self.var_rw = tk.IntVar(value=40)
        self.var_rh = tk.IntVar(value=100)

        self.var_tmin = tk.StringVar(value="0")
        self.var_tmax = tk.StringVar(value="100")

        self.var_create_archive = tk.BooleanVar(value=False)
        self.var_create_tech_file = tk.BooleanVar(value=False)

        self._resize_after_id: str | None = None
        self._last_configure_wh: tuple[int, int] = (0, 0)
        self._busy = False
        self._step2_user_zoom = 1.0
        self._step2_scroll_wh: tuple[int, int] = (800, 600)
        self._step2_pan_active = False

        self.var_step_title = tk.StringVar(value=_STEP_TITLES[0])

        # --- Основной лэйаут (grid, чтобы фиксировать навигацию и статус снизу) ---
        layout = ttk.Frame(self.root)
        layout.pack(fill=tk.BOTH, expand=True)
        layout.rowconfigure(1, weight=1)
        layout.columnconfigure(0, weight=1)

        # --- Заголовок шага ---
        hdr = ttk.Frame(layout, padding=(16, 14, 16, 6))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            hdr,
            textvariable=self.var_step_title,
            style="Title.TLabel",
        ).pack(anchor=tk.W)

        # --- Контейнер шагов ---
        self.container = ttk.Frame(layout, padding=(12, 4, 12, 8))
        self.container.grid(row=1, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.step1 = ttk.Frame(self.container, padding=16)
        self.step2 = ttk.Frame(self.container, padding=12)
        self.step3 = ttk.Frame(self.container, padding=16)
        self.step4 = ttk.Frame(self.container, padding=16)

        self._build_step1(self.step1)
        self._build_step2(self.step2)
        self._build_step3(self.step3)
        self._build_step4(self.step4)

        for f in (self.step1, self.step2, self.step3, self.step4):
            f.grid(row=0, column=0, sticky="nsew")

        ttk.Separator(layout, orient=tk.HORIZONTAL).grid(
            row=2, column=0, sticky="ew"
        )

        # --- Навигация ---
        nav = ttk.Frame(layout, padding=(16, 8, 16, 12))
        nav.grid(row=3, column=0, sticky="ew")
        self.btn_back = ttk.Button(nav, text="Назад", command=self.on_nav_back)
        self.btn_back.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_next = ttk.Button(
            nav,
            text="Далее",
            command=self.on_nav_next,
            style="Accent.TButton",
        )
        self.btn_next.pack(side=tk.LEFT, padx=4)
        self.btn_analyze = ttk.Button(
            nav, text="Анализ", command=self.on_scan, style="Accent.TButton"
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=4)

        self.status = tk.StringVar(
            value="Шаг 1: выберите BMP и папку для результатов, затем «Далее»."
        )
        status_bar = ttk.Frame(layout, style="StatusBar.TFrame")
        status_bar.grid(row=4, column=0, sticky="ew")
        ttk.Label(
            status_bar,
            textvariable=self.status,
            style="Status.TLabel",
            anchor=tk.W,
        ).pack(fill=tk.X)

        # Горячие клавиши навигации (на случай если панель скрыта темой/окном).
        for seq, handler in (
            ("<Control-Right>", self.on_nav_next),
            ("<Alt-Right>", self.on_nav_next),
            ("<Control-Left>", self.on_nav_back),
            ("<Alt-Left>", self.on_nav_back),
            ("<Control-Return>", self.on_scan),
            ("<Control-KP_Enter>", self.on_scan),
        ):
            self.root.bind(seq, lambda e, h=handler: h())

        self._show_step(1)
        self.root.after(120, self.poll_queue)

    def _show_certificate_dialog(self) -> None:
        """Диалог проверки файла сертификата; при невалидности только текст в окне, без messagebox."""
        win = tk.Toplevel(self.root)
        win.title("Сертификат")
        win.geometry("480x260")
        win.minsize(400, 200)
        win.transient(self.root)
        win.grab_set()
        win.configure(bg=_UI_BG)
        outer = ttk.Frame(win, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        def close() -> None:
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        btn_row = ttk.Frame(win, padding=(12, 0, 12, 12))
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Закрыть", command=close).pack(side=tk.RIGHT)
        win.protocol("WM_DELETE_WINDOW", close)

        last_pick: dict[str, str | bool] = {}

        def run_verify_pick() -> None:
            path = filedialog.askopenfilename(
                parent=win,
                title="Выберите файл сертификата",
                filetypes=[
                    ("Сертификат ThermalDecoder", "*.cert"),
                    ("Все файлы", "*.*"),
                ],
            )
            if not path:
                return
            ok, reason, _issued, _until = verify_cert_path(Path(path))
            last_pick.clear()
            if ok:
                saved = save_verified_cert_path(Path(path))
                if not saved:
                    last_pick["path"] = path
                    last_pick["ok"] = False
                    last_pick[
                        "reason"
                    ] = "не удалось сохранить состояние (нет прав на запись)"
                    messagebox.showwarning(
                        "Сертификат",
                        "Подпись подтверждена, но сохранить состояние проверки не удалось. "
                        "Запустите приложение из папки, где разрешена запись, "
                        "или попробуйте снова.",
                        parent=win,
                    )
                else:
                    last_pick["path"] = path
                    last_pick["ok"] = True
            else:
                last_pick["path"] = path
                last_pick["ok"] = False
                last_pick["reason"] = reason
            refresh()

        def refresh() -> None:
            for w in body.winfo_children():
                w.destroy()
            view = evaluate_saved_license()
            if view.ok and view.valid_until_utc:
                vu = view.valid_until_utc.strftime("%Y-%m-%d")
                ttk.Label(
                    body,
                    text="Подпись проверена.",
                    style="Title.TLabel",
                ).pack(anchor=tk.W, pady=(0, 8))
                ttk.Label(
                    body,
                    text=f"Срок действия сертификата до: {vu} (UTC).",
                    style="Hint.TLabel",
                ).pack(anchor=tk.W, pady=(0, 12))
                row = ttk.Frame(body)
                row.pack(fill=tk.X, pady=4)

                def _clear_and_refresh() -> None:
                    if not clear_license_state():
                        messagebox.showwarning(
                            "Сертификат",
                            "Не удалось удалить сохранённое состояние сертификата "
                            "(нет прав на запись в каталог программы или профиля).",
                            parent=win,
                        )
                    refresh()

                ttk.Button(
                    row,
                    text="Отменить проверку лицензии",
                    command=_clear_and_refresh,
                ).pack(side=tk.LEFT, padx=(0, 8))
                ttk.Button(
                    row,
                    text="Проверить заново",
                    command=run_verify_pick,
                ).pack(side=tk.LEFT)
                if last_pick and last_pick.get("ok") is False:
                    ttk.Label(
                        body,
                        text=(
                            "Выбранный файл не принят (предыдущая подпись сохранена): "
                            f"{last_pick.get('reason', '')}."
                        ),
                        style="Hint.TLabel",
                        wraplength=440,
                    ).pack(anchor=tk.W, pady=(12, 0))
                return

            ttk.Label(
                body,
                text=(
                    "Укажите файл сертификата (ThermalDecoder.cert), "
                    "полученный в комплекте с этой версией приложения."
                ),
                style="Hint.TLabel",
                wraplength=440,
            ).pack(anchor=tk.W, pady=(0, 8))
            if view.reason != "проверка не выполнялась":
                ttk.Label(
                    body,
                    text=f"Текущее состояние: не действителен — {view.reason}.",
                    style="Hint.TLabel",
                    wraplength=440,
                ).pack(anchor=tk.W, pady=(0, 8))
            ttk.Button(body, text="Проверить…", command=run_verify_pick).pack(
                anchor=tk.W, pady=(0, 8)
            )
            if last_pick:
                ok = bool(last_pick.get("ok"))
                reason = str(last_pick.get("reason", ""))
                if ok:
                    ttk.Label(
                        body,
                        text="Результат: валиден.",
                        style="Hint.TLabel",
                    ).pack(anchor=tk.W)
                else:
                    ttk.Label(
                        body,
                        text=f"Результат: не валиден — {reason}.",
                        style="Hint.TLabel",
                        wraplength=440,
                    ).pack(anchor=tk.W)

        refresh()

    def _show_text_dialog(
        self, title: str, body: str, *, geometry: str = "520x320"
    ) -> None:
        """Модальное окно с прокручиваемым текстом (меню «О системе», «О программе», «Лицензия»)."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry(geometry)
        win.minsize(400, 240)
        win.transient(self.root)
        win.grab_set()
        win.configure(bg=_UI_BG)
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        txt = scrolledtext.ScrolledText(
            frm,
            wrap=tk.WORD,
            font=_mono_font(10),
            relief=tk.FLAT,
            borderwidth=0,
            bg=_UI_LF_BG,
            fg="#1f2937",
            insertbackground="#1f2937",
        )
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, body)
        txt.configure(state=tk.DISABLED)
        btn_row = ttk.Frame(win, padding=(12, 0, 12, 12))
        btn_row.pack(fill=tk.X)

        def _close() -> None:
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        ttk.Button(btn_row, text="Закрыть", command=_close).pack(side=tk.RIGHT)
        win.protocol("WM_DELETE_WINDOW", _close)

    def _show_instruction(self) -> None:
        """Меню «Инструкция»: текст из файла ИНСТРУКЦИЯ.txt."""
        p = _instruction_txt_path()
        if p is None:
            messagebox.showwarning(
                "Инструкция",
                "Файл ИНСТРУКЦИЯ.txt не найден. Положите его в папку с программой "
                "или запускайте приложение из каталога, где лежит этот файл.",
            )
            return
        try:
            body = p.read_text(encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Инструкция", f"Не удалось прочитать файл:\n{e}")
            return
        self._show_text_dialog("Инструкция", body, geometry="720x560")

    def _system_info_lines(self) -> list[str]:
        """Строки сведений об ОС и железе (как в меню «О системе»)."""
        lines: list[str] = []
        lines.append(f"Имя узла: {socket.gethostname()}")
        lines.append(f"ОС: {platform.system()} {platform.release()}")
        lines.append(f"Версия ОС (platform.version): {platform.version()}")
        lines.append(f"Машина: {platform.machine()}")
        arch = platform.architecture()
        lines.append(f"Разрядность процесса: {arch[0]}, связывание: {arch[1]}")
        proc = platform.processor()
        if proc:
            lines.append(f"Процессор: {proc}")
        ncpu = os.cpu_count()
        if ncpu is not None:
            lines.append(f"Логических процессоров: {ncpu}")
        if sys.platform == "win32":
            lines.append(f"win32_ver: {platform.win32_ver()}")
            try:
                edition = platform.win32_edition()
            except (AttributeError, OSError, ValueError):
                edition = None
            if edition:
                lines.append(f"Издание Windows: {edition}")
        return lines

    @staticmethod
    def _find_spec_ok(name: str) -> bool:
        """True, если модуль доступен для импорта (importlib.util.find_spec)."""
        return importlib.util.find_spec(name) is not None

    def _program_info_lines(self) -> list[str]:
        """Строки сведений о приложении и зависимостях (меню «О программе»)."""
        lines: list[str] = []
        lines.append(f"Приложение: {_APP_NAME}")
        lines.append(f"Версия приложения: {APP_VERSION}")
        lines.append("")
        lines.append(f"Python: {sys.version.splitlines()[0]}")
        lines.append(f"Исполняемый файл: {sys.executable}")
        lines.append("")
        lines.append("Версии библиотек:")
        lines.append(f"  OpenCV (cv2): {cv2.__version__}")
        lines.append(f"  NumPy: {np.__version__}")
        lines.append(f"  Pillow: {getattr(Image, '__version__', '—')}")
        lines.append("")
        lines.append("Наличие модулей (importlib.util.find_spec):")
        for mod in (
            "thermal_decoder",
            "thermal_decoder.scale_detector",
            "thermal_decoder.thermal_decoder",
            "thermal_decoder.io_export",
            "cv2",
            "numpy",
            "PIL",
        ):
            ok = self._find_spec_ok(mod)
            lines.append(f"  {mod}: {'да' if ok else 'нет'}")
        return lines

    def _show_about_system(self) -> None:
        """Показывает сведения об ОС и железе через _show_text_dialog."""
        self._show_text_dialog("О системе", "\n".join(self._system_info_lines()))

    def _show_about_program(self) -> None:
        """Показывает версии Python, библиотек и наличие подмодулей."""
        self._show_text_dialog("О программе", "\n".join(self._program_info_lines()))

    def _software_info_report_text(self, now: datetime.datetime) -> str:
        """Текст .txt с датой создания, блоками «Система» и «Программа»."""
        now_utc = now.astimezone(datetime.timezone.utc)
        lic_status, lic_until = license_report_status_and_expiry(now=now_utc)
        lines: list[str] = [
            "Дата и время создания файла: "
            f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.isoformat(timespec='seconds')})",
            "",
            "Система",
            "",
            *self._system_info_lines(),
            "",
            "Программа",
            "",
            *self._program_info_lines(),
            "",
            "Лицензия (сертификат)",
            "",
            f"Статус: {lic_status}",
            f"Срок действия сертификата (дата окончания, UTC): {lic_until}",
        ]
        return "\n".join(lines)

    def _show_license(self) -> None:
        """Текст отказа от гарантий и условий распространения (Unlicense)."""
        body = (
            "Программа разработана в любительских целях и не предназначена для "
            "профессионального использования. Температура восстанавливается по "
            "цвету пикселя: он сопоставляется с дискретной палитрой, построенной "
            "по строкам выделенной области шкалы (выбирается ближайший уровень "
            "палитры, без интерполяции между соседними уровнями), затем значение "
            "линейно масштабируется между заданными пользователем Min и Max. "
            "Фиксированная погрешность в градусах не гарантируется: ориентир по "
            "шагу дискретизации — порядка |Max − Min| / (N − 1), где N — высота "
            "прямоугольника шкалы в пикселях; неверные Min/Max искажают всю карту "
            "пропорционально. На результат влияют шум, качество изображения и "
            "точность выделения шкалы. Для измерений с жёсткими требованиями к "
            "точности результаты следует перепроверять. Программа распространяется "
            "на условиях лицензии Unlicense."
        )
        self._show_text_dialog("Лицензия", body)

    def _build_step1(self, parent: ttk.Frame) -> None:
        """Шаг 1: выбор BMP и папки вывода."""
        ttk.Label(
            parent,
            text=(
                "Выберите термограмму в формате BMP и папку для результатов.\n"
                "Обычно там появятся оверлей и CSV; "
                "при включённом ZIP на шаге 4 — только архив после закрытия окна просмотра."
            ),
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(0, 12))

        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=6)
        ttk.Button(row, text="Открыть BMP…", command=self.on_open).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Label(row, textvariable=self.var_image_label).pack(side=tk.LEFT)

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="Папка вывода…", command=self.on_pick_outdir).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Label(row2, textvariable=self.var_outdir, wraplength=720).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

    def _build_step2(self, parent: ttk.Frame) -> None:
        """Шаг 2: ROI шкалы, Min/Max и превью с масштабом и перетаскиванием прямоугольника."""
        hint = ttk.Label(
            parent,
            text=(
                "Область цветной шкалы и значения Min/Max задаются только вручную (поля X,Y,W,H и числа; "
                "прямоугольник можно перетащить). Ctrl + колесо — масштаб; при увеличении — "
                "перемещение изображения зажатой левой кнопкой (вне прямоугольника шкалы). "
                "Двойной щелчок по кадру после анализа — температура в точке. Верх шкалы = Max."
            ),
            justify=tk.LEFT,
            style="Hint.TLabel",
        )
        hint.pack(anchor=tk.W, pady=(0, 8))

        frm_scale = ttk.LabelFrame(parent, text="Область шкалы (пиксели)", padding=10)
        frm_scale.pack(side=tk.TOP, fill=tk.X, pady=2)

        for i, (lab, var) in enumerate(
            [
                ("X", self.var_rx),
                ("Y", self.var_ry),
                ("W", self.var_rw),
                ("H", self.var_rh),
            ]
        ):
            ttk.Label(frm_scale, text=lab).grid(row=0, column=i * 2, sticky=tk.E)
            ttk.Spinbox(frm_scale, from_=0, to=20000, textvariable=var, width=8).grid(
                row=0, column=i * 2 + 1, padx=2
            )

        for v in (self.var_rx, self.var_ry, self.var_rw, self.var_rh):
            v.trace_add("write", lambda *_: self.redraw_canvas())

        frm_t = ttk.LabelFrame(parent, text="Температура min / max", padding=10)
        frm_t.pack(side=tk.TOP, fill=tk.X, pady=4)

        ttk.Label(frm_t, text="Min").grid(row=0, column=0)
        ttk.Entry(frm_t, textvariable=self.var_tmin, width=10).grid(
            row=0, column=1, padx=2
        )
        ttk.Label(frm_t, text="Max").grid(row=0, column=2)
        ttk.Entry(frm_t, textvariable=self.var_tmax, width=10).grid(
            row=0, column=3, padx=2
        )

        canvas_frm = ttk.Frame(parent)
        canvas_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        canvas_frm.rowconfigure(0, weight=1)
        canvas_frm.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frm,
            bg="#2b2b2b",
            height=480,
            highlightthickness=0,
        )
        self._step2_vscroll = ttk.Scrollbar(
            canvas_frm, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self._step2_hscroll = ttk.Scrollbar(
            canvas_frm, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=self._step2_vscroll.set,
            xscrollcommand=self._step2_hscroll.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._step2_vscroll.grid(row=0, column=1, sticky="ns")
        self._step2_hscroll.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_probe_temp)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        _bind_ctrl_mousewheel(self.canvas, self.on_step2_ctrl_wheel)

    def _build_step3(self, parent: ttk.Frame) -> None:
        """Шаг 3: проверка градиента шкалы и доп. обработка (без настроек сохранения)."""
        wrap = 880
        ttk.Label(
            parent,
            text=(
                "Параметры проверки шкалы и обработки изображения. Область цветной шкалы и "
                "Min/Max задаются на шаге 2. Сохранение файлов и архив — на шаге 4."
            ),
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=wrap,
        ).pack(anchor=tk.W, pady=(0, 12))

        lf_scale = ttk.LabelFrame(parent, text="Шкала", padding=12)
        lf_scale.pack(fill=tk.X, pady=(0, 8))
        lf_scale.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(lf_scale, text="Проверка градиента").grid(
            row=r, column=0, sticky=tk.W, pady=2
        )
        self.cmb_grad = ttk.Combobox(
            lf_scale,
            values=("Строгий", "Мягкий"),
            state="readonly",
            width=14,
        )
        self.cmb_grad.set("Строгий")
        self.cmb_grad.grid(row=r, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(
            lf_scale,
            text=(
                "Проверка согласованности цветов вдоль ROI шкалы перед расчётом матрицы. "
                "«Строгий» — меньше допуск к отклонениям; «Мягкий» — для шумных шкал."
            ),
            style="MicroHint.TLabel",
            wraplength=wrap - 48,
            justify=tk.LEFT,
        ).grid(row=r + 1, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))
        r += 2

        lf_extra = ttk.LabelFrame(parent, text="Дополнительно", padding=12)
        lf_extra.pack(fill=tk.BOTH, expand=True)
        lf_extra.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            lf_extra, text="Размытие Gauss перед анализом", variable=self.var_blur
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        ttk.Label(
            lf_extra,
            text="Сглаживает изображение перед выделением шкалы и расчётом температур (может смягчить шум).",
            style="MicroHint.TLabel",
            wraplength=wrap - 48,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky=tk.W, pady=(0, 8))

    def _build_step4(self, parent: ttk.Frame) -> None:
        """Шаг 4: куда пишутся файлы и опция ZIP-архива."""
        wrap = 880
        ttk.Label(
            parent,
            text=(
                "Кнопка «Анализ» запускает расчёт и запись результатов в папку, выбранную на шаге 1. "
                "Чтобы обработать другой BMP, вернитесь на шаг 1."
            ),
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=wrap,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(
            parent,
            text=(
                "Обычно в папке вывода создаются:\n"
                "  • result_overlay.bmp — оттенки серого исходной термограммы;\n"
                "  • data.csv — таблица температур по пикселям.\n"
                "При включённой опции ниже дополнительно создаётся текстовый файл "
                "informaciya_o_po_*.txt со сведениями о системе и программе.\n"
                "Если включён архив ZIP, отдельные файлы в папку не пишутся: после закрытия "
                "окна просмотра в папке остаётся один ZIP со всем содержимым (включая снимки "
                "с крестами, если вы нажали «Сохранить» в окне просмотра, и технологический "
                "файл, если он включён)."
            ),
            justify=tk.LEFT,
            style="Hint.TLabel",
            wraplength=wrap,
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Checkbutton(
            parent,
            text=(
                "Создать технологический файл (информация о системе и ПО) — сохраняется "
                "в папку вывода или включается в ZIP при включённом архиве"
            ),
            variable=self.var_create_tech_file,
        ).pack(anchor=tk.W, pady=(0, 12))

        lf = ttk.LabelFrame(parent, text="Архив", padding=12)
        lf.pack(fill=tk.X, pady=(0, 8))
        lf.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            lf,
            text="Создать один ZIP-архив вместо отдельных файлов",
            variable=self.var_create_archive,
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        ttk.Label(
            lf,
            text=(
                "Промежуточные файлы пишутся во временную папку; ZIP с оверлеем, CSV "
                "и при необходимости result_overlay_annotated*.bmp "
                "и технологическим informaciya_o_po_*.txt (если включён) "
                "появляется в выбранной папке после закрытия окна «Просмотр температур». "
                "Имя: имя исходного BMP и время запуска анализа."
            ),
            style="MicroHint.TLabel",
            wraplength=wrap - 24,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky=tk.W)

    def _show_step(self, n: int) -> None:
        """Переключает видимый шаг, заголовок, кнопки «Далее»/«Анализ» и при необходимости обновляет превью шага 2."""
        self._step = max(1, min(4, n))
        self.var_step_title.set(_STEP_TITLES[self._step - 1])

        if self._step == 1:
            self.step1.tkraise()
        elif self._step == 2:
            self.step2.tkraise()
            self._refresh_step2_preview()
        elif self._step == 3:
            self.step3.tkraise()
        else:
            self.step4.tkraise()

        if self._step == 1:
            self.btn_back.state(["disabled"])
            self.btn_next.pack(side=tk.LEFT, padx=4)
            self.btn_next.state(["!disabled"])
            self.btn_analyze.pack_forget()
        elif self._step == 2:
            self.btn_back.state(["!disabled"])
            self.btn_next.pack(side=tk.LEFT, padx=4)
            self.btn_next.state(["!disabled"])
            self.btn_analyze.pack_forget()
        elif self._step == 3:
            self.btn_back.state(["!disabled"])
            self.btn_next.pack(side=tk.LEFT, padx=4)
            self.btn_next.state(["!disabled"])
            self.btn_analyze.pack_forget()
        else:
            self.btn_back.state(["!disabled"])
            self.btn_next.pack_forget()
            self.btn_analyze.pack(side=tk.LEFT, padx=4)

        if not self._busy:
            self._apply_nav_enabled(True)

    def _refresh_step2_preview(self) -> None:
        """Перерисовывает изображение на шаге 2 после смены шага или данных."""
        if self._bgr is not None:
            self.root.update_idletasks()
            self._show_bgr(self._bgr)

    def _apply_nav_enabled(self, enabled: bool) -> None:
        """Включает или отключает кнопки навигации с учётом текущего шага (на шаге 1 «Назад» недоступна)."""
        if not enabled:
            for b in (
                self.btn_back,
                self.btn_next,
                self.btn_analyze,
            ):
                b.state(["disabled"])
            return
        if self._step == 1:
            self.btn_back.state(["disabled"])
        else:
            self.btn_back.state(["!disabled"])
        if self._step < 4:
            self.btn_next.state(["!disabled"])
        else:
            self.btn_analyze.state(["!disabled"])

    def on_nav_back(self) -> None:
        """Обработчик «Назад»: переход на предыдущий шаг, если не идёт анализ."""
        if self._busy:
            return
        if self._step > 1:
            self._show_step(self._step - 1)

    def on_nav_next(self) -> None:
        """Обработчик «Далее»: проверки шагов 1–2 и переходы 2→3→4."""
        if self._busy:
            return
        if self._step == 1:
            if not self.image_path:
                messagebox.showwarning(
                    "Шаг 1",
                    "Выберите файл BMP.",
                )
                return
            if not self.var_outdir.get().strip():
                messagebox.showwarning(
                    "Шаг 1",
                    "Укажите папку для сохранения результатов.",
                )
                return
            self._show_step(2)
            self.status.set(
                "Шаг 2: настройте область цветной шкалы и Min/Max, затем «Далее»."
            )
        elif self._step == 2:
            if self._bgr is None:
                messagebox.showwarning("Шаг 2", "Нет загруженного изображения.")
                return
            self._show_step(3)
            self.status.set(
                "Шаг 3: параметры анализа и визуализации, затем «Далее» к сохранению."
            )
        elif self._step == 3:
            self._show_step(4)
            self.status.set(
                "Шаг 4: при необходимости включите архив и нажмите «Анализ»."
            )

    def poll_queue(self) -> None:
        """Периодически забирает сообщения из очереди фона и планирует следующий опрос."""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                self._handle_async(kind, payload)
        except queue.Empty:
            pass
        self.root.after(120, self.poll_queue)

    def _set_busy(self, busy: bool) -> None:
        """Режим занятости: блокирует навигацию на время фонового сканирования."""
        self._busy = busy
        if busy:
            self._apply_nav_enabled(False)
        else:
            self._apply_nav_enabled(True)

    def on_canvas_configure(self, event) -> None:
        """При изменении размера canvas на шаге 2 откладывает пересчёт масштаба превью (debounce)."""
        if event.widget is not self.canvas or self._bgr is None or self._busy:
            return
        if self._step != 2:
            return
        pair = (int(event.width), int(event.height))
        if pair == self._last_configure_wh:
            return
        self._last_configure_wh = pair
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(180, self._apply_canvas_resize)

    def _apply_canvas_resize(self) -> None:
        """Отложенный пересчёт отображения после on_canvas_configure."""
        self._resize_after_id = None
        if self._bgr is not None and not self._busy and self._step == 2:
            self._show_bgr(self._bgr)

    def _finalize_deferred_archive(self, result: dict) -> None:
        """Собирает ZIP в папку пользователя и удаляет временную рабочую папку (режим «только архив»)."""
        if not result.get("deferred_archive"):
            return
        if result.get("_archive_finalized"):
            return
        target = Path(result["archive_target_dir"])
        zip_name = result["archive_zip_name"]
        work = Path(result["output_dir"])
        zip_path = target / zip_name
        try:
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(
                zip_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for p in sorted(work.iterdir()):
                    if p.is_file():
                        zf.write(p, arcname=p.name)
            result["archive_path"] = str(zip_path)
            result["_archive_finalized"] = True
            shutil.rmtree(work, ignore_errors=True)
        except OSError as zerr:
            result["archive_warning"] = str(zerr)
            messagebox.showwarning(
                "Архив",
                "Не удалось создать ZIP. Временная папка с файлами:\n"
                f"{work}\n\n{zerr}",
            )

    def _handle_async(self, kind: str, payload: object) -> None:
        """Обрабатывает результат фонового анализа: обновление статуса и окна просмотра."""
        if kind == "scan_ok":
            result = payload
            assert isinstance(result, dict)
            self._temp_matrix = result.get("temp_matrix")
            if result.get("deferred_archive"):
                ad = result.get("archive_target_dir", "")
                msg = (
                    f"Готово. INVALID: {result.get('invalid_pixel_count', 0)}. "
                    f"Режим архива: ZIP будет записан в «{ad}» после закрытия окна просмотра."
                )
            else:
                msg = (
                    f"Готово. INVALID: {result.get('invalid_pixel_count', 0)}. "
                    f"Файлы в {result.get('output_dir')}"
                )
                ap = result.get("archive_path")
                if ap:
                    msg += f". Архив: {ap}"
            self.status.set(msg)
            aw = result.get("archive_warning")
            if aw and not result.get("deferred_archive"):
                messagebox.showwarning(
                    "Архив",
                    "Анализ выполнен, но не удалось создать ZIP:\n" + str(aw),
                )
            tw = result.get("tech_file_warning")
            if tw:
                messagebox.showwarning(
                    "Технологический файл",
                    "Не удалось записать файл с информацией о ПО:\n" + str(tw),
                )
            ov = result.get("result_overlay")
            if ov and Path(ov).is_file():
                img = imread_bgr(ov)
                if img is not None:
                    self._bgr = img
                    if self._step == 2:
                        self._show_bgr(img)
            viewer_opened = self._open_result_view_window(result)
            if result.get("deferred_archive") and not viewer_opened:
                self._finalize_deferred_archive(result)
                if not result.get("archive_warning"):
                    ap2 = result.get("archive_path")
                    if ap2:
                        self.status.set(
                            f"Готово. Архив сохранён (окно просмотра не открыто): {ap2}"
                        )
            self._set_busy(False)
        elif kind == "scan_err":
            err = payload
            if isinstance(err, System_OCV_Vis_Temp_Error):
                messagebox.showerror("Ошибка шкалы", str(err))
            else:
                messagebox.showerror("Ошибка", str(err))
            self.status.set("Ошибка сканирования.")
            self._set_busy(False)

    def _open_result_view_window(self, result: dict) -> bool:
        """Открывает окно интерактивного просмотра температур по пикселям с маркерами и сохранением BMP."""
        tm = result.get("temp_matrix")
        if tm is None:
            messagebox.showwarning(
                "Просмотр температур",
                "Матрица температур недоступна в результате анализа. Повторите анализ "
                "или проверьте сообщения об ошибках.",
            )
            return False
        ip = result.get("image_path")
        if not ip:
            return False
        bgr = imread_bgr(ip)
        if bgr is None:
            messagebox.showerror("Просмотр", "Не удалось открыть исходный BMP.")
            return False
        th, tw = tm.shape[:2]
        ih, iw = bgr.shape[:2]
        if th != ih or tw != iw:
            messagebox.showerror(
                "Просмотр",
                "Размер матрицы температур не совпадает с изображением.",
            )
            return False

        deferred = bool(result.get("deferred_archive"))
        hint_save = (
            "«Сохранить» записывает цветной result_overlay_annotated.bmp и ч/б "
            "result_overlay_annotated_bw.bmp (кресты красные на ч/б). "
            "При режиме «только ZIP» эти файлы попадут в архив при закрытии окна."
            if deferred
            else "«Сохранить» записывает в папку вывода цветной result_overlay_annotated.bmp "
            "и ч/б result_overlay_annotated_bw.bmp (кресты красные, подписи как на сером). "
        )

        win = tk.Toplevel(self.root)
        win.title("Просмотр температур")
        win.geometry("1000x760")
        win.minsize(720, 560)
        win.transient(self.root)
        win.configure(bg=_UI_BG)

        ttk.Label(
            win,
            text=(
                "Исходная термограмма. ЛКМ по изображению — температура в точке "
                "(INVALID — нет соответствия палитре или область шкалы). "
                "ПКМ — поставить крестик с подписью температуры (можно несколько). "
                f"{hint_save}"
                "Ctrl + колесо — масштаб; при увеличении — перемещение зажатой ЛКМ."
            ),
            wraplength=960,
            justify=tk.LEFT,
            style="Hint.TLabel",
        ).pack(fill=tk.X, padx=12, pady=(12, 6))

        val_var = tk.StringVar(value="ЛКМ по изображению — температура")
        tk.Label(
            win,
            textvariable=val_var,
            font=_ui_font_tuple(13, "bold"),
            fg="#0f172a",
            bg=_UI_BG,
        ).pack(pady=(0, 6))

        markers: list[tuple[int, int]] = []

        frm = ttk.Frame(win, padding=(4, 0, 4, 8))
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        canvas = tk.Canvas(frm, bg="#2b2b2b", highlightthickness=0)
        vscroll = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=canvas.yview)
        hscroll = ttk.Scrollbar(frm, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(
            yscrollcommand=vscroll.set,
            xscrollcommand=hscroll.set,
        )
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")

        photo_box: list[ImageTk.PhotoImage | None] = [None]
        user_zoom: list[float] = [1.0]
        disp: dict[str, float | int] = {
            "sc": 1.0,
            "ox": 0,
            "oy": 0,
            "W": 400,
            "H": 300,
        }
        last_wh: list[tuple[int, int]] = [(0, 0)]

        def redraw(_: object | None = None) -> None:
            canvas.update_idletasks()
            vp_w = max(int(canvas.winfo_width()), 120)
            vp_h = max(int(canvas.winfo_height()), 120)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            fit = min(vp_w / iw, vp_h / ih)
            sc = max(fit * user_zoom[0], 1e-9)
            dw = max(1, int(round(iw * sc)))
            dh = max(1, int(round(ih * sc)))
            W = max(vp_w, dw)
            H = max(vp_h, dh)
            ox = (W - dw) // 2
            oy = (H - dh) // 2
            disp["sc"] = dw / float(iw)
            disp["ox"] = ox
            disp["oy"] = oy
            disp["W"] = W
            disp["H"] = H
            pil_r = pil.resize((dw, dh), Image.Resampling.LANCZOS)
            photo_box[0] = ImageTk.PhotoImage(pil_r)
            canvas.delete("all")
            canvas.create_image(ox, oy, anchor=tk.NW, image=photo_box[0])
            sc = float(disp["sc"])
            ox_i, oy_i = int(disp["ox"]), int(disp["oy"])
            k = max(2.0, min(sc * 3.5, 20.0))
            fs = max(9, int(round(min(sc * 0.6, 16))))
            font_tk = ("Segoe UI", fs, "bold")
            for mix, miy in markers:
                cx = ox_i + (mix + 0.5) * sc
                cy = oy_i + (miy + 0.5) * sc
                canvas.create_line(
                    cx - k, cy - k, cx + k, cy + k, fill="#ffcc00", width=2
                )
                canvas.create_line(
                    cx - k, cy + k, cx + k, cy - k, fill="#ffcc00", width=2
                )
                v = tm[miy, mix]
                txt = "INVALID" if np.isnan(v) else f"T={float(v):.2f}"
                tx, ty = cx + k * 0.35, cy - k * 1.15
                for dx, dy in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ):
                    canvas.create_text(
                        tx + dx,
                        ty + dy,
                        text=txt,
                        anchor="w",
                        fill="#101010",
                        font=font_tk,
                    )
                canvas.create_text(
                    tx, ty, text=txt, anchor="w", fill="#ffffff", font=font_tk
                )
            canvas.config(scrollregion=(0, 0, W, H))

        def on_configure(event: tk.Event) -> None:
            if event.widget is not canvas:
                return
            wh = (event.width, event.height)
            if wh == last_wh[0]:
                return
            last_wh[0] = wh
            redraw()

        def to_img(ex: int, ey: int) -> tuple[int, int]:
            sc = float(disp["sc"])
            if sc <= 0:
                return 0, 0
            cx = canvas.canvasx(ex) - int(disp["ox"])
            cy = canvas.canvasy(ey) - int(disp["oy"])
            ix = int(cx / sc)
            iy = int(cy / sc)
            return ix, iy

        def on_click(event: tk.Event) -> None:
            ix, iy = to_img(event.x, event.y)
            if not (0 <= ix < iw and 0 <= iy < ih):
                return
            v = tm[iy, ix]
            if np.isnan(v):
                msg = f"({ix}, {iy})  —  INVALID"
            else:
                msg = f"({ix}, {iy})  —  T = {float(v):.2f}"
            val_var.set(msg)
            self.status.set(msg)

        def on_save() -> None:
            out_dir = result.get("output_dir")
            if not out_dir:
                messagebox.showerror("Сохранить", "Нет папки вывода.")
                return
            p = Path(out_dir)
            if not p.is_dir():
                messagebox.showerror("Сохранить", f"Папка недоступна: {p}")
                return
            out_color = p / "result_overlay_annotated.bmp"
            out_bw = p / "result_overlay_annotated_bw.bmp"
            annot = bgr.copy()
            gy_base = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            annot_bw = cv2.cvtColor(gy_base, cv2.COLOR_GRAY2BGR)
            cross_red_bgr = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            th = 2
            s = 5
            for mix, miy in markers:
                fg, bg = _bw_fg_bg(gy_base, mix, miy)
                cc = int(fg)
                bgr_color = (cc, cc, cc)
                bgr_bg = (int(bg), int(bg), int(bg))
                cv2.line(
                    annot,
                    (mix - s, miy - s),
                    (mix + s, miy + s),
                    bgr_color,
                    th,
                    cv2.LINE_AA,
                )
                cv2.line(
                    annot,
                    (mix - s, miy + s),
                    (mix + s, miy - s),
                    bgr_color,
                    th,
                    cv2.LINE_AA,
                )
                cv2.line(
                    annot_bw,
                    (mix - s, miy - s),
                    (mix + s, miy + s),
                    cross_red_bgr,
                    th,
                    cv2.LINE_AA,
                )
                cv2.line(
                    annot_bw,
                    (mix - s, miy + s),
                    (mix + s, miy - s),
                    cross_red_bgr,
                    th,
                    cv2.LINE_AA,
                )
                v = tm[miy, mix]
                lbl = "INVALID" if np.isnan(v) else f"T={float(v):.2f}"
                cv2.putText(
                    annot,
                    lbl,
                    (mix + 6, miy - 6),
                    font,
                    0.45,
                    bgr_bg,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annot,
                    lbl,
                    (mix + 6, miy - 6),
                    font,
                    0.45,
                    bgr_color,
                    1,
                    cv2.LINE_AA,
                )
                bgr_lbl_bg = (int(bg), int(bg), int(bg))
                bgr_lbl_fg = (int(fg), int(fg), int(fg))
                cv2.putText(
                    annot_bw,
                    lbl,
                    (mix + 6, miy - 6),
                    font,
                    0.45,
                    bgr_lbl_bg,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annot_bw,
                    lbl,
                    (mix + 6, miy - 6),
                    font,
                    0.45,
                    bgr_lbl_fg,
                    1,
                    cv2.LINE_AA,
                )
            if not imwrite(out_color, annot) or not imwrite(out_bw, annot_bw):
                messagebox.showerror("Сохранить", "Не удалось записать BMP (права/путь).")
                return
            if result.get("deferred_archive"):
                messagebox.showinfo(
                    "Сохранить",
                    "Маркеры записаны во временную папку и войдут в ZIP при закрытии "
                    "этого окна:\n"
                    f"{out_color.name}\n{out_bw.name}",
                )
                self.status.set(
                    "Маркеры с крестами будут в архиве после закрытия окна просмотра."
                )
            else:
                messagebox.showinfo(
                    "Сохранить",
                    f"Файлы сохранены:\n{out_color}\n{out_bw}",
                )
                self.status.set(f"Сохранено: {out_color} ; {out_bw}")

        def on_right_click(event: tk.Event) -> None:
            ix, iy = to_img(event.x, event.y)
            if not (0 <= ix < iw and 0 <= iy < ih):
                return
            markers.append((ix, iy))
            v = tm[iy, ix]
            if np.isnan(v):
                short = "INVALID"
            else:
                short = f"T={float(v):.2f}"
            val_var.set(f"Маркер: ({ix}, {iy})  —  {short}")
            self.status.set(f"Маркер добавлен: ({ix}, {iy})  {short}")
            redraw()

        ptr: list[tuple[int, int] | None] = [None]
        drag_canvas: list[bool] = [False]

        def on_b1_press(event: tk.Event) -> None:
            ptr[0] = (event.x, event.y)
            drag_canvas[0] = False
            Wv, Hv = int(disp["W"]), int(disp["H"])
            vp_wl = max(canvas.winfo_width(), 1)
            vp_hl = max(canvas.winfo_height(), 1)
            if Wv > vp_wl or Hv > vp_hl:
                canvas.scan_mark(event.x, event.y)
                canvas.config(cursor="hand2")

        def on_b1_motion(event: tk.Event) -> None:
            if ptr[0] is None:
                return
            px, py = ptr[0]
            if not drag_canvas[0]:
                if abs(event.x - px) <= 2 and abs(event.y - py) <= 2:
                    return
                drag_canvas[0] = True
            Wv, Hv = int(disp["W"]), int(disp["H"])
            vp_wl = max(canvas.winfo_width(), 1)
            vp_hl = max(canvas.winfo_height(), 1)
            if Wv > vp_wl or Hv > vp_hl:
                canvas.scan_dragto(event.x, event.y, gain=1)

        def on_b1_release(event: tk.Event) -> None:
            canvas.config(cursor="")
            if ptr[0] is None:
                return
            was_drag = drag_canvas[0]
            ptr[0] = None
            drag_canvas[0] = False
            if not was_drag:
                on_click(event)

        def on_ctrl_wheel(event: tk.Event) -> None:
            d = getattr(event, "delta", 0) or 0
            if d == 0:
                return
            sc_old = float(disp["sc"])
            ox, oy = int(disp["ox"]), int(disp["oy"])
            cx = canvas.canvasx(event.x) - ox
            cy = canvas.canvasy(event.y) - oy
            if sc_old <= 0:
                return
            ix = cx / sc_old
            iy = cy / sc_old

            if d > 0:
                user_zoom[0] *= 1.1
            else:
                user_zoom[0] /= 1.1
            user_zoom[0] = max(0.25, min(16.0, user_zoom[0]))

            redraw()
            sc_new = float(disp["sc"])
            nox, noy = int(disp["ox"]), int(disp["oy"])
            target_x = nox + ix * sc_new
            target_y = noy + iy * sc_new
            Wv, Hv = int(disp["W"]), int(disp["H"])
            vp_wl = max(canvas.winfo_width(), 1)
            vp_hl = max(canvas.winfo_height(), 1)
            if Wv > vp_wl:
                frac = (target_x - event.x) / float(max(Wv, 1))
                canvas.xview_moveto(max(0.0, min(1.0, frac)))
            if Hv > vp_hl:
                frac_y = (target_y - event.y) / float(max(Hv, 1))
                canvas.yview_moveto(max(0.0, min(1.0, frac_y)))

        canvas.bind("<Configure>", on_configure)
        canvas.bind("<ButtonPress-1>", on_b1_press)
        canvas.bind("<B1-Motion>", on_b1_motion)
        canvas.bind("<ButtonRelease-1>", on_b1_release)
        canvas.bind("<ButtonPress-3>", on_right_click)
        _bind_ctrl_mousewheel(canvas, on_ctrl_wheel)
        btn_bar = ttk.Frame(win, padding=(8, 0, 8, 12))
        btn_bar.pack(fill=tk.X)
        ttk.Button(
            btn_bar, text="Сохранить", command=on_save, style="Accent.TButton"
        ).pack(side=tk.LEFT)

        if deferred:

            def _on_viewer_close() -> None:
                self._finalize_deferred_archive(result)
                if not result.get("archive_warning"):
                    ap = result.get("archive_path")
                    if ap:
                        self.status.set(f"Архив сохранён: {ap}")
                win.destroy()

            win.protocol("WM_DELETE_WINDOW", _on_viewer_close)
        win.after(80, redraw)
        return True

    def on_open(self) -> None:
        """Диалог выбора BMP: загрузка в память, сброс матрицы температур, начальная область шкалы."""
        p = filedialog.askopenfilename(
            filetypes=[("BMP", "*.bmp"), ("Все файлы", "*.*")]
        )
        if not p:
            return
        if Path(p).suffix.lower() != ".bmp":
            messagebox.showerror(
                "Файл",
                "Поддерживаются только файлы .bmp.\nВыберите термограмму в формате BMP.",
            )
            return
        self.image_path = p
        bgr = imread_bgr(p)
        if bgr is None:
            messagebox.showerror("Файл", "Не удалось прочитать BMP.")
            return
        self._bgr = bgr
        self._temp_matrix = None
        self._step2_user_zoom = 1.0
        h, w = bgr.shape[:2]
        rw_init = max(20, min(60, w // 15))
        self.var_rh.set(h)
        self.var_rw.set(rw_init)
        self.var_rx.set(max(0, w - rw_init - 4))
        self.var_ry.set(0)
        self.var_image_label.set(str(Path(p)))
        self.status.set(f"Выбран файл: {p}")
        if self._step == 2:
            self._show_bgr(bgr)

    def on_pick_outdir(self) -> None:
        """Диалог выбора папки для result_overlay.bmp и data.csv."""
        d = filedialog.askdirectory()
        if d:
            self.var_outdir.set(d)

    def _parse_float(self, s: str) -> float:
        """Разбор числа из поля ввода с поддержкой десятичной запятой."""
        return float(s.strip().replace(",", "."))

    def on_scan(self) -> None:
        """Запускает ThermalDecoder.scan_ocv в отдельном потоке; результат попадает в очередь для GUI-потока."""
        if self._step != 4:
            messagebox.showinfo("", "Запуск анализа доступен на шаге 4.")
            return
        if not self.image_path:
            messagebox.showwarning("", "Откройте BMP (шаг 1).")
            return
        out = self.var_outdir.get().strip()
        if not out:
            messagebox.showwarning("", "Выберите папку вывода (шаг 1).")
            return
        try:
            tmin = self._parse_float(self.var_tmin.get())
            tmax = self._parse_float(self.var_tmax.get())
        except ValueError:
            messagebox.showerror("", "Некорректные Min/Max.")
            return

        strict = _gradient_strict_from_ui(self.cmb_grad.get())
        # scan_ocv пока не строит цветовой оверлей/сетку из io_export.build_overlay
        overlay = "both"
        cmap = "JET"
        gstep = default_grid_step_px
        rect = self._get_scale_rect_values()
        want_archive = self.var_create_archive.get()
        want_tech_file = self.var_create_tech_file.get()
        image_path = self.image_path

        def task() -> None:
            work_dir: str | None = None
            try:
                stem = Path(image_path).stem or "thermal"
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if want_archive:
                    work_dir = tempfile.mkdtemp(prefix="thermaldec_")
                    out_dir = work_dir
                else:
                    out_dir = out
                dec = ThermalDecoder()
                result = dec.scan_ocv(
                    Path(image_path),
                    output_dir=out_dir,
                    scale_rect=rect,
                    min_temp=tmin,
                    max_temp=tmax,
                    apply_blur=self.var_blur.get(),
                    auto_detect_scale=False,
                    gradient_strict=strict,
                    overlay_mode=overlay,
                    colormap_name=cmap,
                    grid_step=gstep,
                    include_temp_matrix=True,
                )
                if want_tech_file:
                    try:
                        now = datetime.datetime.now().astimezone()
                        tech_path = Path(out_dir) / f"informaciya_o_po_{ts}.txt"
                        tech_path.write_text(
                            self._software_info_report_text(now),
                            encoding="utf-8",
                            newline="\n",
                        )
                    except OSError as e:
                        result["tech_file_warning"] = str(e)
                if want_archive:
                    result["deferred_archive"] = True
                    result["archive_target_dir"] = out
                    result["archive_zip_name"] = f"{stem}_{ts}.zip"
                else:
                    result["deferred_archive"] = False
                self._queue.put(("scan_ok", result))
            except Exception as e:
                if work_dir:
                    shutil.rmtree(work_dir, ignore_errors=True)
                self._queue.put(("scan_err", e))

        self.status.set("Сканирование…")
        self._set_busy(True)
        threading.Thread(target=task, daemon=True).start()

    def _img_to_disp(self, ix: int, iy: int) -> tuple[int, int]:
        """Координаты пикселя кадра в координаты canvas (с учётом масштаба и отступа)."""
        sc = self._disp_scale
        if sc <= 0:
            return self._disp_off_x, self._disp_off_y
        dx = int(round(ix * sc + self._disp_off_x))
        dy = int(round(iy * sc + self._disp_off_y))
        return dx, dy

    def _disp_to_img(self, dx: int, dy: int) -> tuple[int, int]:
        """Координаты события мыши на canvas в координаты пикселя исходного изображения."""
        sc = self._disp_scale
        if sc <= 0:
            return 0, 0
        cx = self.canvas.canvasx(dx) - self._disp_off_x
        cy = self.canvas.canvasy(dy) - self._disp_off_y
        ix = int(cx / sc)
        iy = int(cy / sc)
        return ix, iy

    def on_step2_ctrl_wheel(self, event: tk.Event) -> None:
        """Ctrl+колесо на шаге 2: масштаб с сохранением точки под курсором и прокруткой области просмотра."""
        if self._bgr is None or self._busy or self._step != 2:
            return
        d = getattr(event, "delta", 0) or 0
        if d == 0:
            return
        sc = self._disp_scale
        ox, oy = self._disp_off_x, self._disp_off_y
        cx = self.canvas.canvasx(event.x) - ox
        cy = self.canvas.canvasy(event.y) - oy
        if sc <= 0:
            return
        ix = cx / sc
        iy = cy / sc

        if d > 0:
            self._step2_user_zoom *= 1.1
        else:
            self._step2_user_zoom /= 1.1
        self._step2_user_zoom = max(0.25, min(16.0, self._step2_user_zoom))

        self._show_bgr(self._bgr)

        nsc = self._disp_scale
        nox, noy = self._disp_off_x, self._disp_off_y
        target_x = nox + ix * nsc
        target_y = noy + iy * nsc
        W, H = self._step2_scroll_wh
        vp_w = max(self.canvas.winfo_width(), 1)
        vp_h = max(self.canvas.winfo_height(), 1)
        if W > vp_w:
            frac = (target_x - event.x) / float(max(W, 1))
            self.canvas.xview_moveto(max(0.0, min(1.0, frac)))
        if H > vp_h:
            frac_y = (target_y - event.y) / float(max(H, 1))
            self.canvas.yview_moveto(max(0.0, min(1.0, frac_y)))

    def _show_bgr(self, bgr: np.ndarray) -> None:
        """Вписывает кадр в область canvas, считает масштаб и вызывает redraw_canvas."""
        self._bgr = bgr
        self.canvas.update_idletasks()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        vp_w = max(self.canvas.winfo_width(), 400)
        vp_h = max(self.canvas.winfo_height(), 300)
        iw, ih = pil.size
        if iw <= 0 or ih <= 0:
            return
        fit = min(vp_w / iw, vp_h / ih)
        sc = max(fit * self._step2_user_zoom, 1e-9)
        dw = max(1, int(round(iw * sc)))
        dh = max(1, int(round(ih * sc)))
        W = max(vp_w, dw)
        H = max(vp_h, dh)
        ox = (W - dw) // 2
        oy = (H - dh) // 2
        self._disp_scale = dw / float(iw)
        self._disp_off_x = ox
        self._disp_off_y = oy
        self._step2_scroll_wh = (W, H)
        pil_r = pil.resize((dw, dh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_r)
        self.canvas.config(scrollregion=(0, 0, W, H))
        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        """Рисует изображение и зелёный прямоугольник ROI шкалы на canvas шага 2."""
        self.canvas.delete("all")
        if self._photo is None:
            return
        self.canvas.create_image(
            self._disp_off_x, self._disp_off_y, anchor=tk.NW, image=self._photo
        )
        if self._bgr is None:
            return
        ih, iw = self._bgr.shape[:2]
        rx, ry, rw, rh = self._get_scale_rect_values()
        x1, y1 = self._img_to_disp(rx, ry)
        x2, y2 = self._img_to_disp(min(iw, rx + rw), min(ih, ry + rh))
        self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#00ff00", width=2, tags="scale_rect"
        )

    def _step2_can_pan(self) -> bool:
        """True, если изображение больше видимой области и доступна прокрутка/перетаскивание."""
        W, H = self._step2_scroll_wh
        vp_w = max(self.canvas.winfo_width(), 1)
        vp_h = max(self.canvas.winfo_height(), 1)
        return W > vp_w or H > vp_h

    def on_canvas_press(self, event) -> None:
        """ЛКМ: начало перетаскивания ROI шкалы или панорамирования canvas."""
        if self._bgr is None:
            return
        self._step2_pan_active = False
        ix, iy = self._disp_to_img(event.x, event.y)
        rx, ry, rw, rh = self._get_scale_rect_values()
        if rx <= ix < rx + rw and ry <= iy < ry + rh:
            self._drag_anchor = (ix - rx, iy - ry)
            return
        self._drag_anchor = None
        if self._step2_can_pan():
            self.canvas.scan_mark(event.x, event.y)
            self._step2_pan_active = True
            self.canvas.config(cursor="hand2")

    def on_canvas_motion(self, event) -> None:
        """Движение мыши: панорама или смещение прямоугольника шкалы."""
        if self._bgr is None:
            return
        if self._step2_pan_active:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return
        if self._drag_anchor is None:
            return
        ix, iy = self._disp_to_img(event.x, event.y)
        ax, ay = self._drag_anchor
        nw_x = ix - ax
        nw_y = iy - ay
        ih, iw = self._bgr.shape[:2]
        _, _, rw, rh = self._get_scale_rect_values()
        nw_x = max(0, min(nw_x, iw - rw))
        nw_y = max(0, min(nw_y, ih - rh))
        self.var_rx.set(nw_x)
        self.var_ry.set(nw_y)

    def on_canvas_release(self, event) -> None:
        """Отпускание ЛКМ: сброс режима панорамы и якоря перетаскивания."""
        if self._step2_pan_active:
            self.canvas.config(cursor="")
        self._step2_pan_active = False
        self._drag_anchor = None

    def on_canvas_probe_temp(self, event) -> None:
        """Двойной щелчок: температура в точке из сохранённой матрицы (после анализа)."""
        if self._temp_matrix is None or self._bgr is None:
            self.status.set("Сначала выполните анализ; двойной щелчок — температура.")
            return
        ix, iy = self._disp_to_img(event.x, event.y)
        h, w = self._temp_matrix.shape
        if not (0 <= ix < w and 0 <= iy < h):
            return
        v = self._temp_matrix[iy, ix]
        if np.isnan(v):
            self.status.set(f"Точка ({ix}, {iy}): INVALID")
        else:
            self.status.set(f"Точка ({ix}, {iy}): T = {float(v):.4f}")

    def run(self) -> None:
        """Запускает главный цикл Tkinter."""
        self.root.mainloop()


def run_app() -> None:
    """Точка входа: создаёт приложение и передаёт управление mainloop (вызывается из main.py)."""
    app = ThermalDecoderApp()
    app.run()
