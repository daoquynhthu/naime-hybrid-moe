import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Callable


@dataclass
class StopSignalMonitor:
    """Convert ordinary stop signals into cooperative training shutdown.

    Python normally raises KeyboardInterrupt for Ctrl+C, which can land in the
    middle of backward or checkpoint serialization. This monitor records the
    signal instead, allowing the training loop to finish the current safe unit
    of work and save a complete checkpoint.

    On Windows, additionally installs a SetConsoleCtrlHandler for
    CTRL_CLOSE_EVENT / CTRL_LOGOFF_EVENT / CTRL_SHUTDOWN_EVENT so that
    console window close, user logoff and system shutdown trigger a graceful
    checkpoint save.
    """

    requested: bool = False
    count: int = 0
    last_signal: str = ""
    first_seen_at: float = 0.0
    _previous: dict[int, Callable[[int, FrameType | None], object] | int | None] = field(default_factory=dict)
    _windows_cleanup: list[Callable[[], Any]] = field(default_factory=list)

    def install(self, stop_file: str | Path | None = None) -> None:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (OSError, ValueError):
                continue
        if os.name == "nt":
            self._install_windows_events(stop_file)

    def restore(self) -> None:
        for cleanup in reversed(self._windows_cleanup):
            try:
                cleanup()
            except Exception:
                pass
        self._windows_cleanup.clear()
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                continue
        self._previous.clear()

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        self._record(_signal_name(signum))

    def _record(self, name: str) -> None:
        self.requested = True
        self.count += 1
        self.last_signal = name
        if self.first_seen_at <= 0.0:
            self.first_seen_at = time.time()

    @property
    def reason(self) -> str:
        if not self.requested:
            return ""
        suffix = f" x{self.count}" if self.count > 1 else ""
        return f"{self.last_signal}{suffix}"

    # ------------------------------------------------------------------
    #  Windows native event handling
    # ------------------------------------------------------------------

    def _install_windows_events(self, stop_file: str | Path | None) -> None:
        try:
            self._install_windows_console_handler()
        except Exception:
            pass
        if stop_file:
            try:
                self._install_stop_file_polling(Path(stop_file))
            except Exception:
                pass

    # -- SetConsoleCtrlHandler -------------------------------------------------

    def _install_windows_console_handler(self) -> None:
        import ctypes
        from ctypes import wintypes

        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        @handler_type
        def ctrl_handler(event_type: int) -> int:
            if event_type == CTRL_CLOSE_EVENT:
                self._record("CTRL_CLOSE_EVENT")
            elif event_type == CTRL_LOGOFF_EVENT:
                self._record("CTRL_LOGOFF_EVENT")
            elif event_type == CTRL_SHUTDOWN_EVENT:
                self._record("CTRL_SHUTDOWN_EVENT")
            else:
                return False
            deadline = time.time() + 5.0
            while self.requested and time.time() < deadline:
                time.sleep(0.1)
            return True

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCtrlHandler(ctrl_handler, True)
        self._windows_cleanup.append(lambda: kernel32.SetConsoleCtrlHandler(ctrl_handler, False))

    # -- STOP file background polling ------------------------------------------

    def _install_stop_file_polling(self, stop_file: Path) -> None:
        stop_event = threading.Event()

        def _poll() -> None:
            while not stop_event.is_set():
                if stop_file.exists():
                    self._record(f"STOP_FILE:{stop_file.name}")
                    return
                stop_event.wait(2.0)

        thread = threading.Thread(target=_poll, name="naime-stop-poll", daemon=True)
        thread.start()

        def _cleanup() -> None:
            stop_event.set()

        self._windows_cleanup.append(_cleanup)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal_{signum}"
