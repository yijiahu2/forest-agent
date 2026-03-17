from __future__ import annotations

import fcntl
import os
import pty
import shlex
import struct
import subprocess
import sys
import termios
from typing import Dict, List, Optional, Sequence


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _copy_tty_winsize(slave_fd: int) -> None:
    for stream in (sys.stdout, sys.stdin):
        try:
            if not stream.isatty():
                continue
            winsize = fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
            rows, cols, _, _ = struct.unpack("HHHH", winsize)
            os.environ.setdefault("LINES", str(rows))
            os.environ.setdefault("COLUMNS", str(cols))
            return
        except Exception:
            continue


def _forward_chunk(chunk: bytes) -> None:
    try:
        os.write(sys.stdout.fileno(), chunk)
        return
    except Exception:
        pass

    try:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        return
    except Exception:
        pass

    sys.stdout.write(chunk.decode(errors="replace"))
    sys.stdout.flush()


def run_streaming(
    cmd: Sequence[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    print_cmd: bool = True,
    cmd_label: str = "===== CMD =====",
) -> subprocess.CompletedProcess[str]:
    if print_cmd:
        print(f"\n{cmd_label}")
        print(_format_cmd(cmd))

    master_fd, slave_fd = pty.openpty()
    chunks: List[bytes] = []

    try:
        _copy_tty_winsize(slave_fd)
        proc = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break

            if not chunk:
                break

            chunks.append(chunk)
            _forward_chunk(chunk)
    finally:
        os.close(master_fd)

    returncode = proc.wait()
    stdout = b"".join(chunks).decode(errors="replace")
    return subprocess.CompletedProcess(list(cmd), returncode, stdout, "")
