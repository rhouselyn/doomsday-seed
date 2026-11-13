"""Process locks released by the OS when their owning process exits."""
from contextlib import contextmanager
from errno import EACCES, EAGAIN, EDEADLK
import os
import time


class LockBusy(Exception):
    pass


@contextmanager
def file_lock(path, *, wait=False):
    deadline = time.monotonic() + wait if wait is not True else float('inf')
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            def acquire():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in (EACCES, EAGAIN, EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise LockBusy("另一个进程正在操作种子库") from exc
                time.sleep(0.1)
        # Closing the handle releases the lock. Keep the file to preserve its identity.
        yield
