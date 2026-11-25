from contextlib import redirect_stderr, redirect_stdout
import errno
import msvcrt
import os
from pathlib import Path
import sys
import traceback


def main():
    project = Path(__file__).resolve().parent
    logs = project / "logs"
    logs.mkdir(exist_ok=True)

    with (logs / "watch-start.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno == errno.EACCES:
                return
            raise

        with (
            (logs / "watch.stdout.log").open("a", encoding="utf-8") as out,
            (logs / "watch.stderr.log").open("a", encoding="utf-8") as err,
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            try:
                from codex_memory.cli import main as run

                codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
                (logs / "watch.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
                sys.argv[1:] = ["watch", "--codex-home", str(codex_home)]
                run()
            except Exception:
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
