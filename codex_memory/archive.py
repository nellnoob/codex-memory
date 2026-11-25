from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
from uuid import UUID

from .store import project_key, stable_id

LOG = logging.getLogger(__name__)
BATCH_LINES = 128
CONTEXT = re.compile(r"\s*<environment_context>.*?</environment_context>\s*", re.DOTALL)


def record(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def message(entry: dict):
    payload = entry.get("payload")
    if entry.get("type") != "response_item" or not isinstance(payload, dict):
        return None
    role = payload.get("role")
    if payload.get("type") != "message" or role not in ("user", "assistant"):
        return None
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    blocks = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in ("input_text", "output_text", "text"):
            continue
        text = part.get("text")
        if isinstance(text, str) and not (role == "user" and CONTEXT.fullmatch(text)):
            blocks.append(text)
    body = "\n".join(blocks)
    return (role, body) if body.strip() else None


def tail_hash(source, offset):
    source.seek(max(0, offset - 256))
    return hashlib.sha256(source.read(min(offset, 256))).digest()


def sync_file(store, path: Path, totals: Counter) -> None:
    with path.open("rb") as source:
        first = source.readline()
        if not first.endswith(b"\n"):
            return
        try:
            header = record(first)
            meta = header["payload"]
            if header["type"] != "session_meta" or not isinstance(meta, dict):
                raise ValueError("Expected session_meta")
            thread = str(UUID(meta["id"]))
            cwd = meta.get("cwd")
            project = project_key(cwd) if isinstance(cwd, str) else ""
        except (KeyError, TypeError, ValueError, AttributeError):
            totals["unsupported_files"] += 1
            return
        key = project_key(str(path))
        totals["files"] += 1
        while True:
            with store.write() as db:
                cursor = db.execute("SELECT * FROM cursors WHERE path=?", (key,)).fetchone()
                end = source.seek(0, 2)
                offset, lines = len(first), 1
                if cursor and len(first) <= cursor["offset"] <= end:
                    if tail_hash(source, cursor["offset"]) == cursor["anchor"]:
                        offset, lines = cursor["offset"], cursor["lines"]
                source.seek(offset)
                count = 0
                while count < BATCH_LINES and source.tell() < end:
                    raw = source.readline(end - source.tell())
                    if not raw.endswith(b"\n"):
                        break
                    offset = source.tell()
                    count += 1
                    lines += 1
                    try:
                        entry = record(raw)
                    except ValueError:
                        totals["invalid_lines"] += 1
                        continue
                    item = message(entry)
                    if item is None:
                        continue
                    role, body = item
                    date = entry.get("timestamp")
                    if not isinstance(date, str):
                        date = meta.get("timestamp")
                    inserted = store.insert(
                        db, id=stable_id("m_", thread, lines, role, body), kind="message",
                        thread_id=thread, project=project, line=lines, role=role, body=body,
                        created_at=date if isinstance(date, str) else "",
                    )
                    totals["messages_added"] += int(inserted)
                if count or cursor is None or cursor["offset"] != offset:
                    db.execute(
                        """INSERT INTO cursors(path,offset,lines,anchor) VALUES (?,?,?,?)
                           ON CONFLICT(path) DO UPDATE SET offset=excluded.offset,
                               lines=excluded.lines, anchor=excluded.anchor""",
                        (key, offset, lines, tail_hash(source, offset)),
                    )
            if count < BATCH_LINES:
                return


def sync(store, sources: list[Path]) -> dict:
    totals = Counter(files=0, messages_added=0, invalid_lines=0, unsupported_files=0, read_errors=0)
    files = set()
    for path in sources:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(file.resolve() for file in path.rglob("rollout-*.jsonl") if file.is_file())
        else:
            raise ValueError(f"Archive source does not exist: {path}")
    for path in sorted(files):
        try:
            sync_file(store, path, totals)
        except OSError as error:
            LOG.warning("Unable to read %s (%s)", path.name, type(error).__name__)
            totals["read_errors"] += 1
    return dict(totals)
