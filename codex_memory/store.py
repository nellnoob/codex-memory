from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import heapq
from itertools import islice
import os
from pathlib import Path
import re
import sqlite3
from uuid import UUID

from .embedding import DIMENSIONS, MODEL_ID

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('message', 'note')),
    thread_id TEXT,
    project TEXT NOT NULL,
    created_at TEXT NOT NULL,
    role TEXT,
    line INTEGER,
    source_message_id TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_thread ON items(thread_id, line, id);
CREATE INDEX IF NOT EXISTS items_project ON items(project, kind);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES items(id),
    start INTEGER NOT NULL,
    end INTEGER NOT NULL,
    vector BLOB CHECK(vector IS NULL OR length(vector) = 1536)
);
CREATE INDEX IF NOT EXISTS chunks_pending ON chunks(id) WHERE vector IS NULL;
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text USING fts5(text, tokenize='trigram');
CREATE TABLE IF NOT EXISTS cursors (
    path TEXT PRIMARY KEY,
    offset INTEGER NOT NULL,
    lines INTEGER NOT NULL,
    anchor BLOB NOT NULL
);
"""


def project_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)) if path else ""


def stable_id(prefix: str, *parts) -> str:
    return prefix + hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:32]


def spans(body: str):
    start = 0
    while start < len(body):
        end = min(start + 650, len(body))
        if end < len(body):
            split = max(body.rfind(mark, start + 400, end) for mark in ("\n", "。", " "))
            if split != -1:
                end = split + 1
        if body[start:end].strip():
            yield start, end
        if end == len(body):
            break
        start = end - 75


def terms(query: str) -> list[str]:
    candidates = [query]
    for word in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9_][\w./:\\-]*", query, re.ASCII):
        if len(word) <= 7 or not "\u3400" <= word[0] <= "\u9fff":
            candidates.append(word)
        else:
            candidates.extend(word[i : i + 3] for i in range(len(word) - 2))
    return list(dict.fromkeys(x for x in candidates if len(x) >= 3))[:24]


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA)
            db.executemany(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                [("schema", "1"), ("embedding", MODEL_ID)],
            )
            metadata = dict(db.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema") != "1" or metadata.get("embedding") != MODEL_ID:
                raise RuntimeError("Incompatible database schema or embedding model")
            db.commit()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def write(self):
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()

    @staticmethod
    def insert(db, *, id, kind, thread_id, project, created_at, body,
               role=None, line=None, title="", source_message_id=None) -> bool:
        inserted = db.execute(
            "INSERT OR IGNORE INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (id, kind, thread_id, project, created_at, role, line, source_message_id, title, body),
        )
        if not inserted.rowcount:
            return False
        for start, end in spans(body):
            chunk = db.execute(
                "INSERT INTO chunks(item_id, start, end) VALUES (?, ?, ?)", (id, start, end)
            )
            text = "\n".join(filter(None, (title, body[start:end])))
            db.execute("INSERT INTO chunk_text(rowid, text) VALUES (?, ?)", (chunk.lastrowid, text))
        return True

    def save(self, title, body, project="", source_thread_id=None, source_message_id=None):
        title, body = title.strip(), body.strip()
        if not title or not body or len(title) > 160 or len(body) > 12000:
            raise ValueError("Supply a title of 1–160 characters and a body of 1–12000 characters.")
        project = project_key(project)
        if source_thread_id:
            source_thread_id = str(UUID(source_thread_id))
        with self.write() as db:
            if source_message_id:
                source = db.execute(
                    "SELECT thread_id, project FROM items WHERE id=? AND kind='message'",
                    (source_message_id,),
                ).fetchone()
                if source is None:
                    raise ValueError("Unknown source message ID")
                if source_thread_id and source_thread_id != source["thread_id"]:
                    raise ValueError("Source message belongs to another thread")
                source_thread_id = source["thread_id"]
                project = project or source["project"]
            if source_thread_id and not project:
                known = db.execute(
                    "SELECT project FROM items WHERE thread_id=? AND kind='message' LIMIT 1",
                    (source_thread_id,),
                ).fetchone()
                project = known["project"] if known else ""
            id = stable_id("n_", title, body, project, source_thread_id, source_message_id)
            inserted = self.insert(
                db, id=id, kind="note", thread_id=source_thread_id, project=project,
                created_at=datetime.now(timezone.utc).isoformat(), title=title, body=body,
                source_message_id=source_message_id,
            )
        return {"id": id, "created": inserted, "thread_id": source_thread_id, "project": project}

    def read(self, id, offset=0, max_chars=3500):
        if offset < 0 or not 1 <= max_chars <= 6000:
            raise ValueError("Invalid page bounds")
        with closing(self.connect()) as db:
            row = db.execute(
                """SELECT id,kind,thread_id,project,created_at,role,line,title,source_message_id,
                          length(body) AS total_chars, substr(body, ?, ?) AS text
                   FROM items WHERE id=?""",
                (offset + 1, max_chars, id),
            ).fetchone()
        if row is None:
            raise ValueError("Unknown item ID")
        result = dict(row)
        result["offset"] = offset
        result["next_offset"] = offset + max_chars if offset + max_chars < row["total_chars"] else None
        return result

    def thread(self, thread_id, from_line=1, limit=12):
        if from_line < 1 or not 1 <= limit <= 30:
            raise ValueError("Invalid page bounds")
        with closing(self.connect()) as db:
            rows = db.execute(
                """SELECT id,line,role,project,created_at,substr(body,1,360) AS preview
                   FROM items WHERE kind='message' AND thread_id=? AND line>=?
                   ORDER BY line,id LIMIT ?""",
                (thread_id, from_line, limit + 1),
            ).fetchall()
        return {
            "thread_id": thread_id,
            "messages": [dict(row) for row in rows[:limit]],
            "next_line": rows[limit]["line"] if len(rows) > limit else None,
        }

    def status(self):
        with closing(self.connect()) as db:
            counts = dict(db.execute("SELECT kind,count(*) FROM items GROUP BY kind"))
            files = db.execute("SELECT count(*) FROM cursors").fetchone()[0]
            total, indexed = db.execute("SELECT count(*),count(vector) FROM chunks").fetchone()
        return dict(
            db=str(self.path), files=files, messages=counts.get("message", 0),
            notes=counts.get("note", 0), chunks=total, embeddings=indexed,
            pending_embeddings=total - indexed, model=MODEL_ID,
        )

    def index_batch(self, engine, limit=8):
        with closing(self.connect()) as db:
            rows = db.execute(
                """SELECT c.id, f.text FROM chunks c JOIN chunk_text f ON f.rowid=c.id
                   WHERE c.vector IS NULL ORDER BY c.id LIMIT ?""", (limit,)
            ).fetchall()
        if not rows:
            return 0
        vectors = engine.encode([row["text"] for row in rows])
        with self.write() as db:
            update = db.executemany(
                "UPDATE chunks SET vector=? WHERE id=? AND vector IS NULL",
                [(vector.tobytes(), row["id"]) for row, vector in zip(rows, vectors, strict=True)],
            )
        return update.rowcount

    def search(self, query, *, limit=8, project="", kind="", mode="hybrid", engine=None):
        query = query.strip()
        if not query or len(query) > 1200 or not 1 <= limit <= 20:
            raise ValueError("Supply a 1–1200 character query and a limit of 1–20")
        if kind not in ("", "note", "message") or mode not in ("hybrid", "keyword", "vector"):
            raise ValueError("Invalid search kind or mode")
        scope, params = "", []
        if project:
            scope += " AND i.project=?"
            params.append(project_key(project))
        if kind:
            scope += " AND i.kind=?"
            params.append(kind)
        vector = engine.encode([query])[0] if mode != "keyword" else None
        size = limit * 12
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            keyword = self._keyword(db, query, scope, params, size) if mode != "vector" else []
            semantic = self._semantic(db, vector, scope, params, size) if vector is not None else []
            hits = self._hits(db, keyword, semantic, limit)
        return {"mode": mode, "hits": hits}

    @staticmethod
    def _keyword(db, query, scope, params, size):
        join = """ FROM chunk_text JOIN chunks c ON c.id=chunk_text.rowid
                   JOIN items i ON i.id=c.item_id WHERE """
        tokens = terms(query)
        if tokens:
            match = " OR ".join('"' + term.replace('"', '""') + '"' for term in tokens)
            rows = db.execute(
                "SELECT c.id" + join + "chunk_text MATCH ?" + scope
                + " ORDER BY instr(lower(chunk_text.text),lower(?))=0, bm25(chunk_text), c.id LIMIT ?",
                [match, *params, query, size],
            ).fetchall()
            if rows:
                return [row[0] for row in rows]
        rows = db.execute(
            "SELECT c.id" + join + "instr(lower(chunk_text.text),lower(?))>0" + scope
            + " ORDER BY i.created_at DESC,c.id LIMIT ?", [query, *params, size]
        )
        return [row[0] for row in rows]

    @staticmethod
    def _semantic(db, query, scope, params, size):
        import numpy as np

        rows = db.execute(
            """SELECT c.id,c.vector FROM chunks c JOIN items i ON i.id=c.item_id
               WHERE c.vector IS NOT NULL""" + scope, params
        )
        best = []
        while batch := list(islice(rows, 512)):
            matrix = np.frombuffer(b"".join(row[1] for row in batch), dtype="<f4").reshape(-1, DIMENSIONS)
            scores = np.einsum("ij,j->i", matrix, query)
            best = heapq.nlargest(size, [*best, *((float(s), -row[0]) for s, row in zip(scores, batch))])
        return [(-negative_id, score) for score, negative_id in best]

    @staticmethod
    def _hits(db, keyword, semantic, limit):
        scores = defaultdict(float)
        for rank, id in enumerate(keyword, 1):
            scores[id] += 1.2 / (20 + rank)
        for rank, (id, _) in enumerate(semantic, 1):
            scores[id] += 1 / (20 + rank)
        if not scores:
            return []
        semantic_scores = dict(semantic)
        placeholders = ",".join("?" for _ in scores)
        rows = db.execute(
            """SELECT c.id AS chunk_id, c.start,c.end,i.id,i.kind,i.thread_id,i.role,i.line,
                      i.project,i.title,i.created_at,i.source_message_id,
                      substr(i.body,c.start+1,min(c.end-c.start,550)) AS snippet
               FROM chunks c JOIN items i ON i.id=c.item_id
               WHERE c.id IN (""" + placeholders + ")", list(scores)
        )
        by_id = {row["chunk_id"]: dict(row) for row in rows}
        hits, seen = [], set()
        for id in sorted(scores, key=lambda id: (-scores[id], id)):
            row = by_id[id]
            if row["id"] not in seen:
                seen.add(row["id"])
                row["cosine"] = round(semantic_scores[id], 4) if id in semantic_scores else None
                del row["chunk_id"]
                hits.append(row)
                if len(hits) == limit:
                    break
        return hits
