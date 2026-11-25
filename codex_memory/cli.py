import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from time import sleep

from . import __version__
from .archive import sync
from .embedding import Granite, download_model, load_runtime
from .store import Store

ROOT = Path(__file__).resolve().parent.parent


def integer_between(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"Expected an integer between {low} and {high}")
        return number
    return parse


def archive_args(parser):
    parser.add_argument(
        "--source", type=Path, action="append", default=[],
        help="JSONL file or directory of rollout-*.jsonl files, repeatable",
    )
    parser.add_argument("--codex-home", type=Path, help="Look in this directory's sessions and archived_sessions")


def archive_sources(args):
    sources = list(args.source)
    if args.codex_home:
        home = args.codex_home.expanduser().resolve()
        sources.extend(path for name in ("sessions", "archived_sessions") if (path := home / name).is_dir())
    if not sources:
        raise ValueError("Supply --source or a --codex-home containing session directories")
    return [path.expanduser().resolve() for path in sources]


def emit(value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write((serialized + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    else:
        print(serialized, flush=True)


def watch(store, engine, sources, interval):
    while True:
        try:
            result = Counter()
            for source in sources:
                if source.exists():
                    result.update(sync(store, [source]))
            indexed = sum(store.index_batch(engine) for _ in range(4))
            if indexed or any(result[key] for key in ("messages_added", "invalid_lines", "read_errors", "unsupported_files")):
                emit({**result, "embeddings_added": indexed})
        except Exception as error:
            logging.warning("Archive/index retry in %ss: %s", interval, error)
        sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Local Codex conversation archive and search")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "memory.sqlite3")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "granite")
    parser.add_argument("--threads", type=integer_between(1, 8), default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model", help="Download and verify public Granite model files")
    archive_args(commands.add_parser("sync", help="Import new complete JSONL lines once"))
    watcher = commands.add_parser("watch", help="Continuously archive new chat lines and update embeddings")
    archive_args(watcher)
    watcher.add_argument("--interval", type=integer_between(2, 3600), default=30)
    index = commands.add_parser("index", help="Generate missing vectors in small batches")
    index.add_argument("--max-batches", type=integer_between(1, 100000))
    search = commands.add_parser("search", help="Search without a MCP client")
    search.add_argument("query")
    search.add_argument("--limit", type=integer_between(1, 20), default=8)
    search.add_argument("--kind", choices=["note", "message"], default="")
    search.add_argument("--project", default="")
    search.add_argument("--mode", choices=["hybrid", "keyword", "vector"], default="hybrid")
    read = commands.add_parser("read", help="Read archived text")
    read.add_argument("id")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--max-chars", type=integer_between(1, 6000), default=3500)
    commands.add_parser("status", help="Show archive and index counts")
    commands.add_parser("serve", help="Run the local MCP server over stdio")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    model = args.model_dir.expanduser().resolve()

    try:
        if args.command == "download-model":
            download_model(model)
            emit({"verified_model": str(model)})
            return
        store = Store(args.db.expanduser().resolve())
        engine = Granite(model, args.threads)
        if args.command == "sync":
            emit(sync(store, archive_sources(args)))
        elif args.command == "watch":
            watch(store, engine, archive_sources(args), args.interval)
        elif args.command == "index":
            indexed, batches = 0, 0
            while args.max_batches is None or batches < args.max_batches:
                count = store.index_batch(engine)
                if count == 0:
                    break
                indexed += count
                batches += 1
            emit({"embeddings_added": indexed, **store.status()})
        elif args.command == "search":
            emit(store.search(
                args.query, limit=args.limit, kind=args.kind, project=args.project, mode=args.mode, engine=engine
            ))
        elif args.command == "read":
            emit(store.read(args.id, args.offset, args.max_chars))
        elif args.command == "status":
            emit(store.status())
        elif args.command == "serve":
            # Load native inference libraries before MCP starts its Windows stdio threads.
            load_runtime()
            from .server import make_server

            make_server(store, engine).run(transport="stdio")
    except KeyboardInterrupt:
        return
    except (ValueError, RuntimeError, OSError, sqlite3.Error, ImportError) as error:
        parser.exit(1, f"codex-memory: {error}\n")
