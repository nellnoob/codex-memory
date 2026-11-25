from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

from codex_memory.archive import sync
from codex_memory.embedding import Granite
from codex_memory.store import Store

ROOT = Path(__file__).resolve().parent.parent
THREAD = "475dfd2c-50c8-4bd3-b9a3-e7fd18eb77c8"
PROJECT = "C:/scratch/demo"


def line(type_, payload):
    return (json.dumps({"type": type_, "payload": payload}, ensure_ascii=False) + "\n").encode("utf-8")


def message(body, role="user"):
    return line("response_item", {
        "type": "message", "role": role, "content": [{"type": "input_text", "text": body}],
    })


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root / "memory.sqlite3")
        self.file = self.root / "rollout-test.jsonl"
        self.file.write_bytes(line("session_meta", {"id": THREAD, "cwd": PROJECT}))

    def test_archive_resume_moved_and_filter(self):
        content = message("用 R 图验证，脚本名是 model_fit.R。")
        pending = message("第二轮保留原话。", "assistant")
        with self.file.open("ab") as stream:
            stream.write(content)
            stream.write(b"broken-json\n")
            stream.write(message("<environment_context>ignore me</environment_context>"))
            stream.write(message("non-user instructions", "developer"))
            stream.write(line("response_item", {"type": "function_call_output", "output": "ignore me"}))
            stream.write(pending[:-3])

        first = sync(self.store, [self.root])
        self.assertEqual(1, first["messages_added"])
        self.assertEqual(1, first["invalid_lines"])
        self.assertEqual(0, sync(self.store, [self.root])["messages_added"])
        with self.file.open("ab") as stream:
            stream.write(pending[-3:])
        self.assertEqual(1, sync(self.store, [self.root])["messages_added"])

        archive = self.root / "archive"
        archive.mkdir()
        shutil.move(self.file, archive / self.file.name)
        self.assertEqual(0, sync(self.store, [self.root])["messages_added"])
        self.assertEqual(2, self.store.status()["messages"])

        hits = self.store.search("model_fit.R", project=PROJECT, mode="keyword")["hits"]
        self.assertEqual(1, len(hits))
        self.assertEqual(THREAD, hits[0]["thread_id"])
        self.assertIn("model_fit.R", self.store.read(hits[0]["id"])["text"])
        self.assertEqual(1, len(self.store.search("脚本", mode="keyword")["hits"]))
        self.assertEqual([], self.store.search("model_fit.R", project="different", mode="keyword")["hits"])
        page = self.store.thread(THREAD, limit=1)
        self.assertEqual("user", page["messages"][0]["role"])
        following = self.store.thread(THREAD, from_line=page["next_line"])
        self.assertEqual("assistant", following["messages"][0]["role"])

    def test_note_source_retry_and_paged_text(self):
        body = ("记录数据质量，检查缺失值。\n" * 80) + "最终使用 R 绘图。"
        with self.file.open("ab") as stream:
            stream.write(message(body))
        sync(self.store, [self.file])
        source = self.store.thread(THREAD)["messages"][0]["id"]
        first = self.store.save("绘图约定", "同时保存 R 脚本和工作空间。", source_message_id=source)
        retry = self.store.save("绘图约定", "同时保存 R 脚本和工作空间。", source_message_id=source)
        self.assertEqual(first["id"], retry["id"])
        self.assertFalse(retry["created"])
        self.assertEqual(THREAD, first["thread_id"])
        self.assertEqual(source, self.store.read(first["id"])["source_message_id"])
        self.assertEqual(1, len(self.store.search("工作空间", project=PROJECT, kind="note", mode="keyword")["hits"]))
        parts, offset = [], 0
        while offset is not None:
            page = self.store.read(source, offset=offset, max_chars=123)
            parts.append(page["text"])
            offset = page["next_offset"]
        self.assertEqual(body, "".join(parts))


@unittest.skipUnless(os.environ.get("CODEX_MEMORY_TEST_MODEL"), "Set CODEX_MEMORY_TEST_MODEL to test real model and MCP")
class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_vector_and_mcp_round_trip(self):
        import numpy as np
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        model = Path(os.environ["CODEX_MEMORY_TEST_MODEL"]).resolve()
        engine = Granite(model)
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            store = Store(folder / "memory.sqlite3")
            file = folder / "rollout-test.jsonl"
            file.write_bytes(b"".join([
                line("session_meta", {"id": THREAD, "cwd": PROJECT}),
                message("The decision was to use semantic similarity for retrieving older chat messages."),
                message("Bananas brown after being left out on the counter."),
                message("R 脚本运行失败，model_fit.R 的输入缺了 treatment 列。"),
            ]))
            self.assertEqual(3, sync(store, [file])["messages_added"])
            self.assertEqual(3, store.index_batch(engine))
            self.assertEqual(0, store.index_batch(engine))
            vectors = engine.encode(["本地语义搜索", "Look for semantically similar passages."])
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1], atol=1e-6)
            match = store.search("怎样按意思查找以前的聊天记录？", mode="vector", engine=engine)
            self.assertIn("semantic similarity", match["hits"][0]["snippet"])

            parameters = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "run.py"), "--db", str(store.path), "--model-dir", str(model), "serve"],
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=20)) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    self.assertEqual(
                        {"memory_search", "memory_read", "memory_thread", "memory_write"},
                        {tool.name for tool in tools.tools},
                    )

                    async def call(name, args):
                        result = await client.call_tool(name, args)
                        self.assertFalse(result.isError, result.content)
                        if result.structuredContent is not None:
                            return result.structuredContent
                        return json.loads(result.content[0].text)

                    hits = await call("memory_search", {"query": "model_fit.R", "mode": "hybrid"})
                    source = hits["hits"][0]["id"]
                    self.assertIn("model_fit.R", (await call("memory_read", {"id": source}))["text"])
                    saved = await call("memory_write", {
                        "title": "Source check", "body": "Always check treatment.",
                        "source_message_id": source,
                    })
                    self.assertEqual(source, (await call("memory_read", {"id": saved["id"]}))["source_message_id"])
                    self.assertEqual(3, len((await call("memory_thread", {"thread_id": THREAD}))["messages"]))


if __name__ == "__main__":
    unittest.main()
