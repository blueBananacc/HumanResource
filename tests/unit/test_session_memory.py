"""Session Memory 模块单元测试。

验证：
- Session/SessionMessage 数据模型（时间戳、intent_history）
- SessionMemory append / get_history / get_summary / get_turn_count
- Memory Trimmer: trim_and_summarize（阈值判断、增量摘要、裁剪）
- JSON 持久化与恢复（含新字段）
- Orchestrator 集成：load_context_node 使用存储摘要、post_process_node 元数据记录
- finalize_session episodic 记忆写入
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from human_resource.memory.session import Session, SessionMemory, SessionMessage


# ── SessionMessage 数据模型测试 ──────────────────────────────


class TestSessionMessage:
    def test_auto_timestamp(self):
        """消息应自动设置 timestamp。"""
        msg = SessionMessage(role="user", content="你好")
        assert msg.timestamp != ""
        assert "T" in msg.timestamp  # ISO 格式

    def test_explicit_timestamp(self):
        """传入 timestamp 时不应被覆盖。"""
        msg = SessionMessage(role="user", content="你好", timestamp="2026-01-01T00:00:00")
        assert msg.timestamp == "2026-01-01T00:00:00"

    def test_default_metadata(self):
        msg = SessionMessage(role="assistant", content="回复")
        assert msg.metadata == {}

    def test_custom_metadata(self):
        meta = {"intent_label": "policy_qa", "tools_used": ["lookup"]}
        msg = SessionMessage(role="user", content="查询", metadata=meta)
        assert msg.metadata["intent_label"] == "policy_qa"


# ── Session 数据模型测试 ─────────────────────────────────────


class TestSession:
    def test_auto_timestamps(self):
        """新建 Session 应自动设置 created_at 和 updated_at。"""
        s = Session(session_id="s1")
        assert s.created_at != ""
        assert s.updated_at == s.created_at

    def test_explicit_timestamps_preserved(self):
        s = Session(session_id="s1", created_at="2026-01-01", updated_at="2026-01-02")
        assert s.created_at == "2026-01-01"
        assert s.updated_at == "2026-01-02"

    def test_turn_count_empty(self):
        s = Session(session_id="s1")
        assert s.turn_count == 0

    def test_turn_count(self):
        s = Session(session_id="s1")
        s.messages = [
            SessionMessage(role="user", content="q1"),
            SessionMessage(role="assistant", content="a1"),
            SessionMessage(role="user", content="q2"),
            SessionMessage(role="assistant", content="a2"),
        ]
        assert s.turn_count == 2

    def test_turn_count_odd(self):
        """奇数消息（用户发了但未回复）=> 向下取整。"""
        s = Session(session_id="s1")
        s.messages = [
            SessionMessage(role="user", content="q1"),
            SessionMessage(role="assistant", content="a1"),
            SessionMessage(role="user", content="q2"),
        ]
        assert s.turn_count == 1

    def test_intent_history_default_empty(self):
        s = Session(session_id="s1")
        assert s.intent_history == []


# ── SessionMemory 核心操作测试 ───────────────────────────────


class TestSessionMemoryOps:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SessionMemory(persist_dir=self.tmpdir)

    def test_get_or_create_new(self):
        session = self.sm.get_or_create("new_session")
        assert session.session_id == "new_session"
        assert session.messages == []

    def test_append_basic(self):
        self.sm.append("s1", "user", "你好")
        self.sm.append("s1", "assistant", "你好！有什么帮助？")
        history = self.sm.get_history("s1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].content == "你好！有什么帮助？"

    def test_append_with_metadata(self):
        meta = {"intent_label": "policy_qa", "tools_used": []}
        self.sm.append("s1", "user", "年假政策", metadata=meta)
        history = self.sm.get_history("s1")
        assert history[0].metadata["intent_label"] == "policy_qa"

    def test_append_records_intent_history(self):
        self.sm.append("s1", "user", "q1", metadata={"intent_label": "chitchat"})
        self.sm.append("s1", "assistant", "a1")
        self.sm.append("s1", "user", "q2", metadata={"intent_label": "policy_qa"})
        session = self.sm.get_or_create("s1")
        assert session.intent_history == ["chitchat", "policy_qa"]

    def test_append_no_metadata_no_intent_history(self):
        self.sm.append("s1", "user", "q1")
        session = self.sm.get_or_create("s1")
        assert session.intent_history == []

    def test_append_updates_timestamp(self):
        session = self.sm.get_or_create("s1")
        original_updated = session.updated_at
        self.sm.append("s1", "user", "新消息")
        assert session.updated_at >= original_updated

    def test_get_summary_empty(self):
        assert self.sm.get_summary("s1") == ""

    def test_get_turn_count(self):
        self.sm.append("s1", "user", "q1")
        self.sm.append("s1", "assistant", "a1")
        self.sm.append("s1", "user", "q2")
        self.sm.append("s1", "assistant", "a2")
        assert self.sm.get_turn_count("s1") == 2

    def test_get_turn_count_empty(self):
        assert self.sm.get_turn_count("s1") == 0


# ── Memory Trimmer 测试 ──────────────────────────────────────


class TestTrimAndSummarize:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SessionMemory(persist_dir=self.tmpdir)

    def _add_turns(self, session_id: str, count: int):
        for i in range(count):
            self.sm.append(session_id, "user", f"问题{i+1}")
            self.sm.append(session_id, "assistant", f"回答{i+1}")

    def test_no_trim_below_threshold(self):
        """≤5 轮不裁剪。"""
        self._add_turns("s1", 5)
        stub = MagicMock(return_value="摘要")
        result = self.sm.trim_and_summarize("s1", stub)
        assert result is False
        stub.assert_not_called()
        assert len(self.sm.get_history("s1")) == 10

    def test_trim_above_threshold(self):
        """7 轮 → 裁剪旧消息，保留最近 5 轮。"""
        self._add_turns("s1", 7)
        stub = MagicMock(return_value="旧对话摘要")
        result = self.sm.trim_and_summarize("s1", stub)
        assert result is True
        # 保留 5 轮 = 10 条消息
        assert len(self.sm.get_history("s1")) == 4
        assert self.sm.get_summary("s1") == "旧对话摘要"
        stub.assert_called_once()

    def test_trim_exactly_threshold_plus_one(self):
        """刚好 6 轮 → 应裁剪。"""
        self._add_turns("s1", 6)
        stub = MagicMock(return_value="摘要")
        result = self.sm.trim_and_summarize("s1", stub)
        assert result is True
        assert len(self.sm.get_history("s1")) == 4  # 保留 5 轮

    def test_trim_preserves_recent_messages(self):
        """裁剪后保留的应该是最近的消息。"""
        self._add_turns("s1", 7)
        stub = MagicMock(return_value="摘要")
        self.sm.trim_and_summarize("s1", stub)
        history = self.sm.get_history("s1")
        # 最近 5 轮 → 问题3-问题7
        assert history[0].content == "问题6"
        assert history[-1].content == "回答7"

    def test_incremental_summary_merge(self):
        """增量摘要：已有摘要 + 新旧消息合并。"""
        self._add_turns("s1", 7)
        stub = MagicMock(return_value="第一次摘要")
        self.sm.trim_and_summarize("s1", stub)

        # 再添加 6 轮，触发第二次裁剪
        self._add_turns("s1", 6)
        stub2 = MagicMock(return_value="合并摘要")
        self.sm.trim_and_summarize("s1", stub2)

        call_text = stub2.call_args[0][0]
        assert "已有摘要" in call_text
        assert "第一次摘要" in call_text
        assert self.sm.get_summary("s1") == "合并摘要"

    def test_trim_updates_timestamp(self):
        self._add_turns("s1", 7)
        session = self.sm.get_or_create("s1")
        before = session.updated_at
        stub = MagicMock(return_value="摘要")
        self.sm.trim_and_summarize("s1", stub)
        assert session.updated_at >= before


# ── JSON 持久化测试 ──────────────────────────────────────────


class TestPersistence:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sm = SessionMemory(persist_dir=self.tmpdir)

    def test_save_and_load_basic(self):
        """保存后能正确恢复消息。"""
        self.sm.append("s1", "user", "你好")
        self.sm.append("s1", "assistant", "你好！")
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        history = sm2.get_history("s1")
        assert len(history) == 2
        assert history[0].content == "你好"

    def test_save_and_load_metadata(self):
        """持久化保留消息 metadata。"""
        self.sm.append("s1", "user", "查询", metadata={"intent_label": "policy_qa"})
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        history = sm2.get_history("s1")
        assert history[0].metadata["intent_label"] == "policy_qa"

    def test_save_and_load_timestamps(self):
        """持久化保留 created_at / updated_at。"""
        self.sm.append("s1", "user", "msg")
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        session = sm2.get_or_create("s1")
        assert session.created_at != ""
        assert session.updated_at != ""

    def test_save_and_load_summary(self):
        """持久化保留 summary。"""
        for i in range(7):
            self.sm.append("s1", "user", f"q{i}")
            self.sm.append("s1", "assistant", f"a{i}")
        stub = MagicMock(return_value="会话摘要")
        self.sm.trim_and_summarize("s1", stub)
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        assert sm2.get_summary("s1") == "会话摘要"

    def test_save_and_load_intent_history(self):
        """持久化保留 intent_history。"""
        self.sm.append("s1", "user", "q1", metadata={"intent_label": "chitchat"})
        self.sm.append("s1", "assistant", "a1")
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        session = sm2.get_or_create("s1")
        assert session.intent_history == ["chitchat"]

    def test_save_and_load_message_timestamp(self):
        """持久化保留消息级 timestamp。"""
        self.sm.append("s1", "user", "msg")
        self.sm.save("s1")

        sm2 = SessionMemory(persist_dir=self.tmpdir)
        history = sm2.get_history("s1")
        assert history[0].timestamp != ""

    def test_json_structure(self):
        """验证 JSON 文件结构完整性。"""
        self.sm.append("s1", "user", "q", metadata={"intent_label": "x"})
        self.sm.save("s1")

        fp = Path(self.tmpdir) / "s1.json"
        data = json.loads(fp.read_text(encoding="utf-8"))
        assert "session_id" in data
        assert "summary" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "intent_history" in data
        assert "messages" in data
        msg = data["messages"][0]
        assert "timestamp" in msg
        assert "metadata" in msg


# ── load_context_node 存储摘要优先测试 ───────────────────────


class TestLoadContextWithStoredSummary:
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_uses_stored_summary(self, mock_sm):
        """有存储摘要时，摘要 + 消息一起返回。"""
        from human_resource.agents.orchestrator import load_context_node

        sm = MagicMock()
        sm.get_history.return_value = [
            SessionMessage(role="user", content="最近的问题"),
            SessionMessage(role="assistant", content="最近的回答"),
        ]
        sm.get_summary.return_value = "这是历史摘要"
        mock_sm.return_value = sm

        state = {"session_id": "test"}
        result = load_context_node(state)

        snippets = result["session_context"]
        assert snippets[0] == "[历史摘要] 这是历史摘要"
        assert "user: 最近的问题" in snippets
        assert "assistant: 最近的回答" in snippets
        assert result["memory_context"] == []

    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_no_summary_returns_all_messages(self, mock_sm):
        """无存储摘要时直接返回所有消息（压缩由 post_process 负责）。"""
        from human_resource.agents.orchestrator import load_context_node

        sm = MagicMock()
        messages = []
        for i in range(7):
            messages.append(SessionMessage(role="user", content=f"q{i}"))
            messages.append(SessionMessage(role="assistant", content=f"a{i}"))
        sm.get_history.return_value = messages
        sm.get_summary.return_value = ""
        mock_sm.return_value = sm

        state = {"session_id": "test"}
        result = load_context_node(state)

        # 无摘要，直接获取全部 14 条消息
        assert len(result["session_context"]) == 14
        assert result["session_context"][0] == "user: q0"
        assert result["memory_context"] == []


# ── post_process_node 元数据与 trim 集成测试 ─────────────────


class TestPostProcessMetadata:
    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_records_intent_metadata(self, mock_sm, mock_should, mock_write, mock_comp):
        """post_process_node 应将 intent_label 传入 metadata。"""
        from human_resource.agents.orchestrator import post_process_node
        from human_resource.schemas.models import IntentItem, IntentLabel, IntentResult

        sm = MagicMock(spec=SessionMemory)
        mock_sm.return_value = sm
        mock_should.return_value = None
        mock_comp.return_value = MagicMock()

        intent = IntentResult(
            intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)]
        )
        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [HumanMessage(content="年假政策是什么")],
            "final_response": "年假政策如下...",
            "intent": intent,
        }
        post_process_node(state)

        # 验证 user 消息的 append 调用含 metadata
        user_calls = [
            c for c in sm.append.call_args_list if c[0][1] == "user"
        ]
        assert len(user_calls) == 1
        kwargs = user_calls[0].kwargs  # keyword args
        assert kwargs["metadata"]["intent_label"] == "policy_qa"

    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_calls_trim_and_summarize(self, mock_sm, mock_should, mock_write, mock_comp):
        """post_process_node 应调用 trim_and_summarize。"""
        from human_resource.agents.orchestrator import post_process_node

        sm = MagicMock(spec=SessionMemory)
        sm.trim_and_summarize.return_value = False
        mock_sm.return_value = sm
        mock_should.return_value = None

        compressor = MagicMock()
        mock_comp.return_value = compressor

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [HumanMessage(content="你好")],
            "final_response": "你好！",
        }
        post_process_node(state)

        sm.trim_and_summarize.assert_called_once_with(
            "s1", summarize_fn=compressor.summarize_text,
        )

    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_saves_twice_when_trimmed(self, mock_sm, mock_should, mock_write, mock_comp):
        """裁剪后应再次 save。"""
        from human_resource.agents.orchestrator import post_process_node

        sm = MagicMock(spec=SessionMemory)
        sm.trim_and_summarize.return_value = True  # 触发了裁剪
        mock_sm.return_value = sm
        mock_should.return_value = None
        mock_comp.return_value = MagicMock()

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [HumanMessage(content="你好")],
            "final_response": "你好！",
        }
        post_process_node(state)

        # save 应被调用 2 次：一次常规保存，一次裁剪后保存
        assert sm.save.call_count == 2
