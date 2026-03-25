"""Long-term Memory 模块单元测试。

使用 mock 替代 mem0 / LLM 调用，验证：
- load_context_node 压缩逻辑
- memory_retrieval_node 检索流程 + 用户画像加载
- memory_node 检索流程
- 三级触发机制 (_is_explicit_memory_command, _is_turn_interval_trigger, _assess_memory_worthiness, _should_write_memory)
- post_process_node 三级触发集成
- _extract_memorable_info LLM 提取
- _write_longterm_memory 分类型写入
- 图流程中 memory_retrieval 节点位置
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from human_resource.agents.orchestrator import (
    _assess_memory_worthiness,
    _extract_memorable_info,
    _extract_user_message,
    _is_explicit_memory_command,
    _is_turn_interval_trigger,
    _should_write_memory,
    _write_longterm_memory,
    load_context_node,
    memory_node,
    memory_retrieval_node,
    post_process_node,
)
from human_resource.memory.session import SessionMemory, SessionMessage


# ── _extract_user_message 测试 ───────────────────────────────


class TestExtractUserMessage:
    def test_extract_last_human_message(self):
        messages = [
            HumanMessage(content="第一条"),
            AIMessage(content="回复"),
            HumanMessage(content="第二条"),
        ]
        assert _extract_user_message(messages) == "第二条"

    def test_empty_messages(self):
        assert _extract_user_message([]) == ""

    def test_no_human_message(self):
        messages = [AIMessage(content="只有AI消息")]
        assert _extract_user_message(messages) == ""


# ── load_context_node 测试 ───────────────────────────────────


class TestLoadContextNode:
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_no_summary_returns_all_messages(self, mock_sm):
        """无摘要时直接返回所有消息。"""
        sm = MagicMock()
        sm.get_history.return_value = [
            SessionMessage(role="user", content="问题1"),
            SessionMessage(role="assistant", content="回答1"),
            SessionMessage(role="user", content="问题2"),
            SessionMessage(role="assistant", content="回答2"),
            SessionMessage(role="user", content="问题3"),
            SessionMessage(role="assistant", content="回答3"),
        ]
        sm.get_summary.return_value = ""
        mock_sm.return_value = sm

        state = {"session_id": "test"}
        result = load_context_node(state)

        assert len(result["session_context"]) == 6
        assert result["session_context"][0] == "user: 问题1"
        assert result["memory_context"] == []
        assert result["reflection_count"] == 0
        assert result["current_agent_index"] == 0

    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_with_stored_summary(self, mock_sm):
        """有存储摘要时，摘要 + 消息一起返回。"""
        sm = MagicMock()
        sm.get_history.return_value = [
            SessionMessage(role="user", content="最近问题"),
            SessionMessage(role="assistant", content="最近回答"),
        ]
        sm.get_summary.return_value = "早期对话摘要"
        mock_sm.return_value = sm

        state = {"session_id": "test"}
        result = load_context_node(state)

        assert result["session_context"][0] == "[历史摘要] 早期对话摘要"
        assert "user: 最近问题" in result["session_context"]
        assert len(result["session_context"]) == 3  # 1 摘要 + 2 消息
        assert result["memory_context"] == []

    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_empty_history(self, mock_sm):
        sm = MagicMock()
        sm.get_history.return_value = []
        mock_sm.return_value = sm

        state = {"session_id": "test"}
        result = load_context_node(state)
        assert result["session_context"] == []
        assert result["memory_context"] == []


# ── memory_retrieval_node 测试 ───────────────────────────────


class TestMemoryRetrievalNode:
    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_retrieves_memories(self, mock_ltm, mock_profile_cls):
        ltm = MagicMock()
        ltm.search.return_value = [
            {"memory": "用户是研发部的工程师"},
            {"memory": "用户之前问过年假政策"},
        ]
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [HumanMessage(content="年假还有几天")],
            "memory_context": [],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)

        assert len(result["memory_context"]) == 2
        assert "用户是研发部的工程师" in result["memory_context"]
        assert "用户之前问过年假政策" in result["memory_context"]
        ltm.search.assert_called_once_with("年假还有几天", user_id="u1", top_k=3)

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_no_results(self, mock_ltm, mock_profile_cls):
        ltm = MagicMock()
        ltm.search.return_value = []
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [HumanMessage(content="你好")],
            "memory_context": [],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)
        assert result["memory_context"] == []

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_empty_messages(self, mock_ltm, mock_profile_cls):
        ltm = MagicMock()
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [],
            "memory_context": ["existing"],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)
        assert result["memory_context"] == ["existing"]

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_mem0_failure_graceful(self, mock_ltm, mock_profile_cls):
        """mem0 异常时不崩溃，返回已有 context。"""
        ltm = MagicMock()
        ltm.search.side_effect = Exception("mem0 连接超时")
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [HumanMessage(content="查询")],
            "memory_context": ["existing"],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)
        assert result["memory_context"] == ["existing"]

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_empty_memory_text_filtered(self, mock_ltm, mock_profile_cls):
        """空 memory 文本被过滤掉。"""
        ltm = MagicMock()
        ltm.search.return_value = [
            {"memory": "有效记忆"},
            {"memory": ""},
            {"other_field": "no memory key"},
        ]
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [HumanMessage(content="查询")],
            "memory_context": [],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)
        assert len(result["memory_context"]) == 1
        assert "有效记忆" in result["memory_context"]

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_loads_user_profile(self, mock_ltm, mock_profile_cls):
        """验证加载用户画像到 state.user_profile。"""
        ltm = MagicMock()
        ltm.search.return_value = []
        mock_ltm.return_value = ltm

        profile_store = MagicMock()
        profile_store.get_profile.return_value = {
            "department": "研发部",
            "role": "工程师",
        }
        mock_profile_cls.return_value = profile_store

        state = {
            "messages": [HumanMessage(content="你好")],
            "memory_context": [],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)

        assert result["user_profile"] == {"department": "研发部", "role": "工程师"}

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_profile_failure_graceful(self, mock_ltm, mock_profile_cls):
        """画像加载异常不影响后续流程。"""
        ltm = MagicMock()
        ltm.search.return_value = [{"memory": "记忆内容"}]
        mock_ltm.return_value = ltm

        mock_profile_cls.side_effect = Exception("画像加载失败")

        state = {
            "messages": [HumanMessage(content="查询")],
            "memory_context": [],
            "user_id": "u1",
        }
        result = memory_retrieval_node(state)

        # 画像失败但记忆检索仍正常
        assert "记忆内容" in result["memory_context"]
        assert "user_profile" not in result


# ── memory_node 测试 ──────────────────────────────────────────


class TestMemoryNode:
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_retrieves_and_deduplicates(self, mock_ltm):
        """memory_node 检索后去重已有的长期记忆。"""
        ltm = MagicMock()
        ltm.search.return_value = [
            {"memory": "已存在的记忆"},
            {"memory": "新记忆"},
        ]
        mock_ltm.return_value = ltm

        state = {
            "messages": [HumanMessage(content="我之前说过什么")],
            "memory_context": ["已存在的记忆"],
            "user_id": "u1",
        }
        result = memory_node(state)

        # "已存在的记忆" 不应重复添加
        count = sum(1 for s in result["memory_context"] if "已存在的记忆" in s)
        assert count == 1
        assert "新记忆" in result["memory_context"]


# ── post_process_node 测试 ───────────────────────────────────


class TestPostProcessNodeFixed:
    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_extracts_last_user_msg(self, mock_sm, mock_should, mock_write, mock_comp):
        """修复后应取最后一条 HumanMessage，而非遍历取最后匹配的。"""
        sm = MagicMock(spec=SessionMemory)
        mock_sm.return_value = sm
        mock_should.return_value = None  # 不触发写入
        mock_comp.return_value = MagicMock()

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [
                HumanMessage(content="第一条"),
                AIMessage(content="回复第一条"),
                HumanMessage(content="第二条"),
            ],
            "final_response": "这是回复",
        }
        post_process_node(state)

        # 验证 user 消息含 metadata
        user_call = [c for c in sm.append.call_args_list if c[0][1] == "user"]
        assert len(user_call) == 1
        assert user_call[0][0][2] == "第二条"  # content
        # 验证 assistant 消息
        sm.append.assert_any_call("s1", "assistant", "这是回复")

    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_triggers_longterm_write_when_should(self, mock_sm, mock_should, mock_write, mock_comp):
        """触发条件满足时写入长期记忆。"""
        sm = MagicMock(spec=SessionMemory)
        mock_sm.return_value = sm
        mock_should.return_value = "explicit_command"
        mock_comp.return_value = MagicMock()

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [HumanMessage(content="记住这个：我是研发部的张三")],
            "final_response": "好的，我记住了您是研发部的张三。",
        }
        post_process_node(state)

        mock_write.assert_called_once_with(
            "u1", "s1", "记住这个：我是研发部的张三", "好的，我记住了您是研发部的张三。"
        )

    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_no_write_when_should_returns_none(self, mock_sm, mock_should, mock_write, mock_comp):
        """触发条件不满足时不写入。"""
        sm = MagicMock(spec=SessionMemory)
        mock_sm.return_value = sm
        mock_should.return_value = None
        mock_comp.return_value = MagicMock()

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [HumanMessage(content="你好")],
            "final_response": "你好！有什么可以帮你的？",
        }
        post_process_node(state)

        mock_write.assert_not_called()

    @patch("human_resource.agents.orchestrator._get_compressor")
    @patch("human_resource.agents.orchestrator._write_longterm_memory")
    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_no_write_when_empty(self, mock_sm, mock_should, mock_write, mock_comp):
        """空消息不触发写入。"""
        sm = MagicMock(spec=SessionMemory)
        mock_sm.return_value = sm
        mock_comp.return_value = MagicMock()

        state = {
            "session_id": "s1",
            "user_id": "u1",
            "messages": [],
            "final_response": "",
        }
        post_process_node(state)

        mock_should.assert_not_called()
        mock_write.assert_not_called()


# ── 三级触发机制测试 ─────────────────────────────────────────


class TestIsExplicitMemoryCommand:
    def test_chinese_keywords(self):
        assert _is_explicit_memory_command("记住这个：我在研发部") is True
        assert _is_explicit_memory_command("请记住我的部门") is True
        assert _is_explicit_memory_command("帮我记一下这个信息") is True

    def test_english_keywords(self):
        assert _is_explicit_memory_command("remember this: I'm in R&D") is True

    def test_no_keyword(self):
        assert _is_explicit_memory_command("年假还有几天？") is False
        assert _is_explicit_memory_command("你好") is False

    def test_case_insensitive(self):
        assert _is_explicit_memory_command("Remember This") is True


class TestIsTurnIntervalTrigger:
    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_no_trigger_at_non_interval(self, mock_sm):
        """第 2 轮不触发（间隔 = 3）。"""
        sm = MagicMock()
        sm.get_history.return_value = [
            SessionMessage(role="user", content="q1"),
            SessionMessage(role="assistant", content="a1"),
        ]
        mock_sm.return_value = sm
        assert _is_turn_interval_trigger("s1") is False

    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_triggers_at_6th_turn(self, mock_sm):
        """第 6 轮也应触发。"""
        sm = MagicMock()
        messages = []
        for i in range(9):
            messages.append(SessionMessage(role="user", content=f"q{i+1}"))
            messages.append(SessionMessage(role="assistant", content=f"a{i+1}"))
        sm.get_history.return_value = messages
        mock_sm.return_value = sm
        assert _is_turn_interval_trigger("s1") is True

    @patch("human_resource.agents.orchestrator._get_session_memory")
    def test_first_turn(self, mock_sm):
        """第 1 轮不触发。"""
        sm = MagicMock()
        sm.get_history.return_value = []
        mock_sm.return_value = sm
        assert _is_turn_interval_trigger("s1") is False


class TestAssessMemoryWorthiness:
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_true_for_yes(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "yes"
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        assert _assess_memory_worthiness("我是研发部的", "好的") is True

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_false_for_no(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "no"
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        assert _assess_memory_worthiness("你好", "你好！") is False

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_false_on_exception(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("LLM 故障")
        assert _assess_memory_worthiness("消息", "回复") is False


class TestShouldWriteMemory:
    @patch("human_resource.agents.orchestrator._assess_memory_worthiness")
    @patch("human_resource.agents.orchestrator._is_turn_interval_trigger")
    def test_explicit_command_highest_priority(self, mock_interval, mock_assess):
        """用户显式命令优先级最高。"""
        result = _should_write_memory("记住这个：重要信息", "好的", "s1")
        assert result == "explicit_command"
        # 后续策略不应被调用
        mock_interval.assert_not_called()
        mock_assess.assert_not_called()

    @patch("human_resource.agents.orchestrator._assess_memory_worthiness")
    @patch("human_resource.agents.orchestrator._is_turn_interval_trigger")
    def test_turn_interval_second_priority(self, mock_interval, mock_assess):
        mock_interval.return_value = True
        result = _should_write_memory("普通问题", "普通回答", "s1")
        assert result == "turn_interval"
        mock_assess.assert_not_called()

    @patch("human_resource.agents.orchestrator._assess_memory_worthiness")
    @patch("human_resource.agents.orchestrator._is_turn_interval_trigger")
    def test_llm_assessment_third_priority(self, mock_interval, mock_assess):
        mock_interval.return_value = False
        mock_assess.return_value = True
        result = _should_write_memory("我的上级是张三", "好的", "s1")
        assert result == "llm_assessment"

    @patch("human_resource.agents.orchestrator._assess_memory_worthiness")
    @patch("human_resource.agents.orchestrator._is_turn_interval_trigger")
    def test_no_trigger(self, mock_interval, mock_assess):
        mock_interval.return_value = False
        mock_assess.return_value = False
        result = _should_write_memory("你好", "你好！", "s1")
        assert result is None


# ── LLM 记忆提取测试 ─────────────────────────────────────────


class TestExtractMemorableInfo:
    @patch("human_resource.agents.orchestrator.get_llm")
    def test_extracts_profile_info(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = '[{"type": "profile", "content": "用户是研发部的工程师", "importance": 0.9}]'
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        result = _extract_memorable_info("我是研发部的工程师", "好的，已记录")
        assert len(result) == 1
        assert result[0]["type"] == "profile"
        assert result[0]["importance"] == 0.9

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_empty_for_chitchat(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "[]"
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        result = _extract_memorable_info("你好", "你好！")
        assert result == []

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_handles_markdown_code_block(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = '```json\n[{"type": "factual", "content": "上级是张三", "importance": 0.8}]\n```'
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        result = _extract_memorable_info("我的上级是张三", "好的")
        assert len(result) == 1
        assert result[0]["type"] == "factual"

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_empty_on_exception(self, mock_get_llm):
        mock_get_llm.side_effect = Exception("LLM 故障")
        assert _extract_memorable_info("消息", "回复") == []

    @patch("human_resource.agents.orchestrator.get_llm")
    def test_returns_empty_for_non_list(self, mock_get_llm):
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = '{"not": "a list"}'
        mock_llm.invoke.return_value = response
        mock_get_llm.return_value = mock_llm

        assert _extract_memorable_info("消息", "回复") == []


# ── _write_longterm_memory 测试（新版：按类型写入） ──────────


class TestWriteLongtermMemory:
    @patch("human_resource.agents.orchestrator._extract_memorable_info")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_writes_by_type(self, mock_ltm, mock_extract):
        """根据提取结果按类型写入 mem0。"""
        ltm = MagicMock()
        mock_ltm.return_value = ltm
        mock_extract.return_value = [
            {"type": "profile", "content": "用户是研发部的", "importance": 0.9},
            {"type": "factual", "content": "上级是张三", "importance": 0.8},
        ]

        _write_longterm_memory("u1", "s1", "我是研发部的，上级是张三", "好的")

        assert ltm.add.call_count == 2
        # 第一次调用 — profile
        call1 = ltm.add.call_args_list[0]
        assert call1[1]["memory_type"] == "profile"
        # 第二次调用 — factual
        call2 = ltm.add.call_args_list[1]
        assert call2[1]["memory_type"] == "factual"

    @patch("human_resource.agents.orchestrator._extract_memorable_info")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_filters_low_importance(self, mock_ltm, mock_extract):
        """importance ≤ 阈值的记忆不写入。"""
        ltm = MagicMock()
        mock_ltm.return_value = ltm
        mock_extract.return_value = [
            {"type": "episodic", "content": "用户问了你好", "importance": 0.2},
        ]

        _write_longterm_memory("u1", "s1", "你好", "你好！")

        ltm.add.assert_not_called()

    @patch("human_resource.agents.orchestrator._extract_memorable_info")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_empty_extraction_no_write(self, mock_ltm, mock_extract):
        mock_extract.return_value = []
        ltm = MagicMock()
        mock_ltm.return_value = ltm

        _write_longterm_memory("u1", "s1", "你好", "你好！")

        ltm.add.assert_not_called()

    @patch("human_resource.agents.orchestrator._extract_memorable_info")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_write_failure_does_not_raise(self, mock_ltm, mock_extract):
        """mem0 写入失败不应抛异常。"""
        ltm = MagicMock()
        ltm.add.side_effect = Exception("mem0 写入超时")
        mock_ltm.return_value = ltm
        mock_extract.return_value = [
            {"type": "factual", "content": "信息", "importance": 0.9},
        ]

        # 不应抛出异常
        _write_longterm_memory("u1", "s1", "消息", "回复")

    @patch("human_resource.agents.orchestrator._extract_memorable_info")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_skips_empty_content(self, mock_ltm, mock_extract):
        """content 为空时跳过。"""
        ltm = MagicMock()
        mock_ltm.return_value = ltm
        mock_extract.return_value = [
            {"type": "factual", "content": "", "importance": 0.9},
        ]

        _write_longterm_memory("u1", "s1", "消息", "回复")

        ltm.add.assert_not_called()


# ── Graph 流程测试 ──────────────────────────────────────────


class TestGraphWithMemoryRetrieval:
    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator.get_llm")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_graph_includes_memory_retrieval_node(self, mock_ltm, mock_get_llm, mock_profile_cls):
        """验证图中包含 memory_retrieval 节点。"""
        from human_resource.agents.graph import build_graph

        graph = build_graph()
        compiled = graph.compile()

        # 检查节点名称中包含 memory_retrieval
        node_names = list(compiled.get_graph().nodes.keys())
        assert "memory_retrieval" in node_names

    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator.get_llm")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_graph_flow_order(self, mock_ltm, mock_get_llm, mock_profile_cls):
        """验证 memory_retrieval 在 load_context 和 classify_intent 之间。"""
        from human_resource.agents.graph import build_graph

        graph = build_graph()
        compiled = graph.compile()
        graph_dict = compiled.get_graph()

        # 检查边: load_context → memory_retrieval → classify_intent
        edges = [(e.source, e.target) for e in graph_dict.edges]
        assert ("load_context", "memory_retrieval") in edges
        assert ("memory_retrieval", "classify_intent") in edges

    @patch("human_resource.agents.orchestrator._should_write_memory")
    @patch("human_resource.agents.orchestrator.UserProfileStore")
    @patch("human_resource.agents.orchestrator.get_llm")
    @patch("human_resource.agents.orchestrator._get_longterm_memory")
    def test_full_graph_with_mem0(self, mock_ltm, mock_get_llm, mock_profile_cls, mock_should):
        """完整图执行流程，mem0 被正常调用。"""
        # 设置触发条件
        mock_should.return_value = None  # 不触发写入

        # 设置 profile mock
        profile_store = MagicMock()
        profile_store.get_profile.return_value = {}
        mock_profile_cls.return_value = profile_store

        # 设置 mem0 mock
        ltm = MagicMock()
        ltm.search.return_value = [{"memory": "用户喜欢简洁回答"}]
        ltm.add.return_value = None
        mock_ltm.return_value = ltm

        # 设置 LLM mock
        mock_llm = MagicMock()

        intent_response = MagicMock()
        intent_response.content = '{"intents": [{"label": "chitchat", "confidence": 0.95, "entities": {}, "requires_tools": []}]}'

        reply_response = MagicMock()
        reply_response.content = "你好！有什么可以帮你的？"

        mock_llm.invoke.side_effect = [intent_response, reply_response]
        mock_get_llm.return_value = mock_llm

        from human_resource.agents.graph import build_graph

        graph = build_graph()
        compiled = graph.compile()

        result = compiled.invoke({
            "messages": [HumanMessage(content="你好")],
            "session_id": "test_mem",
            "user_id": "u1",
        })

        # mem0 search 应被调用（memory_retrieval_node）
        ltm.search.assert_called()
        # 触发条件被检查
        mock_should.assert_called()
        assert result.get("final_response") is not None
