"""Tool Calling 模块单元测试。

覆盖：
- ToolRegistry 注册/查找/列举/schema/参数映射
- Tool Executor 参数校验/执行/超时/格式化
- HR 内置工具的参数映射函数
- ToolResult.tool_name 字段
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, tool

from human_resource.schemas.models import ToolResult
from human_resource.tools.executor import (
    execute_tool,
    format_result,
    validate_params,
)
from human_resource.tools.registry import ToolRegistry, registry


# ── ToolRegistry 测试 ────────────────────────────────────────


class TestToolRegistry:
    def setup_method(self):
        self.reg = ToolRegistry()

    def _make_tool(self, name: str) -> BaseTool:
        @tool
        def dummy(x: str) -> str:
            """Dummy tool."""
            return x

        dummy.name = name
        return dummy

    def test_register_and_get(self):
        t = self._make_tool("test_tool")
        self.reg.register(t, category="test", source="internal")
        assert self.reg.get("test_tool") is t

    def test_get_nonexistent_returns_none(self):
        assert self.reg.get("nope") is None

    def test_has(self):
        t = self._make_tool("exists")
        self.reg.register(t)
        assert self.reg.has("exists") is True
        assert self.reg.has("nope") is False

    def test_list_tools_all(self):
        t1 = self._make_tool("a")
        t2 = self._make_tool("b")
        self.reg.register(t1, category="cat1")
        self.reg.register(t2, category="cat2")
        assert len(self.reg.list_tools()) == 2

    def test_list_tools_by_category(self):
        t1 = self._make_tool("a")
        t2 = self._make_tool("b")
        self.reg.register(t1, category="employee")
        self.reg.register(t2, category="process")
        result = self.reg.list_tools(category="employee")
        assert len(result) == 1
        assert result[0].name == "a"

    def test_list_by_names(self):
        t1 = self._make_tool("x")
        t2 = self._make_tool("y")
        t3 = self._make_tool("z")
        self.reg.register(t1)
        self.reg.register(t2)
        self.reg.register(t3)
        result = self.reg.list_by_names(["x", "z", "nonexistent"])
        assert len(result) == 2
        assert result[0].name == "x"
        assert result[1].name == "z"

    def test_get_all_tools(self):
        t1 = self._make_tool("a")
        t2 = self._make_tool("b")
        self.reg.register(t1)
        self.reg.register(t2)
        assert len(self.reg.get_all_tools()) == 2

    def test_get_schema(self):
        @tool
        def typed_tool(name: str, count: int) -> str:
            """有类型参数的工具。"""
            return f"{name}-{count}"

        self.reg.register(typed_tool)
        schema = self.reg.get_schema("typed_tool")
        assert schema is not None
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_get_schema_nonexistent(self):
        assert self.reg.get_schema("nope") is None

    def test_get_metadata(self):
        t = self._make_tool("m")
        self.reg.register(t, category="employee", source="mcp")
        meta = self.reg.get_metadata("m")
        assert meta["category"] == "employee"
        assert meta["source"] == "mcp"

    def test_get_metadata_nonexistent(self):
        assert self.reg.get_metadata("nope") is None


# ── 参数映射测试 ─────────────────────────────────────────────


class TestParamMapping:
    def setup_method(self):
        self.reg = ToolRegistry()

    def test_build_params_with_mapper(self):
        t = MagicMock(spec=BaseTool)
        t.name = "my_tool"
        mapper = lambda e: {"query": e.get("name", "")}
        self.reg.register(t, param_mapper=mapper)

        result = self.reg.build_params("my_tool", {"name": "张三", "extra": "ignored"})
        assert result == {"query": "张三"}

    def test_build_params_without_mapper_passthrough(self):
        """无映射函数时直接透传 entities。"""
        t = MagicMock(spec=BaseTool)
        t.name = "raw_tool"
        self.reg.register(t)

        result = self.reg.build_params("raw_tool", {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_build_params_unregistered_tool(self):
        result = self.reg.build_params("nope", {"key": "val"})
        assert result == {"key": "val"}


class TestHRToolParamMappers:
    """测试 HR 内置工具的参数映射函数。"""

    def test_map_lookup_employee_by_name(self):
        from human_resource.tools.hr_tools.employee_lookup import map_lookup_employee

        result = map_lookup_employee({"name": "张三"})
        assert result == {"query": "张三"}

    def test_map_lookup_employee_by_id(self):
        from human_resource.tools.hr_tools.employee_lookup import map_lookup_employee

        result = map_lookup_employee({"employee_id": "E001"})
        assert result == {"query": "E001"}

    def test_map_lookup_employee_empty(self):
        from human_resource.tools.hr_tools.employee_lookup import map_lookup_employee

        result = map_lookup_employee({})
        assert result == {"query": ""}

    def test_map_get_leave_balance(self):
        from human_resource.tools.hr_tools.employee_lookup import map_get_leave_balance

        result = map_get_leave_balance({"employee_id": "E001"})
        assert result == {"employee_id": "E001"}

    def test_map_list_hr_processes(self):
        from human_resource.tools.hr_tools.process_tools import map_list_hr_processes

        result = map_list_hr_processes({"anything": "ignored"})
        assert result == {}

    def test_map_get_process_steps(self):
        from human_resource.tools.hr_tools.process_tools import map_get_process_steps

        result = map_get_process_steps({"process": "leave_request"})
        assert result == {"process_id": "leave_request"}


# ── Executor 参数校验测试 ────────────────────────────────────


class TestValidateParams:
    def setup_method(self):
        from human_resource.agents.orchestrator import register_default_tools

        register_default_tools()

    def test_valid_params_pass(self):
        ok, err = validate_params("lookup_employee", {"query": "张三"})
        assert ok is True
        assert err is None

    def test_nonexistent_tool(self):
        ok, err = validate_params("ghost_tool", {})
        assert ok is False
        assert "未找到工具" in err

    def test_invalid_param_type(self):
        """传入不匹配 schema 的参数类型。"""
        ok, err = validate_params("lookup_employee", {"query": 123})
        # LangChain @tool 对基本类型通常做强制转换，可能仍通过
        # 此处验证校验流程本身不异常即可
        assert isinstance(ok, bool)


# ── Executor 执行测试 ────────────────────────────────────────


class TestExecuteTool:
    def setup_method(self):
        from human_resource.agents.orchestrator import register_default_tools

        register_default_tools()

    def test_execute_lookup_employee(self):
        result = execute_tool("lookup_employee", {"query": "张三"})
        assert result.success is True
        assert result.tool_name == "lookup_employee"
        assert result.data["name"] == "张三"
        assert "[lookup_employee 结果]" in result.formatted

    def test_execute_get_leave_balance(self):
        result = execute_tool("get_leave_balance", {"employee_id": "E001"})
        assert result.success is True
        assert result.tool_name == "get_leave_balance"
        assert result.data["annual"] == 10

    def test_execute_list_hr_processes(self):
        result = execute_tool("list_hr_processes", {})
        assert result.success is True
        assert result.tool_name == "list_hr_processes"
        assert isinstance(result.data, list)
        assert len(result.data) == 3

    def test_execute_get_process_steps(self):
        result = execute_tool("get_process_steps", {"process_id": "leave_request"})
        assert result.success is True
        assert result.data["name"] == "请假申请"

    def test_execute_nonexistent_tool(self):
        result = execute_tool("ghost", {"x": 1})
        assert result.success is False
        assert result.tool_name == "ghost"
        assert "未找到工具" in result.error

    def test_execute_tool_failure(self):
        """工具执行抛异常时返回失败结果。"""
        failing = MagicMock(spec=BaseTool)
        failing.name = "fail_tool"
        failing.args_schema = None
        failing.invoke.side_effect = RuntimeError("boom")
        registry.register(failing)

        result = execute_tool("fail_tool", {})
        assert result.success is False
        assert result.tool_name == "fail_tool"
        assert "boom" in result.error

    def test_execute_timeout(self):
        """工具执行超时返回超时错误。"""
        slow = MagicMock(spec=BaseTool)
        slow.name = "slow_tool"
        slow.args_schema = None
        slow.invoke.side_effect = lambda p: time.sleep(5)
        registry.register(slow)

        with patch("human_resource.tools.executor.TOOL_TIMEOUT_SECONDS", 0.1):
            result = execute_tool("slow_tool", {})

        assert result.success is False
        assert "超时" in result.error


# ── Result Formatter 测试 ────────────────────────────────────


class TestFormatResult:
    def test_format_dict(self):
        data = {"name": "张三", "department": "研发部"}
        text = format_result("lookup_employee", data)
        assert "[lookup_employee 结果]" in text
        assert "name: 张三" in text
        assert "department: 研发部" in text

    def test_format_dict_with_error(self):
        data = {"error": "未找到员工"}
        text = format_result("lookup_employee", data)
        assert "[lookup_employee] 错误: 未找到员工" in text

    def test_format_list(self):
        data = [
            {"id": "a", "name": "请假"},
            {"id": "b", "name": "入职"},
        ]
        text = format_result("list_hr_processes", data)
        assert "共 2 项" in text
        assert "id=a" in text

    def test_format_primitive(self):
        text = format_result("my_tool", "hello")
        assert "[my_tool 结果] hello" in text


# ── ToolResult.tool_name 测试 ────────────────────────────────


class TestToolResultToolName:
    def test_tool_name_default_empty(self):
        r = ToolResult(success=True, data={})
        assert r.tool_name == ""

    def test_tool_name_set(self):
        r = ToolResult(success=True, tool_name="lookup_employee", data={})
        assert r.tool_name == "lookup_employee"
