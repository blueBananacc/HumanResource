"""基础冒烟测试。"""

from human_resource.schemas.models import ToolResult


def test_import_config():
    """验证 config 模块可正常导入。"""
    from human_resource.config import MODEL_CONFIG, CHUNK_SIZE

    assert isinstance(MODEL_CONFIG, dict)
    assert CHUNK_SIZE > 0


def test_tool_result():
    """验证 ToolResult 数据结构。"""
    result = ToolResult(success=True, data={"name": "张三"}, formatted="张三")
    assert result.success
    assert result.data["name"] == "张三"
