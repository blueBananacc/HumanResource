"""基础冒烟测试。"""

from human_resource.schemas.models import IntentLabel, IntentItem, IntentResult, ToolResult
from human_resource.intent.router import resolve_route


def test_import_config():
    """验证 config 模块可正常导入。"""
    from human_resource.config import MODEL_CONFIG, CHUNK_SIZE

    assert isinstance(MODEL_CONFIG, dict)
    assert CHUNK_SIZE > 0


def test_intent_label_enum():
    """验证 IntentLabel 枚举值。"""
    assert IntentLabel.POLICY_QA.value == "policy_qa"
    assert IntentLabel.CHITCHAT.value == "chitchat"


def test_intent_result_primary():
    """验证 IntentResult 主意图选择逻辑。"""
    result = IntentResult(
        intents=[
            IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9),
            IntentItem(label=IntentLabel.CHITCHAT, confidence=0.3),
        ]
    )
    primary = result.primary_intent
    assert primary is not None
    assert primary.label == IntentLabel.POLICY_QA


def test_tool_result():
    """验证 ToolResult 数据结构。"""
    result = ToolResult(success=True, data={"name": "张三"}, formatted="张三")
    assert result.success
    assert result.data["name"] == "张三"


def test_router_resolve():
    """验证路由映射基本逻辑。"""
    intent = IntentResult(
        intents=[IntentItem(label=IntentLabel.EMPLOYEE_LOOKUP, confidence=0.95)]
    )
    agents = resolve_route(intent)
    assert agents == ["tool_agent"]
