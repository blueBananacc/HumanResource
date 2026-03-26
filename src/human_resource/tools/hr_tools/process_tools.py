"""HR 流程工具。

MVP 使用模拟数据，后续接入实际流程引擎。
"""

from __future__ import annotations

from langchain_core.tools import tool

# 模拟 HR 流程数据
_MOCK_PROCESSES = {
    "leave_request": {
        "name": "请假申请",
        "steps": [
            "1. 登录 HR 系统",
            "2. 进入请假申请页面",
            "3. 选择假期类型和日期",
            "4. 填写请假原因",
            "5. 提交申请",
            "6. 等待直属上级审批",
        ],
    },
    "onboarding": {
        "name": "入职流程",
        "steps": [
            "1. 签署劳动合同",
            "2. 提交个人资料",
            "3. 领取工牌和设备",
            "4. 参加入职培训",
            "5. 与上级进行入职面谈",
        ],
    },
    "resignation": {
        "name": "离职流程",
        "steps": [
            "1. 提交离职申请",
            "2. 上级审批",
            "3. HR 确认",
            "4. 工作交接",
            "5. 资产归还",
            "6. 办理离职手续",
        ],
    },
}


@tool
def list_hr_processes() -> list[dict]:
    """列出所有可用的 HR 流程。

    Returns:
        流程列表，包含流程 ID 和名称。
    """
    return [
        {"id": pid, "name": info["name"]}
        for pid, info in _MOCK_PROCESSES.items()
    ]


@tool
def get_process_steps(process_id: str) -> dict:
    """查询具体 HR 流程的步骤。

    Args:
        process_id: 流程 ID（如 leave_request, onboarding, resignation）。

    Returns:
        流程详情，包含名称和步骤列表。
    """
    process = _MOCK_PROCESSES.get(process_id)
    if process is None:
        return {"error": f"未找到流程: {process_id}，可用流程: {list(_MOCK_PROCESSES.keys())}"}
    return process

