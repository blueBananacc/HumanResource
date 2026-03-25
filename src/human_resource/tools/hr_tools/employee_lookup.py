"""员工信息查询工具。

MVP 使用模拟数据，后续接入实际 HR 系统。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

# MVP 模拟员工数据
_MOCK_EMPLOYEES = {
    "E001": {
        "id": "E001",
        "name": "张三",
        "department": "研发部",
        "position": "高级工程师",
        "manager": "李四",
        "hire_date": "2022-03-15",
    },
    "E002": {
        "id": "E002",
        "name": "李四",
        "department": "研发部",
        "position": "研发总监",
        "manager": "王五",
        "hire_date": "2020-01-10",
    },
    "E003": {
        "id": "E003",
        "name": "王五",
        "department": "人力资源部",
        "position": "HR 总监",
        "manager": "",
        "hire_date": "2019-06-01",
    },
}

# 模拟假期余额
_MOCK_LEAVE_BALANCE = {
    "E001": {"annual": 10, "sick": 5, "personal": 3},
    "E002": {"annual": 15, "sick": 5, "personal": 3},
    "E003": {"annual": 15, "sick": 5, "personal": 3},
}


@tool
def lookup_employee(query: str) -> dict:
    """根据姓名或工号查询员工信息。

    Args:
        query: 员工姓名或工号。

    Returns:
        员工信息字典，未找到则返回错误信息。
    """
    # 按工号查找
    if query in _MOCK_EMPLOYEES:
        return _MOCK_EMPLOYEES[query]

    # 按姓名查找
    for emp in _MOCK_EMPLOYEES.values():
        if emp["name"] == query:
            return emp

    return {"error": f"未找到员工: {query}"}


@tool
def get_leave_balance(employee_id: str) -> dict:
    """查询员工假期余额。

    Args:
        employee_id: 员工工号。

    Returns:
        假期余额字典。
    """
    balance = _MOCK_LEAVE_BALANCE.get(employee_id)
    if balance is None:
        return {"error": f"未找到员工假期信息: {employee_id}"}
    return {"employee_id": employee_id, **balance}


# ── 参数映射函数 ─────────────────────────────────────────────


def map_lookup_employee(entities: dict[str, Any]) -> dict[str, Any]:
    """将意图 entities 映射为 lookup_employee 工具参数。"""
    query = entities.get("name", entities.get("employee_id", ""))
    return {"query": str(query)}


def map_get_leave_balance(entities: dict[str, Any]) -> dict[str, Any]:
    """将意图 entities 映射为 get_leave_balance 工具参数。"""
    return {"employee_id": str(entities.get("employee_id", ""))}
