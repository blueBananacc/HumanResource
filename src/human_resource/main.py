"""应用入口 + CLI 交互循环。"""

from __future__ import annotations

import uuid

from human_resource.config import DATA_DIR


def _ensure_data_dirs() -> None:
    """确保运行时数据目录存在。"""
    for sub in ("documents", "chroma_db", "sessions"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


def run() -> None:
    """CLI 交互循环入口。"""
    _ensure_data_dirs()

    session_id = uuid.uuid4().hex[:8]
    print("=" * 50)
    print("  HR 智能助手 (输入 exit 退出)")
    print(f"  Session: {session_id}")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        # TODO: 接入 Orchestrator Agent 处理用户输入
        print(f"助手: [系统搭建中] 收到: {user_input}")


if __name__ == "__main__":
    run()
