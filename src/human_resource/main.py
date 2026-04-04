"""应用入口 + CLI 交互循环。"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import HumanMessage

from human_resource.config import DATA_DIR
from human_resource.memory.session import SessionMemory

logger = logging.getLogger(__name__)


def _ensure_data_dirs() -> None:
    """确保运行时数据目录存在。"""
    for sub in ("documents", "chroma_db", "sessions"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


def _on_exit(session_id: str, user_id: str) -> None:
    # 停止 MCP Client（关闭本地 MCP Server 进程）
    try:
        import asyncio

        from human_resource.mcp.client import stop_mcp_client

        asyncio.run(stop_mcp_client())
    except Exception:
        logger.exception("MCP Client 停止失败")


def _select_session() -> str:
    """显示会话选择菜单，返回选中或新建的 session_id。"""
    sm = SessionMemory()
    sessions = sm.list_sessions()

    print("\n" + "=" * 50)
    print("  会话管理")
    print("=" * 50)
    print("  [0] 新建会话")

    if sessions:
        for i, s in enumerate(sessions, 1):
            updated = s["updated_at"][:19].replace("T", " ") if s["updated_at"] else "未知"
            last_msg = s["last_message"] or "(无消息)"
            summary = s["msg_summary"] or "(暂无)"
            print(f"  [{i}] {s['session_id']}  |  {s['turn_count']}轮  |  {updated}")
            print(f"       最近: {last_msg}")
            print(f"       摘要: {summary}")
    else:
        print("  (无历史会话)")

    print("=" * 50)

    while True:
        try:
            choice = input("请选择 (输入编号): ").strip()
        except (EOFError, KeyboardInterrupt):
            return uuid.uuid4().hex[:8]

        if choice == "0" or choice == "":
            return uuid.uuid4().hex[:8]

        try:
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]["session_id"]
        except ValueError:
            pass

        print("  无效输入，请重新选择。")


def run() -> None:
    """CLI 交互循环入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    _ensure_data_dirs()

    # 编译 LangGraph
    from human_resource.agents.graph import compile_graph

    app = compile_graph()

    session_id = _select_session()
    user_id = "default_user"
    print("\n" + "=" * 50)
    print("  HR 智能助手 (输入 exit 退出)")
    print(f"  Session: {session_id}")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            _on_exit(session_id, user_id)
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            _on_exit(session_id, user_id)
            break

        # 通过 LangGraph 图处理用户输入
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=user_input)],
                "session_id": session_id,
                "user_id": user_id,
            })

            response = result.get("final_response", "")
            if response:
                print(f"\n助手: {response}")
            else:
                print("\n助手: 抱歉，我暂时无法回答这个问题。")

        except Exception:
            logger.exception("处理请求时发生错误")
            print("\n助手: 抱歉，系统处理出错，请稍后再试。")


if __name__ == "__main__":
    run()
