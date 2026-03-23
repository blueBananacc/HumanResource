"""应用入口 + CLI 交互循环。"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import HumanMessage

from human_resource.config import DATA_DIR

logger = logging.getLogger(__name__)


def _ensure_data_dirs() -> None:
    """确保运行时数据目录存在。"""
    for sub in ("documents", "chroma_db", "sessions"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


def _on_exit(session_id: str, user_id: str) -> None:
    """会话结束时执行清理和 episodic 记忆写入。"""
    try:
        from human_resource.agents.orchestrator import finalize_session

        finalize_session(session_id, user_id)
    except Exception:
        logger.exception("会话结束处理失败")


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

    session_id = uuid.uuid4().hex[:8]
    user_id = "default_user"
    print("=" * 50)
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
