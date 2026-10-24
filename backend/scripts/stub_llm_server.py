"""T6 链路验证用 OpenAI 兼容 LLM 桩。

不依赖外网 LLM 即可驱动 sidecar 完整工具链路:
assistant chat -> pi-agent -> /chat/completions(本桩)-> tool_calls -> 工具回打 backend API。

路由规则(按最后一条 user 消息关键词):
- 含 "手册|怎么|导入|help|文档"   -> search_product_help
- 含 "提醒|watcher"              -> create_watcher(写工具,走确认卡)
- 含 "筛选|filter"               -> save_filter(写工具)
- 其他                           -> search_talents

messages 里已有 role=tool 的结果 -> 回终态文本。
stream=true 时回 SSE chunk,否则回单个 JSON completion。
仅在 docker 网内监听 backend:8799,验证脚本用后应删除对应 provider。
"""
from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8799


def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _tool_for_text(user_msg: str) -> dict:
    if any(k in user_msg for k in ("手册", "怎么", "导入", "help", "文档")):
        return {"name": "search_product_help", "arguments": {"q": "上传简历"}}
    if any(k in user_msg for k in ("提醒", "watcher")):
        return {
            "name": "create_watcher",
            "arguments": {"kind": "candidate_stalled", "params": {"days": 7}},
        }
    if any(k in user_msg for k in ("筛选", "filter")):
        return {
            "name": "save_filter",
            "arguments": {
                "page": "talents",
                # saved_filters 对 (tenant,user,page,name) 唯一 — 带时间戳保证可重复验证
                "name": f"T6链路筛选-{int(time.time())}",
                "params": {"status": "interview"},
            },
        }
    return {"name": "search_talents", "arguments": {"q": "", "limit": 5}}


def _final_text(messages) -> str:
    """非工具调用的终态文本 — backend field-assist/summarize/doc-draft 也走这里。"""
    whole = "\n".join(_msg_text(m.get("content")) for m in messages)
    if "摘要" in whole or "summar" in whole.lower():
        return (
            "**AI 摘要(桩)**\n"
            "- 候选人当前处于活跃管道,最近一轮面试反馈正面。\n"
            "- 技能与岗位要求基本匹配,沟通记录显示响应及时。\n"
            "- 建议:推进下一轮技术面试,关注薪资期望对齐。"
        )
    if "起草" in whole or "模板" in whole or "文档" in whole:
        return (
            "兹证明以下事项(桩生成草稿):\n\n"
            "一、基本情况:该人员系我司正式员工/候选人,相关信息属实。\n"
            "二、说明事项:经核查,各项工作与记录符合公司制度要求。\n\n"
            "特此证明。\n\n(本稿由 AI 起草,请人工复核后使用)"
        )
    if any(k in user_msg_last(messages) for k in ("手册", "怎么", "导入")):
        return "已为您检索产品手册,结果见上方卡片。"
    return "以下是为您生成的内容(桩):结构清晰、语气专业,可直接编辑后使用。"


def user_msg_last(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _msg_text(m.get("content"))
    return ""


def _pick_tool(body) -> dict | None:
    """返回终态(无工具)时返回 None。

    sidecar pi-agent 会带 ``tools`` 字段发起请求;backend litellm 的
    field-assist/summarize/doc-draft 不带 — 以此区分两类调用。
    """
    if not body.get("tools"):
        return None
    messages = body.get("messages") or []
    # 只有"最后一条是 tool 结果"才是收尾轮 — 会话历史里残留的旧 tool
    # 消息不算数,否则同一对话第二轮起永远不再调工具。
    if messages and messages[-1].get("role") == "tool":
        return None
    return _tool_for_text(user_msg_last(messages))


def _build_response(model: str, body) -> dict:
    tool = _pick_tool(body)
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if tool:
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "arguments": json.dumps(tool["arguments"], ensure_ascii=False),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": _final_text(body.get("messages") or [])}
        finish = "stop"
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def _chunks_for(model: str, body) -> list[dict]:
    tool = _pick_tool(body)
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    if tool:
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        first = {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "arguments": json.dumps(tool["arguments"], ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        return [first, last]
    first = {
        **base,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": _final_text(body.get("messages") or [])}, "finish_reason": None}
        ],
    }
    last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return [first, last]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401 - 安静
        return

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": "stub-model", "object": "model"}]})
        else:
            self._json(200, {"ok": True})

    def do_POST(self):
        path = self.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": {"message": f"bad json: {e}"}})
            return
        model = body.get("model") or "stub-model"
        if body.get("stream"):
            payload = b""
            for ch in _chunks_for(model, body):
                payload += f"data: {json.dumps(ch, ensure_ascii=False)}\n\n".encode()
            payload += b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self._json(200, _build_response(model, body))


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"stub llm on :{PORT}", flush=True)
    srv.serve_forever()
