"""P0 文档引擎:docx / xlsx / pptx 三套办公文档生成。

设计原则「AI 先成稿、人再微调」:服务层产出字节流,
``store.save_document`` 统一落对象存储 + 登记 generated_documents,
API 层负责权限/审计/下载。
"""
