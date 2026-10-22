"""测试一律打独立的 *_test 库。

套件里有清空 Tenant 全表的用例(如 test_sso 的 public 场景),直接在 dev
compose 里跑会 wipe 掉开发库。conftest 在任何 app 模块导入前把
DATABASE_URL 重定向到同名 ``_test`` 库;若已指向 ``_test`` 库则不动。
"""
from __future__ import annotations

import os

_url = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://freehire:freehire@postgres:5432/freehire"
)
if not _url.rsplit("/", 1)[-1].endswith("_test"):
    os.environ["DATABASE_URL"] = _url.rsplit("/", 1)[0] + "/freehire_test"

os.environ.setdefault("MACHINE_FINGERPRINT_OVERRIDE", "test-machine-id")
