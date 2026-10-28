"""核心 ORM 模型。

实体边界:
- Tenant     一个客户/公司
- OrgUnit    租户下的组织节点(公司 / 部门 / 项目组)
- User       租户下的用户(admin / hr / interviewer / hiring_manager / viewer)
- License    当前激活的许可证(单实例:同一时刻只有一条记录)
- Job        岗位 / JD
- Candidate  候选人(邮箱+手机哈希去重)
- Resume     简历(关联 candidate,允许多版本)
- EmailAccount         租户级 IMAP 邮箱配置
- Interview / InterviewTurn / QuestionSet / ResumeJobMatch / VoiceProvider 等
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, time
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy import false as sql_false
from sqlalchemy import true as sql_true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class OrgUnit(Base):
    """组织节点。

    采用简单邻接表(parent_id 指向上级)保存树结构:
    - ``kind``: company / department / project
    - ``parent_id`` 为空表示根节点
    """

    __tablename__ = "org_units"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="department")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True, index=True
    )
    email: Mapped[str] = mapped_column(String(256), nullable=False, index=True, unique=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="hr")
    # 团队管理 — admin 可禁用账户(disabled=不允许登录,数据保留)
    # active   — 正常使用
    # disabled — 禁用,登录时返 401 / token 已签发的请求返 403
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="active", index=True
    )
    # EPIC-03 SSO:``local`` 走密码登录,``sso`` 由 IdP 首次登录自动建号。
    # SSO 用户的 ``password_hash`` 仍是 hash 占位串 — 用 secrets.token_urlsafe 派生,
    # 任何明文都不可能命中,从而禁用本地密码登录路径。
    auth_source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="local"
    )
    # SSO 用户的 IdP subject(``sub`` claim),用于"邮箱被改" / 多 IdP 区分。
    # 本地账号此列为 NULL。
    external_subject: Mapped[str | None] = mapped_column(
        String(256), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # 审批代理人(E3):解析步骤审批人时把代理人一并纳入可审批集合
    delegate_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class License(Base):
    """已激活的 license。同一时刻只保留一条最新记录。

    payload 与 signature 都是 base64 字符串,直接持久化以便审计与跨实例共享。
    校验逻辑在 ``app.infra.license.verifier``。
    """

    __tablename__ = "licenses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lic_payload: Mapped[str] = mapped_column(Text, nullable=False)
    lic_signature: Mapped[str] = mapped_column(Text, nullable=False)
    activated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    activated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)


class Job(Base):
    """JD / 岗位。``skills`` 是字符串数组(JSON 列存)。

    EPIC-05 岗位治理字段:
    - ``competency_model`` JSON — 结构化能力模型(数组)
    - ``publish_status`` — 治理状态机: draft / pending_approval / published / closed
      与原 ``status`` (open/paused/closed) 正交:
        ``status`` 控制运行时可见度 / 匹配入队;
        ``publish_status`` 控制治理流程(只有 published 状态才让 HR 发起面试 / 自动匹配)。
      旧数据通过 migration server_default='published' 视为已发布,平滑过渡。
    - ``current_version`` — 当前版本号,内容每变更一次 +1
    - ``submitted_*`` / ``approved_*`` / ``approval_note`` — 审批留痕
    """

    __tablename__ = "jobs"

    LEVELS = ("entry", "intermediate", "advanced")
    STATUSES = ("open", "paused", "closed")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("org_units.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    level: Mapped[str] = mapped_column(String(32), nullable=False, default="intermediate")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    skills: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # ---- EPIC-05 岗位治理 ----
    competency_model: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    publish_status: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="published", index=True
    )
    current_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    submitted_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    approval_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---- T12 筛选画像(惰性派生,按 JD 指纹缓存) ----
    # {jd_hash, derived_at,
    #  questions: [{id, question, severity(1-10), why}],   # 硬性门槛题
    #  emphasis:  {skills_w, experience_w, ...}}            # 评分权重
    gate_profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # ---- T13 声明式评分规则(挂 NULL → 走 T12 固定五维路径) ----
    scoring_rule_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("scoring_rules.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Candidate(Base):
    """候选人。邮箱与手机哈希用于跨简历去重(隐私优先)。

    ``display_email`` / ``display_phone`` 为最近一次简历解析得到的明文,
    仅在 UI 展示时使用;按 PIPL 后期可加 mask 配置。

    EPIC-04 人才库运营字段:
    - ``tags`` JSON 字符串数组 — P0 直接挂在候选人上,简化模型;
      列表筛选用 PostgreSQL ``JSON 包含``即可
    - ``is_blacklisted`` / ``blacklist_reason`` / ``blacklisted_at`` / ``blacklisted_by``
      — 黑名单是高频筛选字段,冗余 bool 列方便索引
    - ``last_active_at`` — 任何相关动作发生时刷新,用于"最近活跃"排序
    - ``owner_user_id`` — 候选人主负责 HR;P1 可做"我的候选人"视图
    - ``status`` — active / archived,P1 可扩 hired / left
    """

    __tablename__ = "candidates"

    STATUSES = ("active", "archived")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("org_units.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    email_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    phone_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    display_email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    display_phone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ---- EPIC-04 人才库运营字段 ----
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="active", index=True
    )
    is_blacklisted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_false(), index=True
    )
    blacklist_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    blacklisted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    blacklisted_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    last_active_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )
    owner_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # ---- 两级查重(Moka 精准/疑似同款)----
    # ``dedup_flag`` = "suspect":与其他候选人存在疑似重复,进入待审队列;
    # 合并 / 手动确认非重复后清除。``dedup_peer_id`` 记录命中的对端候选,
    # 供队列页直接展示"与谁重复"。
    dedup_flag: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    dedup_peer_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # ---- 候选人锁定(Moka 锁定/释放同款)----
    # 锁定期间其他 HR 不可对该候选人做流转/编辑/合并等变更,防止内部抢人;
    # ``locked_until`` 过期视为自动释放(懒释放,无需后台任务)。
    locked_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Resume(Base):
    """简历(原始文件 + 解析产物)。

    ``storage_key`` 走 ``app.infra.storage.ObjectStore``,实际位置由后端配置决定。
    ``parsed_data`` 是结构化抽取结果(name/email/phone/experience/education/skills/...)。
    """

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String(256), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # `.docx` 等 Office 新格式 MIME 长度可达 71 字符
    # (application/vnd.openxmlformats-officedocument.wordprocessingml.document),
    # 加上历史/自定义 type 的余量,放宽到 255(RFC 6838 推荐上限)。
    file_mime: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")
    parsed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # 简历解析改为 Celery 异步任务后,UI 通过 parse_status 轮询当前进度。
    # pending  — 入库,任务待 worker 拉取
    # parsing  — worker 已开工(parse_started_at 同步写入)
    # done     — 解析完成,parsed_text/parsed_data 已写入
    # failed   — 解析失败,parse_error 含错误片段(任务不会自动重试,UI 提示用户重传)
    parse_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    parse_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    parse_finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    uploaded_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ResumeReadState(Base):
    """简历"已读"标记 —— 按用户维度(与通知已读语义一致)。

    复合主键 (resume_id, user_id):一个用户对一份简历最多一条已读记录。
    概览页「最近简历」用它渲染未读点/已读置灰,"标为已读/未读"写删本表行。
    """

    __tablename__ = "resume_read_states"

    resume_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resumes.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    read_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    # 「仅移除此条消息」:概览小部件里隐藏该简历,人才库档案保留;
    # NULL=未移除。移除不删数据,随时可在人才库找回
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ActivityDismissal(Base):
    """概览「通知与动态」活动流的按用户移除标记。

    动态流条目由多源(audit / 管道事件 / 预约 / 答卷)归并而来,没有统一
    主键,item_key 用 ``{源}:{行id}`` 复合标识,如 ``audit:xxx``。
    移除只是从当前用户的小部件视图里隐藏,不删审计/业务数据。
    """

    __tablename__ = "activity_dismissals"

    item_key: Mapped[str] = mapped_column(String(80), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    dismissed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Interview(Base):
    """AI 面试会话。状态机:in_progress → done / abandoned。

    ``summary`` JSON 字段在面试结束时填充,包含:
    - dimension_scores: {accuracy, completeness, clarity, latency} 0-100
    - recommendation: "推荐进入下一轮" / "保留" / "不推荐"
    - evidence_quotes: 关键证据片段列表

    ``mode`` 字段:
    - ``self_test`` — HR 自测,在带登录态的页面自己答题(原 ``text`` 模式,
      用于校验 LLM 配置 / 演示 / 题目质量评估,无业务意义)
    - ``remote``    — 候选人远程异步答题。HR 发起后生成 ``invite_token_hash``
      + ``expires_at``,候选人通过免登录链接 ``/i/{token}`` 进入答题页。
      首题 lazy 生成 — 链接没人点不消耗 LLM。

    ``modality`` 字段(M6 语音面试):
    - ``text``  — 候选人在网页打字回答(历史默认,完全兼容)
    - ``voice`` — 候选人浏览器录音回答,后端 STT 转写后写回 ``InterviewTurn.answer``,
                  现有 LLM 评分链不变。"伪实时"产品规则(不可暂停 / 不可重录 /
                  题目间不留思考时间)由前端 ``VoiceSession`` 强制实现。

    向后兼容:历史 ``mode='text'`` 视同 ``self_test``,迁移会就地改名。
    """

    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False
    )
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="self_test")
    # M6 语音面试:答题载体。"text" / "voice"。默认 "text" 让历史数据零迁移成本。
    modality: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="text", default="text"
    )
    # 单题录音上限秒数(仅 voice 模式生效)。前端倒计时 + 服务端 sanity check 都看它。
    single_turn_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="90", default=90
    )
    # 整场全程录音(可选,用于 HR 反作弊回放)。各 turn 的分段录音另存。
    full_audio_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="in_progress", index=True
    )
    level: Mapped[str] = mapped_column(String(32), nullable=False, default="intermediate")
    # 题数 / 题型 — 替代之前硬编码的 MAX_TURNS=5 与全题型混合。
    # remote 模式 HR 在发起表单选;self_test 默认 5 题、4 个题型全选。
    question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    kinds: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=lambda: ["tech", "project", "scenario", "soft"]
    )
    # remote 模式独有 — token + expires_at + 候选人通知邮箱。
    # token 用 sha256(token_urlsafe(32)) 存,明文只发给 HR 一次。
    invite_token_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    notify_email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    delivery: Mapped[str] = mapped_column(String(16), nullable=False, default="link")
    # 候选人首次打开链接时间 — 区分"已邀请未答" / "已开始"
    candidate_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # _finish 后给 HR 发完成通知是否已发出(SMTP 模块上线后启用)
    hr_notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class InterviewTurn(Base):
    """单轮问答。``scores`` 是 4 个维度的得分,``evidence`` 是 LLM 给出的证据片段。

    M6 语音面试:
    - ``audio_storage_key`` 指向候选人本题录音文件(走 ObjectStore)。
    - ``transcript`` 是 STT 转写文本;同时也写回 ``answer`` 字段,让现有 LLM 评分
      链路一行不动。两者保留有差异是为了 HR 在报告页能看到 STT 原文 + 任何后续编辑。
    - ``transcript_status`` idle/pending/transcribing/done/failed 控制评分流的入口:
      voice 面试要等 transcript_status='done' 才能进入 score_status 链。
    - ``voice_signals`` 存语音特有指标(语速 / 静默率 / 填充词 / 声纹 embedding /
      多人检测),由 STT 后处理填充,供 HR 反作弊参考。
    - 文本面试这些字段全为 NULL,完全兼容历史。
    """

    __tablename__ = "interview_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    interview_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(String(32), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    answered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 评分链异步化后,UI 通过 score_status 轮询本轮处理进度。
    # idle    — 还没作答(占位,等用户提交)
    # pending — 用户已提交答案,任务待 worker 拉取
    # scoring — worker 已开工(score_started_at 同步写入)
    # done    — 评分 + 下一题(若有)/ 结束 都完成
    # failed  — 任一 LLM 调用异常,score_error 含错误片段
    score_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="idle", index=True
    )
    score_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    score_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    score_finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # ---- M6 语音面试增量字段(text 面试一律 NULL,兼容历史) ----
    # 候选人单题录音的 ObjectStore key,例:interviews/{interview_id}/turns/{idx}.webm
    audio_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    audio_uploaded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # STT 转写文本。落库后会同时写到 answer,但保留独立字段方便 HR 报告页看 "原始转写"。
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 转写状态机(语音面试专用,文本面试始终 'idle'):
    # idle          — 候选人还没录(text 模式恒在此态)
    # pending       — 录音已上传,等 STT worker
    # transcribing  — STT worker 正在转写
    # done          — 转写完成,answer 已填,可进入 score 链
    # failed        — STT 异常,transcript_error 含原因;HR 可重试
    transcript_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="idle", index=True
    )
    transcript_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 语音特有信号 JSON:{speech_rate_wpm, silence_ratio, filler_word_count,
    # voiceprint_embedding, background_voices_count}。供 HR 反作弊面板与 Interview
    # summary["voice_analysis"] 聚合用。
    voice_signals: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # ---- 留痕字段 ----
    # LLM 评分原始返回(完整 JSON),便于事后审计/复核 AI 决策过程。
    llm_raw_output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # HR 人工复核覆盖评分,格式同 scores:{accuracy,completeness,clarity,latency}
    hr_score_override: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    hr_score_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    hr_scored_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    hr_scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LLMProvider(Base):
    """租户级 LLM Provider 配置。

    - ``api_key_encrypted`` 是 :func:`app.infra.crypto.encrypt` 输出
    - 同租户最多一条 ``is_active=true``(由 partial unique index 强制)
    - ``model`` 直接写 LiteLLM 标识符(如 ``openai/gpt-4o-mini``、``deepseek/deepseek-chat``、
      ``azure/<deployment>``),后端不再做前缀映射,LiteLLM 原生路由
    """

    __tablename__ = "llm_providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    api_key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # embedding 可与 chat 走不同服务(如 chat=DeepSeek、embedding=HF/SiliconFlow);
    # 两者留空则回退共用 base_url/api_key
    embedding_base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class EmailAccount(Base):
    """租户级邮箱配置。

    - 密码 Fernet 加密存储
    - 后台轮询任务从这里读所有 ``is_enabled=true`` 的账户,拉新邮件附件入简历库
    - ``last_synced_at`` / ``last_status`` / ``last_error`` 用于 UI 展示同步状态
    """

    __tablename__ = "email_accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(256), nullable=False)
    imap_host: Mapped[str] = mapped_column(String(256), nullable=False)
    imap_port: Mapped[int] = mapped_column(Integer, nullable=False, default=993)
    imap_ssl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    password_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    folder: Mapped[str] = mapped_column(String(64), nullable=False, default="INBOX")
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SMTPAccount(Base):
    """租户级 SMTP 发件配置(M4 远程面试邀请 / 完成通知用)。

    设计:
    - 每个租户最多一条记录(API 用 upsert,UI 是单表单不是列表),够用且简洁
    - 密码 Fernet 加密(同 :class:`EmailAccount`)
    - ``use_tls=True`` 走 STARTTLS(默认 587 端口);False 走 implicit SSL(465)
    - ``from_email`` 通常 = ``username``,但部分企业邮箱允许独立发件人地址
    - 失败诊断信息在 ``last_status`` / ``last_error``,UI 上展示给管理员

    与 :class:`EmailAccount` (IMAP 收件) 完全独立 — 不复用同一表 / 同一连接,
    职责清晰。
    """

    __tablename__ = "smtp_accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,  # 一个租户最多一条
    )
    host: Mapped[str] = mapped_column(String(256), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False, default=587)
    use_tls: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    username: Mapped[str] = mapped_column(String(256), nullable=False)
    password_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    from_email: Mapped[str] = mapped_column(String(256), nullable=False)
    from_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SSOConfig(Base):
    """租户级 SSO 配置(EPIC-03 P0:仅支持 OIDC / OAuth2)。

    设计:
    - 一个租户最多一条(``tenant_id`` unique),UI 是单表单 — 大多数客户
      只接一个 IdP,P0 不做多 IdP 并存
    - ``client_secret`` 走 Fernet 加密,DB 只存密文
    - 字段对齐 ``07-epic-03-sso-integration.md`` 建议:
      issuer / client_id/secret / authorize_url / token_url / userinfo_url /
      scopes / redirect_uri / 自动建号开关 / 默认角色 / 默认组织 /
      claim 名称 + 映射规则
    - 自动建号 + 默认角色 + 默认组织三件套保证"首次登录后权限不悬空"
    - role_mapping_rules / org_mapping_rules 用 JSON 存
      ``{"claim_value": "role_or_org_id"}``,P0 简单等值映射,P1 再扩条件表达式

    本地登录兜底:``enabled=False`` 或本表无记录时,登录页只显示密码登录入口。
    管理员把 SSO 配错后仍可用本地账号登录,不会被锁死。
    """

    __tablename__ = "sso_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,
    )

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # P0 固定 oidc;留字段是为了 P1 区分 saml / cas / casdoor
    provider_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="oidc"
    )
    display_name: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="企业统一登录"
    )

    # ---- OIDC 端点 ----
    issuer_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    authorize_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    userinfo_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    client_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    client_secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    scopes: Mapped[str] = mapped_column(
        String(256), nullable=False, server_default="openid profile email"
    )
    redirect_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ---- 自动建号策略 ----
    auto_provision_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    default_role: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="hr"
    )
    default_org_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("org_units.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ---- claim 解析 ----
    email_claim: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="email"
    )
    name_claim: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="name"
    )
    role_claim: Mapped[str | None] = mapped_column(String(64), nullable=True)
    org_claim: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ---- 映射规则 ----
    # role_mapping_rules: {"claim_value": "system_role"}
    # org_mapping_rules:  {"claim_value": "org_unit_id"}
    role_mapping_rules: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    org_mapping_rules: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # ---- 状态 ----
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class CandidateGroup(Base):
    """候选人分组(人才池)。EPIC-04:手动加人 + 自动入库规则。

    设计:
    - 单租户内 ``name`` 唯一(管理员维护)
    - 与候选人多对多关系走 ``candidate_group_members`` 关联表
    - ``auto_rule`` JSONB — 自动入库规则(北森"自动入库"同款):
      ``{"mode": "all"|"any", "conditions": [{field, op, value}]}``
      字段词表见 ``app.services.group_rules.FIELDS``;评估确定性
      (可解释、可复现),命中即入组,只增不删 — 手动加入的成员
      不因规则变化被移除。
    """

    __tablename__ = "candidate_groups"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_candidate_groups_tenant_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    auto_rule: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class CandidateGroupMember(Base):
    """候选人 ↔ 分组 关联。``source``:manual 手动 / auto 规则命中。"""

    __tablename__ = "candidate_group_members"
    __table_args__ = (
        UniqueConstraint(
            "group_id", "candidate_id", name="uq_candidate_group_members"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    group_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("candidate_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    added_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="manual"
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class CandidateNote(Base):
    """候选人备注(人才库时间线主要来源之一)。

    ``author_email`` 冗余存储,避免用户被删除后旧备注显示空白。
    P0 不做编辑 / 删除 — append-only,简化模型 + 留痕一致。
    """

    __tablename__ = "candidate_notes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_id: Mapped[str] = mapped_column(String(36), nullable=False)
    author_email: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class QuestionSet(Base):
    """简历 → 面试题 题集。

    HR 在某份简历上点「生成面试题」即创建一条记录,worker 调 LLM 生成题目,
    每道题落为一条 :class:`QuestionLibraryItem`(``question_set_id`` 指回本题集)。
    状态机:pending → generating → done / failed。

    ``questions`` 列已废弃(0027 起题目迁入题库,该列置空,仅为降级保留)。旧结构::

        [{
          "question":      "题干",
          "answer_points": ["要点1", "要点2", ...],
          "dimensions":    ["技术深度", "项目复盘"],
          "difficulty":    "初级|中级|高级|专家",
          "follow_up":     "追问题(可选)"
        }, ...]

    设计:
    - resume_id 必填(题目从简历内容衍生)
    - job_id 可选 — 不绑岗位时,LLM 完全按简历技能 / 经验出题
    - kinds 多选,LLM 在 prompt 里拼成 "题目类型应覆盖: ..."
    - 不存 LLM 原始 raw response,只保留结构化结果(节省存储)
    """

    __tablename__ = "question_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    level: Mapped[str] = mapped_column(String(32), nullable=False, default="intermediate")
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    kinds: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    questions: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ResumeJobMatch(Base):
    """简历 ↔ 岗位 AI 匹配评分。

    每对 (resume, job) 一条记录(unique 复合键),worker 异步调 LLM 给匹配度
    评分 + 短评。HR 在简历库 / 岗位"匹配候选人"页直接看分数排序,而不是
    人眼挑简历。

    状态机:pending → matching → done / failed / gated(T12 门槛淘汰,
    gated 时 score=NULL — 拒绝与 0 分两种语义)。
    autoretry=0,UI 显式"重新评估"才会重跑(同 :class:`QuestionSet` 模式)。

    LLM 输出 schema(写到 ``score`` / ``strengths`` / ``gaps`` / ``comment``):
        {
          "score":     0-100,
          "strengths": ["匹配亮点 1", ...],   # ≤5 条
          "gaps":      ["关键短板 1", ...],   # ≤5 条
          "comment":   "<=120 字总结"
        }

    触发来源:
    - 简历解析完成 → ``evaluate_matches_for_resume.delay`` 对所有 active 岗位
    - 岗位创建/置 open → ``evaluate_matches_for_job.delay`` 对最近简历
    - HR 手工 → ``POST /api/matches/.../evaluate-all`` 或 ``regen``

    设计:
    - 完全照抄 :class:`QuestionSet` 状态机风格
    - 一对 (resume, job) 不重跑(已 done 的不重新算,节约 LLM 调用)
    - ``created_by`` 区分自动(NULL)/手动(HR user_id)触发,便于审计
    """

    __tablename__ = "resume_job_matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strengths: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    gaps: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ---- T12:门槛判定 + 证据先行评分 ----
    # gate: {passed, answers:[{id,question,verdict,evidence}],
    #        disqualifiers:[{question,severity,evidence}]}
    #   status='gated' 时 score 保持 NULL — 门槛拒绝和 0 分是两种语义。
    gate: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # criteria: [{key, name, weight, level(0-4), evidence}] — 逐维引用原文的锚定评分
    criteria: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    # red_flags: {"major": [维度名], "minor": [维度名]} — 高权重维度 level=0
    red_flags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # ---- T13:评估时冻结的评分规则快照(规则被改/删后仍可回溯评分依据)----
    # {rule_id, name, version, dimensions: [{id,name,weight,evaluator}]}
    rule_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # ---- T15:确定性关键词覆盖子分(零 LLM)----
    # {score 0-100, hit: [...], missing: [...], total} — JD 技能在简历的整词覆盖率
    coverage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # HR 在面试页"推荐面试"区点"忽略"后写入 — 该匹配不再出现在建议列表
    suggestion_dismissed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("resume_id", "job_id", name="uq_resume_job_match"),
    )


class ScoringRule(Base):
    """T13 声明式评分规则(租户级,可挂在多个岗位)。

    ``dimensions`` JSON 数组:
        [{id, name, weight, description?,
          evaluator: {method: keyword|duration|ai|boolean|range,
                      keywords?, matchMode?, minYears?, preferredYears?,
                      aiPrompt?, booleanField?, field?, minValue?, maxValue?}}]

    - 确定性方法(keyword/duration/boolean/range)纯函数评估,0 token;
    - ai 方法由一次批量 LLM 调用评所有 ai 维(evidence+level 证据先行);
    - ``version`` 每次内容修改 +1,评估时快照冻结进 match.rule_snapshot。
    """

    __tablename__ = "scoring_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimensions: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class TalentProfile(Base):
    """T16 人才画像 —— 可复用的「招什么样的人」模板(区别于岗位编制)。

    ``spec`` JSON:
        {skills: [], keywords: [], nice_to_have: [],
         min_years: 0, education: ""}

    - skills 喂 coverage(整词匹配),keywords 加分;
    - min_years/education 做扫描预筛门槛(gate-lite,可过滤);
    - ``source_job_id`` 记录画像来源岗位(从岗位沉淀画像时回填);
    - ``last_used_at``/``use_count`` 支撑「上次 X 月用过」的对话依据。
    """

    __tablename__ = "talent_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    scoring_rule_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("scoring_rules.id", ondelete="SET NULL"), nullable=True
    )
    source_job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="active")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SourcingRun(Base):
    """T16 扫描留痕 —— 每次人才库扫描的记录(审计 + 渠道效果统计原料)。"""

    __tablename__ = "sourcing_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    profile_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("talent_profiles.id", ondelete="SET NULL"), nullable=True
    )
    spec_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    scanned_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    prefiltered: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    llm_scored: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    returned: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class JobVersion(Base):
    """岗位版本快照(EPIC-05 T8)。

    每次岗位"内容字段"(title / level / description / skills / competency_model)
    实际发生变化时,落一条历史快照。``version_no`` 与 ``Job.current_version`` 对齐。

    ``change_kind`` 记录是 ``content_update`` / ``approve`` / ``reject`` /
    ``submit_approval`` / ``competency_generated`` / ``jd_optimized`` 等触发原因。
    审批 / 状态切换也会落一条版本(content 字段为空),帮助详情页统一时间线展示。
    """

    __tablename__ = "job_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    change_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    change_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 字段快照 — content_update 时填,纯状态事件可留 NULL
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    competency_model: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    publish_status: Mapped[str | None] = mapped_column(String(24), nullable=True)

    author_id: Mapped[str] = mapped_column(String(36), nullable=False)
    author_email: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class JobComment(Base):
    """岗位协作备注(EPIC-05 T10)。

    与 :class:`CandidateNote` 设计一致 — append-only,author_email 冗余存储。
    用人经理 / HR / 审批人都可留言,详情页时间线统一展示。
    """

    __tablename__ = "job_comments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_id: Mapped[str] = mapped_column(String(36), nullable=False)
    author_email: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


# =====================================================================
# HRIS 员工核心(领域模型见 docs/design/hrms-suite-design.md)
# =====================================================================


class Employee(Base):
    """员工主数据 — HRIS 核心实体。

    - ``reports_to_id`` 直属上级(自引用),既是组织架构图的边,
      也是"上级审批下属请假/报销"的行级数据范围依据
    - ``user_id`` 员工账号绑定(role='employee' 的 User,唯一);
      员工登录后 /api/me/* 全部按它行级过滤(ESS)
    - ``source_candidate_id`` 录用转化来源(offer accepted → 建档)
    - ``status``: active / leave / suspended / terminated
    - ``personal`` JSON 存低频字段(地址/紧急联系人/证件),避免表膨胀
    """

    __tablename__ = "employees"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_no: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True, index=True
    )
    designation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reports_to_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, unique=True
    )
    work_email: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hire_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    employment_type: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="full_time"
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="active", index=True
    )
    personal: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source_candidate_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True
    )
    leave_policy_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("leave_policies.id", ondelete="SET NULL"), nullable=True
    )
    position_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("positions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "employee_no", name="uq_employees_tenant_no"),
    )


class LeaveType(Base):
    """假期类型:年假 / 病假 / 事假 / 调休 …。``annual_days`` 为年度默认额度。

    ``is_comp`` 标记调休类型:加班申请审批通过后把额度结算到这类
    类型上(写 allocation + 台账),员工再按正常请假流程消耗。
    """

    __tablename__ = "leave_types"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    annual_days: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    is_paid: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_false())
    is_comp: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_false())
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_leave_types_tenant_name"),
    )


class LeaveAllocation(Base):
    """假期额度发放:某员工某类型某年度分得多少天。"""

    __tablename__ = "leave_allocations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    leave_type_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    days: Mapped[float] = mapped_column(Float, nullable=False)
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "employee_id", "leave_type_id", "year", name="uq_leave_alloc_emp_type_year"
        ),
    )


class LeaveApplication(Base):
    """请假申请。``days`` 支持 0.5(半天)。

    status: pending → approved / rejected / cancelled(员工撤回)
    """

    __tablename__ = "leave_applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    leave_type_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    days: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    approver_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class LeaveLedgerEntry(Base):
    """假期台账 — allocation(+) / application(-) 各一条,余额按汇总算。"""

    __tablename__ = "leave_ledger_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    leave_type_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    delta_days: Mapped[float] = mapped_column(Float, nullable=False)
    ref_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ref_id: Mapped[str] = mapped_column(String(36), nullable=False)
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class EmployeeCheckin(Base):
    """打卡记录(原始流水)。attendance 由它每日汇总。"""

    __tablename__ = "employee_checkins"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)  # in / out
    method: Mapped[str] = mapped_column(String(16), nullable=False, server_default="web")
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # 打卡来源 IP(ip_check 开关之外也留痕,供审计/争议取证)
    ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Attendance(Base):
    """按天的考勤汇总。status: present / absent / leave / remote / holiday。

    - checkin 产生的 first_in/last_out 自动刷新 work_hours
    - 请假单 approved 且覆盖当天 → status='leave'
    - HR 可手工修正(manual source)
    """

    __tablename__ = "attendances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="present", index=True
    )
    first_in: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_out: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    work_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default="checkin")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("employee_id", "day", name="uq_attendance_emp_day"),
    )


class AttendanceConfirmation(Base):
    """员工对月度考勤结果的在线确认(Moka"考勤结果在线确认"同款)。

    员工在「我的考勤」核对当月汇总后确认或提出异议,封账前给 HR 一个
    确认信号;异议由 HR 在考勤管理侧处理(resolve)。

    status: confirmed 员工已确认 / disputed 员工提了异议(带 note) /
            resolved   HR 已处理异议。
    """

    __tablename__ = "attendance_confirmations"

    STATUSES = ("confirmed", "disputed", "resolved")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    month: Mapped[str] = mapped_column(String(7), nullable=False)  # "YYYY-MM"
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    resolved_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    resolved_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("employee_id", "month", name="uq_attconfirm_emp_month"),
    )


class WorkCalendar(Base):
    """租户工作日历(每租户一条)。

    - ``workdays`` JSON 存 ISO  weekday 列表(1=周一…7=周日),默认 [1,2,3,4,5]
    - ``workday_hours`` 一个工作日的标准工时,加班申请折算调休天数用
    具体日期是工作日还是节假日由 ``Holiday`` 逐日覆盖
    (kind=holiday 放假 / kind=workday 调休补班)。
    """

    __tablename__ = "work_calendars"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False,
        index=True, unique=True,
    )
    workdays: Mapped[list[int]] = mapped_column(
        JSON, nullable=False, default=lambda: [1, 2, 3, 4, 5]
    )
    workday_hours: Mapped[float] = mapped_column(Float, nullable=False, server_default="8")
    # 期望作息(部署时区墙钟),迟到/早退判定的基准时刻
    work_start: Mapped[time] = mapped_column(
        Time, nullable=False, default=lambda: time(9, 0), server_default="09:00:00"
    )
    work_end: Mapped[time] = mapped_column(
        Time, nullable=False, default=lambda: time(18, 0), server_default="18:00:00"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AttendancePolicy(Base):
    """考勤策略(每租户一行)。

    - 宽限:``grace_in_minutes`` 分钟内到班不算迟到;``grace_out_minutes``
      分钟内离岗不算早退
    - ``ip_check_enabled`` + ``allowed_ips``(CIDR)开启打卡网络白名单;
      开关开但列表为空时按未启用处理,避免管理员把自己锁死
    - ``track_late_early`` 总开关关闭时 summarize 不产生 attendance_flags
    """

    __tablename__ = "attendance_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True, unique=True,
    )
    grace_in_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    grace_out_minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    track_late_early: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    ip_check_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_false()
    )
    allowed_ips: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    notify_on_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AttendanceFlag(Base):
    """迟到/早退异常台账(summarize_day 自动产出,HR 可免计)。

    ``minutes`` 是超出期望时刻的实际分钟数(明细口径);``detail`` 留
    判定现场(expected/actual 墙钟)。upsert 时保留已免计状态;考勤事实
    不再满足条件时整行删除,台账与事实保持一致。
    """

    __tablename__ = "attendance_flags"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attendance_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("attendances.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # late_in / early_out
    minutes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    excused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_false()
    )
    excused_by_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    excused_note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    excused_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("attendance_id", "kind", name="uq_attflag_att_kind"),
    )


class Holiday(Base):
    """节假日历条目:tenant × day 唯一。

    ``kind``: holiday(法定节假日,即使落在工作日也休息)/
    workday(调休补班,即使落在周末也上班)。
    """

    __tablename__ = "holidays"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, server_default="holiday")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "day", name="uq_holidays_tenant_day"),
    )


class AttendanceRequest(Base):
    """补卡申请(ESS 发起,上级审批)。

    status: pending → approved(自动补写 checkin + 重算当日考勤)/
    rejected / cancelled(员工撤回)。
    """

    __tablename__ = "attendance_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    kind: Mapped[str] = mapped_column(String(8), nullable=False)  # in / out
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    approver_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class OvertimeRequest(Base):
    """加班申请:批准后按 workday_hours 折算天数,结算到 ``leave_type_id``
    指定的调休类型(upsert 当年 allocation + 写台账 +delta)。"""

    __tablename__ = "overtime_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    work_date: Mapped[date] = mapped_column(Date, nullable=False)
    hours: Mapped[float] = mapped_column(Float, nullable=False)
    # 批准时按工作日历折算出的入账天数 — 写死在行上,日历后续变更不回算,
    # 撤销回冲 / 展示都以它为准(pending 时为 None,展示按当前日历预算)
    comp_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    leave_type_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    approver_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Shift(Base):
    """班次定义(H3 排班):如「早班 09:00-17:00 需 2 人」。

    - ``start_min``/``end_min`` 存分钟数(0-1440)而不是 time 类型:
      跨天班(22:00-06:00)直接 end < start,比较/展示都更直白
    - ``required`` 每个工作日默认需要的人数,生成器按它填空位
    """

    __tablename__ = "shifts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    start_min: Mapped[int] = mapped_column(Integer, nullable=False, server_default="540")
    end_min: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1020")
    required: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_shifts_tenant_name"),
    )


class ShiftAssignment(Base):
    """排班明细:某员工某天某班次。

    - ``source``: manual 手工 / auto 生成器 / cover 替班
    - 同一员工同一天只允许一条(跨班不叠加);unique 约束兜底
    """

    __tablename__ = "shift_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    shift_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("shifts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    day: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default="manual")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "employee_id", "day", name="uq_shiftassign_emp_day"),
    )


class LeavePolicy(Base):
    """假期政策:一组 (leave_type × annual_days) + 结转上限。

    ``items`` JSON: [{leave_type_id, days}];``carry_forward`` 为结转到
    下一年的额度上限,None 表示不结转。员工挂 ``leave_policy_id``
    后,每年 run 一次自动发放/对齐额度(台账记 policy_run delta)。
    """

    __tablename__ = "leave_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    carry_forward: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_leave_policies_tenant_name"),
    )


class LifecycleTemplate(Base):
    """入职/离职/调动清单模板。``tasks`` JSON: [{title}, …]"""

    __tablename__ = "lifecycle_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    tasks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class LifecycleEvent(Base):
    """员工生命周期事件(入职/调动/晋升/离职)。

    ``payload`` JSON 存各类型的扩展字段,如 transfer 的
    {from_org_unit_id, to_org_unit_id}、promotion 的 {from_designation, to_designation}。
    全部任务完成 → status 自动 completed。
    """

    __tablename__ = "lifecycle_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="in_progress", index=True
    )
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LifecycleTask(Base):
    __tablename__ = "lifecycle_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("lifecycle_events.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    done: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_false())
    done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sort: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class JobRequisition(Base):
    """招聘需求(编制)审批。

    hiring_manager 提交 → admin/hr 审批 → approved 后可一键转 Job。
    status: pending / approved / rejected / cancelled
    """

    __tablename__ = "job_requisitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    headcount: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    level: Mapped[str] = mapped_column(String(32), nullable=False, server_default="intermediate")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    requested_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approver_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class JobOffer(Base):
    """录用通知书 Offer。accepted → 创建 Employee + onboarding 事件。

    status: draft → sent → accepted / declined / withdrawn
    ``salary`` JSON: {base, currency, period} — 轻量记录不做核算。
    """

    __tablename__ = "job_offers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    position: Mapped[str | None] = mapped_column(String(128), nullable=True)
    salary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    letter: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft", index=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    responded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    employee_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Goal(Base):
    """员工目标(OKR-lite)。progress 0-100,status: active / done / dropped"""

    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="active", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AppraisalCycle(Base):
    """考核周期。status: draft / open / closed"""

    __tablename__ = "appraisal_cycles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Appraisal(Base):
    """绩效考核表。cycle×employee 唯一。

    流程: draft → 员工自评提交(self_submitted) → 上级/HR 评(reviewed)
    """

    __tablename__ = "appraisals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cycle_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("appraisal_cycles.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    self_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    self_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manager_review: Mapped[str | None] = mapped_column(Text, nullable=True)
    manager_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="draft", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("cycle_id", "employee_id", name="uq_appraisal_cycle_emp"),
    )


class ExpenseType(Base):
    __tablename__ = "expense_types"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_expense_types_tenant_name"),
    )


class ExpenseClaim(Base):
    """报销单。items 子表合计 total。审批流同请假单。"""

    __tablename__ = "expense_claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    total: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    approver_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ExpenseItem(Base):
    __tablename__ = "expense_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("expense_claims.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    type_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("expense_types.id", ondelete="SET NULL"), nullable=True
    )
    occurred_on: Mapped[date] = mapped_column(Date, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)


class SalarySlip(Base):
    """工资条(轻量登记制,不做自动核算)。employee × period 唯一。

    ``items`` JSON: [{label, kind: earning|deduction, amount}]
    gross/deductions/net 由 items 汇总或由 HR 直填。
    status: draft / issued(issued 后员工可见)
    """

    __tablename__ = "salary_slips"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    period: Mapped[str] = mapped_column(String(7), nullable=False)  # 'YYYY-MM'
    gross: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    deductions: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    net: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    currency: Mapped[str] = mapped_column(String(8), nullable=False, server_default="CNY")
    items: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft", index=True
    )
    issued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # 员工电子签收(薪人薪事同款):发放后员工在自助端确认;撤回重发需重新签收
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    confirmed_by_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("employee_id", "period", name="uq_salary_slip_emp_period"),
    )


class QuestionLibraryItem(Base):
    """租户级题库条目 — 题目是唯一实体,题集只是题目的集合。

    ``source``: "manual"(HR 手工录入) / "ai_generated"(题集出题或直接 AI 生成)。
    ``question_set_id``: 所属题集(为空=独立题目);题集删除时一并删除。
    ``position``: 在题集内的顺序;``dimensions``: 出题时 LLM 给的考察维度。
    ``use_count`` / ``avg_score`` 在面试引用本题后异步更新,供区分度分析用。
    """

    __tablename__ = "question_library_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_points: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="tech")
    difficulty: Mapped[str] = mapped_column(String(32), nullable=False, default="intermediate")
    category: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    skill: Mapped[str | None] = mapped_column(String(128), nullable=True)
    follow_up: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_score: Mapped[float | None] = mapped_column(nullable=True)
    generated_from_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    question_set_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("question_sets.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dimensions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AuditLog(Base):
    """操作审计日志。EPIC-02 审计合规中心的数据底座。

    ``actor_email`` 反规范化冗余存储,防止用户删除后日志无法显示操作人。
    ``detail`` 存操作附加信息,如评分覆盖前后值、邀请变更等。
    ``result`` 默认 ``success``;失败(异常 / 业务拒绝)与拒绝(403 / 越权)
    都需要单独记录,客户安全部门常用"failure / denied"做异常检测。
    ``ip`` / ``user_agent`` 来自请求头,用于事后排查异常访问来源;为空表示
    内部触发(如 worker)或请求头缺失。
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    actor_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor_email: Mapped[str] = mapped_column(String(256), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    result: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="success"
    )
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(256), nullable=True)
    via: Mapped[str | None] = mapped_column(String(32), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class VoiceProvider(Base):
    """租户级语音(STT/TTS)Provider 配置(M6 V5 — Settings UI)。

    设计参考 :class:`SMTPAccount`:租户内最多一条(``tenant_id`` unique),
    UI 是单表单。STT 和 TTS 共享一份记录是因为大多数客户两者用同一家厂商
    (例:阿里云 dashscope 同一个 API key 同时支持 ASR 和 TTS)。

    与 .env 配置的关系(优先级):
    1. DB 中存在该 tenant 行 + ``is_enabled=True`` → 用 DB 配置
    2. 否则 → 走 ``settings.stt_backend`` / ``settings.tts_backend``(.env 兜底)
    3. 都没配 → 业务层报错(STT/TTS Error)

    API key 走 :func:`app.infra.crypto.encrypt`,DB 只存密文。

    ``last_tested_at`` / ``last_status`` / ``last_error``:UI 上展示连通状态,
    由 ``POST /api/voice/test`` 写入。

    与之前的 ``LLMProvider`` 不同点:
    - LLMProvider 允许同租户多条 +``is_active`` 单选(因为 LLM 可能切换试用)
    - VoiceProvider 是 unique tenant_id 单条:STT/TTS 切换频率低,简化即可
    """

    __tablename__ = "voice_providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,  # 一个租户最多一条
    )

    # ---- STT ----
    stt_backend: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="mock"
    )
    stt_api_base: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stt_api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    stt_model: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default="whisper-1"
    )
    stt_language: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="zh"
    )

    # ---- TTS ----
    tts_backend: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="mock"
    )
    tts_api_base: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tts_api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    tts_model: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default="tts-1"
    )
    tts_voice: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="alloy"
    )
    tts_format: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="mp3"
    )

    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class JobApplication(Base):
    """候选人 × 岗位的申请管道卡(7 态收敛状态机)。

    stage 状态机(全部允许双向流转,只记历史不锁死):
        applied → screening → interview → offer → hired
                            ↘ rejected          withdrawn ↗

    联动钩子(见 ``app/services/applications.py``):
    - 发起 remote 面试 → applied/screening 自动推进 interview
    - offer send → offer;accepted → hired;declined → rejected;
      withdrawn → 回退 interview
    """

    __tablename__ = "job_applications"

    STAGES = ("applied", "screening", "interview", "offer", "hired", "rejected", "withdrawn")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="applied", index=True
    )
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="manual"
    )
    owner_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    rejected_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    stage_changed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "candidate_id", "job_id", name="uq_applications_tenant_cand_job"),
    )


class JobApplicationEvent(Base):
    """管道卡的阶段变迁历史(含系统联动动作)。"""

    __tablename__ = "job_application_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    application_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("job_applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    from_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_stage: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class PromptTemplate(Base):
    """租户级 AI 提示词模板(按用途 key 管理)。

    ``key`` 对应 ``app.services.prompting.PROMPT_REGISTRY`` 的注册项;
    没有自定义行时各服务用代码内置默认值。``{{var}}`` 占位符由调用方
    传入的变量替换;未知变量原样保留便于发现笔误。
    """

    __tablename__ = "prompt_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(48), nullable=False)
    # scope_module:按模块隔离的覆盖("" = 全局,非空 = 模块名)。
    # "" = 全局覆盖;非空(如 "jobs") = 只在该模块语境生效,render_prompt 两级回落。
    scope_module: Mapped[str] = mapped_column(
        String(48), nullable=False, server_default=""
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    updated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "key", "scope_module", name="uq_prompt_templates_tenant_key"
        ),
    )


class Skill(Base):
    """技能/胜任力字典(统一技能库 + 岗位/候选人侧评分)。

    category: tech / language / soft / cert / other
    """

    __tablename__ = "skills"

    CATEGORIES = ("tech", "language", "soft", "cert", "other")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="other", index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # ---- 技能图谱(Workday Skills Cloud 轻量版)----
    # parent_id:技能层级边(Python ⊃ Django 的"上位/下位"关系),
    #   匹配时子技能可向上兑现父级要求(衰减兑现,非完全等价)。
    # aliases:同义/别名归并("Golang"→"Go"),检索与匹配按 canonical 归一。
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="SET NULL"), nullable=True, index=True
    )
    aliases: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_skills_tenant_name"),
    )


class JobSkill(Base):
    """岗位技能要求:importance = must(必备)/ nice(加分)。"""

    __tablename__ = "job_skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    importance: Mapped[str] = mapped_column(
        String(8), nullable=False, server_default="must"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("job_id", "skill_id", name="uq_job_skills_pair"),
    )


class CandidateSkill(Base):
    """候选人技能:level 1-5(参考 CompetencyRatings 评分),source manual/resume。"""

    __tablename__ = "candidate_skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    level: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="manual"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("candidate_id", "skill_id", name="uq_candidate_skills_pair"),
    )


class Room(Base):
    """会议室。"""

    __tablename__ = "rooms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    location: Mapped[str | None] = mapped_column(String(128), nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, server_default="4")
    equipment: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Resource(Base):
    """可预约资源:设备 / 电话亭等。"""

    __tablename__ = "resources"

    TYPES = ("equipment", "phone_booth", "other")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="other"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Reservation(Base):
    """预约:占会议室或资源一段时间。

    status: active / cancelled。冲突校验在 API 层
    (重叠 + 同对象 + status=active → 409)。
    """

    __tablename__ = "reservations"

    KINDS = ("room", "resource", "interview")
    STATUSES = ("active", "cancelled")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    room_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("rooms.id", ondelete="CASCADE"), nullable=True, index=True
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("resources.id", ondelete="CASCADE"), nullable=True, index=True
    )
    interview_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("interviews.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    starts_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ends_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    interviewer_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="active", index=True
    )
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Survey(Base):
    """问卷/调查。

    status: draft(编辑中)/ published(可填写)/ closed(停止收集)。
    公开填写走 /api/public/surveys/{token}(仅 published,token 对应
    ``survey_links`` 里的一次性链接)。
    """

    __tablename__ = "surveys"

    STATUSES = ("draft", "published", "closed")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft", index=True
    )
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SurveyQuestion(Base):
    """问卷题目:type = short_text / long_text / single_choice / multi_choice / rating。"""

    __tablename__ = "survey_questions"

    TYPES = ("short_text", "long_text", "single_choice", "multi_choice", "rating")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    survey_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(String(256), nullable=False)
    options: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class SurveyLink(Base):
    """一次性公开填写链接(参考面试 invite_tokens 模式)。

    token 明文只在创建响应里出现一次,库中存 sha256;``used_at`` 置位即作废,
    ``expires_at`` 到期后不可再填。可绑定 ``candidate_id`` 追踪答卷归属。
    """

    __tablename__ = "survey_links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    survey_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    candidate_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SurveyResponse(Base):
    """答卷:respondent_key 为前端 localStorage 生成的随机串,与
    ``link_id`` 一次性链接双重防重复提交;``candidate_id`` 来自绑定链接。"""

    __tablename__ = "survey_responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    survey_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("surveys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    link_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("survey_links.id", ondelete="SET NULL"), nullable=True
    )
    candidate_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True
    )
    respondent_key: Mapped[str] = mapped_column(String(64), nullable=False)
    respondent_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    happiness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("survey_id", "respondent_key", name="uq_survey_resp_key"),
    )


class SurveyResponseAnswer(Base):
    """答卷明细:value 随题型为 str 或 list[str] 或 int。"""

    __tablename__ = "survey_response_answers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    response_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("survey_responses.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    question_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("survey_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)


class EmailTemplate(Base):
    """租户级邮件模板。

    ``key`` 对应 ``app.services.mail_templates.EMAIL_SPECS`` 注册项;
    subject / body_text / body_html 里写 ``{{var}}`` 占位,发送时按
    上下文变量替换。body_html 为空 → 只发纯文本。
    无自定义行 → 回落 ``app.workers.tasks.email`` 内置文案。
    """

    __tablename__ = "email_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(48), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    body_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    updated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "key", name="uq_email_templates_tenant_key"),
    )


class DocTemplate(Base):
    """文档模板 —— AI 起草 offer 信/证明等。

    ``key`` 对应 ``app.services.doc_templates.DOC_SPECS`` 内置注册项;自定义
    模板用 ``custom.<uuid>`` key 直接落行(注册表外)。``{{var}}`` 占位由
    ``build_doc_vars`` 预填,剩余空位由 LLM 补写。
    """

    __tablename__ = "doc_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    entity: Mapped[str] = mapped_column(String(32), nullable=False, server_default="any")
    body: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    updated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "key", name="uq_doc_templates_tenant_key"),
    )


class PromptTemplateVersion(Base):
    """提示词模板历史版本(每次保存追加一行,可回滚)。"""

    __tablename__ = "prompt_template_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    template_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

class Notification(Base):
    """站内通知 — watcher / 工单 / 申请等事件的统一投递渠道。

    前端通知铃按 (user_id, read_at IS NULL) 计未读。
    """

    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="system")
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    body: Mapped[str | None] = mapped_column(String(512), nullable=True)
    link: Mapped[str | None] = mapped_column(String(256), nullable=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_notifications_user_unread", "user_id", "read_at"),
    )


class AiWatcher(Base):
    """触发式任务(AI-native 轻量工作流)。

    ``kind`` 注册于 ``app.services.watchers.EVALUATORS``,每个评估器读 params
    检查条件,满足即写 Notification。每 tick 由 lifespan 的 _watch_loop 驱动。
    """

    __tablename__ = "ai_watchers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AutomationRule(Base):
    """事件驱动自动化规则(租户级)。

    挂在 ``write_audit`` 漏斗上:``entity_type`` + ``trigger`` 命中审计
    (entity_type, action) 后,先过 ``watch_fields``(仅 update,与
    ``detail.fields`` 键取交集)再过 ``conditions`` 快照条件,全通过则
    按 ``action`` 分发 —— notify 写 Notification,email 走 celery。

    ``trigger`` 取值为审计动作名:create/update/delete 是快捷值,
    其余具名动作(approve/reject/merge/…)自由匹配。
    """

    __tablename__ = "automation_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    trigger: Mapped[str] = mapped_column(String(64), nullable=False)
    # update 触发时限定关注字段;[]/None = 任意字段变更都算
    watch_fields: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # [{field, op, value}] AND 语义;field 必须是注册表快照白名单键
    conditions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False, server_default="notify")
    # [{"kind":"actor"|"role"|"user"|"record_owner"|"record_manager","value":?}]
    recipients: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    title_tpl: Mapped[str] = mapped_column(String(256), nullable=False)
    body_tpl: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AutomationRun(Base):
    """自动化执行记账 —— 幂等键 + 可观测性。

    ``UniqueConstraint(rule_id, event_key)``:event_key 取触发审计行的
    id,同一事件对同一规则只会执行一次,重复分发被唯一约束拦截。
    """

    __tablename__ = "automation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rule_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("automation_rules.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    event_key: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    trigger: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    # ok / skipped_no_recipients / queued_email / sent / email_failed / error
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="ok")
    recipients: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("rule_id", "event_key", name="uq_autorun_rule_event"),
    )


class Announcement(Base):
    """公司公告。audience 为空 = 全公司。"""

    __tablename__ = "announcements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    audience_org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="draft")
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class AnnouncementRead(Base):
    """公告已读打点。"""

    __tablename__ = "announcement_reads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    announcement_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("announcements.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    read_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("announcement_id", "user_id", name="uq_announcement_reads_one"),
    )


class EmployeeRequest(Base):
    """员工申请枢纽:培训/福利/设备等诉求的统一入口。

    ESS 提交 -> HR 队列审批;留言流双方可见。
    """

    __tablename__ = "employee_requests"

    KINDS = ("training", "benefit", "equipment", "wellness", "general")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="general")
    subject: Mapped[str] = mapped_column(String(128), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    decided_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class RequestComment(Base):
    """员工申请留言(员工与 HR 双方可见)。"""

    __tablename__ = "request_comments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("employee_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class SalaryBand(Base):
    """薪酬带宽:岗位/职级 -> 薪资区间参考。

    ``designation`` 为空表示通用带宽;``org_unit_id`` 为空表示全公司。
    gross 单位为元(整数);employer_cost 可选记录雇主成本侧。
    """

    __tablename__ = "salary_bands"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    designation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True
    )
    currency: Mapped[str] = mapped_column(String(8), nullable=False, server_default="CNY")
    gross_min: Mapped[int] = mapped_column(Integer, nullable=False)
    gross_max: Mapped[int] = mapped_column(Integer, nullable=False)
    employer_cost_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    employer_cost_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_salary_bands_tenant_name"),
    )


class HrCase(Base):
    """HR 工单:员工诉求通道。

    ``case_no`` 是租户内递增序列(insert 时锁 tenant 行取 max+1)。
    """

    __tablename__ = "hr_cases"

    KINDS = ("it", "admin", "payroll", "facility", "leave_dispute", "other")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_no: Mapped[int] = mapped_column(Integer, nullable=False)
    subject: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="other")
    priority: Mapped[str] = mapped_column(String(16), nullable=False, server_default="normal")
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="new")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    requester_employee_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True, index=True
    )
    assignee_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "case_no", name="uq_hr_cases_tenant_no"),
    )


class HrCaseUpdate(Base):
    """工单留言;``is_internal`` 的备注员工侧不可见。"""

    __tablename__ = "hr_case_updates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("hr_cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    is_internal: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeArticle(Base):
    """知识库文章 —— 制度文档 + assistant RAG 语料。"""

    __tablename__ = "knowledge_articles"

    # product_help 不在列:系统产品文档由 scripts/seed_product_help.py 直写库,
    # API 校验器拒绝创建/编辑/删除,与租户知识天然隔离且只读
    CATEGORIES = ("policy", "benefit", "it", "faq", "onboarding", "other")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False, server_default="other")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="draft")
    is_playbook: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeQueryLog(Base):
    """知识库检索日志(兜底回复标注):记录每次查询的命中数与最高分。

    对标 Moka「兜底回复自动标注 → HR 定期盘点知识缺口」:
    hits_count=0 的查询聚合成 /knowledge/gaps 的缺口榜单,
    让 RAG 形成"搜不到 → 补文章"的自改进飞轮。
    """

    __tablename__ = "knowledge_query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    hits_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    top_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # ui=知识库页搜索 / ess=员工自助问答 / assistant=助手工具
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default="ui")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class ReportSubscription(Base):
    """定时报表订阅:_watch_loop 按 weekday+hour 投递邮件摘要。"""

    __tablename__ = "report_subscriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    weekday: Mapped[int] = mapped_column(Integer, nullable=False)
    hour: Mapped[int] = mapped_column(Integer, nullable=False)
    last_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_true()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", "kind", name="uq_report_subs_one"),
    )


class TalentCampaign(Base):
    """人才再激活外联。

    ``filter`` 快照存筛选条件:{group_id?, skill_ids?, query?};
    创建时 resolve 成 ``talent_campaign_sends`` 行,无邮箱的记 skipped。
    """

    __tablename__ = "talent_campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    filter: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    subject: Mapped[str] = mapped_column(String(128), nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="draft")
    total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    sent_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    done_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class TalentCampaignSend(Base):
    """外联逐条投递记录。"""

    __tablename__ = "talent_campaign_sends"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("talent_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False
    )
    to_email: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="queued")
    error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("campaign_id", "candidate_id", name="uq_campaign_sends_one"),
    )


class McpToken(Base):
    """MCP 个人访问令牌(外部 AI 客户端用)。

    只存 ``sha256(token)`` —— 明文仅签发时返回一次,与 survey_links/invite_tokens
    同一惯例。``revoked_at`` 置位即失效;``last_used_at`` 记录最近兑换时间。
    """

    __tablename__ = "mcp_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SavedFilter(Base):
    """列表页保存筛选:per-user per-page 命名筛选。"""

    __tablename__ = "saved_filters"
    __table_args__ = (
        UniqueConstraint("user_id", "page", "name", name="uq_saved_filters_user_page_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Favorite(Base):
    """记录星标收藏:多态 entity+entity_id。"""

    __tablename__ = "favorites"
    __table_args__ = (
        UniqueConstraint("user_id", "entity", "entity_id", name="uq_favorites_user_entity"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class DictionaryItem(Base):
    """可配置下拉枚举:dict_key 命名空间 + value/label。"""

    __tablename__ = "dictionary_items"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "dict_key", "value", name="uq_dictionary_items_key_value"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dict_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    sort: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class JobRun(Base):
    """后台任务运行打点:loop/celery 任务运行历史。"""

    __tablename__ = "job_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str | None] = mapped_column(String(256), nullable=True)


class ImportJob(Base):
    """AI 导入向导任务:一次文件导入的全生命周期。

    ``status``: uploaded → mapped → checked → done / failed。
    ``mapping`` {field_key: 文件列名};``mapping_meta`` 附带置信度与
    来源(ai/heuristic/user)。``file_bytes`` 存原始文件,check/run
    时按需重新解析,不另建行表。
    """

    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset: Mapped[str] = mapped_column(String(32), nullable=False)
    file_name: Mapped[str] = mapped_column(String(256), nullable=False)
    file_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    columns: Mapped[list] = mapped_column(JSON, nullable=False, server_default="[]")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    mapping: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mapping_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="uploaded")
    check_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    run_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class InterviewScorecard(Base):
    """结构化面试评分卡(T9)—— 一场面试的考察框架。

    与 AI ``Interview`` 的关系:互补不替代。Interview 是机器面试会话;
    scorecard 是**人类面试官**的结构化评价载体(逐维评分 + 录用表态),
    可经 ``interview_id`` 挂在某场 AI 轮次旁,也可独立存在。

    ``criteria`` JSONB [{key,name,dimension,weight}] — dimension ∈
    tech/soft/general;key 创建时生成,反馈里按 key 引用,改名不断链。
    ``interviewer_ids`` JSONB [users.id] — 被指派的面试官(与
    ``Reservation.interviewer_ids`` 同口径:面试官=登录用户,
    展示名经其绑定员工档案解析)。
    ``status``: open → closed(closed 后维度与反馈冻结)。
    ``ai_summary`` JSONB:聚合分析产物 {summary,themes,disagreements,
    recommendation,feedback_count,generated_at}。
    """

    __tablename__ = "interview_scorecards"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    interview_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("interviews.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    criteria: Mapped[list] = mapped_column(JSON, nullable=False, server_default="[]")
    interviewer_ids: Mapped[list] = mapped_column(JSON, nullable=False, server_default="[]")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="open", index=True
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class InterviewFeedback(Base):
    """面试官反馈 — 一个面试官对一张评分卡的一卡。

    ``scores`` JSONB [{criterion,score(1-5),comment}],criterion 命中
    评分卡 criteria[].key;``verdict`` ∈ recommend/hold/reject。
    同一 (scorecard, interviewer_user) 唯一 — 重复提交=更新。
    """

    __tablename__ = "interview_feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scorecard_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("interview_scorecards.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    interviewer_user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    scores: Mapped[list] = mapped_column(JSON, nullable=False, server_default="[]")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "scorecard_id", "interviewer_user_id",
            name="uq_interview_feedback_card_user"
        ),
    )


class SearchIndexDefinition(Base):
    """搜索索引定义 —— 租户级覆盖行(可空即回落代码注册表默认)。

    声明式全局搜索(T9):哪类实体进索引、哪些字段、权重多少、
    标题/正文怎么渲染,都是**数据**。租户行只存覆盖(与 PromptTemplate
    同款模式),不存在 = 用 ``services.search_index.SEARCH_SOURCES`` 默认。
    ``version`` 变更+1,旧版本文档下次 sweep 重建。
    """

    __tablename__ = "search_index_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    fields: Mapped[list | None] = mapped_column(JSON, nullable=True)
    title_template: Mapped[str | None] = mapped_column(String(512), nullable=True)
    body_template: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    keyword_fields: Mapped[list | None] = mapped_column(JSON, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "entity", name="uq_search_def_tenant_entity"),
    )


class SearchDocument(Base):
    """搜索文档 —— 物化投影,**永远不是权威数据**。

    命中只携带 (entity, entity_id, 展示文本);读路径对每个命中重新走
    ``entities.py`` 的 build() 校验权限与行级 scope,源行不可见就丢。
    ``entity_id`` 故意不带外键:投影是重建的不是级联的,源行硬删后
    文档活到下次索引/清扫。
    """

    __tablename__ = "search_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False, server_default="")
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, nullable=False, server_default="[]")
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    definition_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "entity", "entity_id", name="uq_search_doc_entity_row"
        ),
    )


class IdempotencyKey(Base):
    """幂等键 —— 可重试写操作的"一行一锁"(T10)。

    - ``(tenant_id, scope, key)`` 唯一元组即锁:并发同 key 请求只有一个
      能插入,输家读回既有行再裁决。
    - ``request_hash``:canonical 请求(method+path+query+body)的
      SHA-256。同 key 不同 body = REUSED_KEY(409)而非回放。
    - ``status``: in_progress / completed / failed;终态行携带
      ``response_status`` + ``response_body``,重试时原样回放。
    - ``locked_at``:claim 时刻;持锁超过 stale 窗口(默认 120s)可被
      FOR UPDATE 接管 —— 进程中途死掉不会永久占锁。
    - ``expires_at``:默认 24h 回放窗;过期行由 claim 惰性清理 +
      ``_watch_loop`` 定时清扫。
    """

    __tablename__ = "idempotency_keys"

    STATUSES = ("in_progress", "completed", "failed")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    scope: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="in_progress", index=True
    )
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "scope", "key", name="uq_idem_tenant_scope_key"),
    )

class ModelCallLog(Base):
    """逐次 LLM 调用留痕(T11)。

    与 audit_logs 分工:audit 记"谁做了什么业务动作";本表记"每次模型调用的
    输入指纹/输出/耗时/错误" —— AI 可观测性层,回答"模型当时看到什么、返回什么、
    为什么错"。写入走独立 session(services.model_calls),业务回滚不丢日志。
    """

    __tablename__ = "model_call_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    call_type: Mapped[str] = mapped_column(String(48), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, server_default="")
    model: Mapped[str] = mapped_column(String(128), nullable=False, server_default="")
    prompt_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False, server_default="v1")
    input_digest: Mapped[str] = mapped_column(String(16), nullable=False, server_default="")
    output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_kind: Mapped[str] = mapped_column(String(32), nullable=False, server_default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        Index("ix_model_call_logs_tenant_created", "tenant_id", "created_at"),
        Index("ix_model_call_logs_tenant_type", "tenant_id", "call_type"),
    )


class JobKnowledgeSource(Base):
    """岗位级知识源(T11):每个岗位挂载的资料(JD 补充/公司介绍/技术栈说明…)。

    kind=text 时 ``content`` 存内联原文;kind=file 时经 ObjectStore 存原始文件,
    ``resume_parser._extract_text_by_format`` 抽取文本后切块进 ``job_knowledge_chunks``。
    状态机 indexing -> ready / failed,由 services.job_knowledge.index_source 驱动。
    """

    __tablename__ = "job_knowledge_sources"

    KINDS = ("text", "file")
    STATUSES = ("indexing", "ready", "failed")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(8), nullable=False, server_default="text")
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="indexing", index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    extraction_status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="none"
    )
    extraction_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class JobKnowledgeChunk(Base):
    """岗位知识切块 —— 可丢弃索引投影(recruit 同语义)。

    ``meta`` 冗余 {source_name, extraction_status, extraction_method},消费侧
    (match/question/interview prompt)直接拼上下文;``embedding`` best-effort,
    有则 cosine 召回,无则按序截断。
    """

    __tablename__ = "job_knowledge_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    embedding: Mapped[list[float] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_jk_chunks_job_src_idx", "job_id", "source_id", "chunk_index"),
    )


class ApprovalPolicy(Base):
    """审批策略(E3 审批中心):某 request_type 满足条件时走这条多级链。

    - ``request_type``: leave / overtime / attendance_request /
      expense_claim / employee_request
    - ``conditions`` JSON: [{field, op, value}],字段白名单见
      ``services.approvals.CONDITION_FIELDS``,op ∈ eq|ne|gt|lt|ge|le|in,
      全部 AND;空数组 = 该类型默认链
    - ``steps`` JSON: [{resolver, key?, label?}],resolver ∈
      manager(申请人直属上级)|role(该角色全部用户)|user(指定 user_id);
      key 对 role=角色名、user=user_id、manager 不用
    - 同类型多条策略按 ``priority`` 降序取首个匹配
    """

    __tablename__ = "approval_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    request_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    conditions: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    steps: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_true())
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ApprovalInstance(Base):
    """一次审批流的运行实例:某单据被某策略接管。

    ``request_type + request_id`` 唯一 —— 一张单据至多被一条链接管;
    无实例的单据走旧散装一级审批,向后兼容。
    """

    __tablename__ = "approval_instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    request_type: Mapped[str] = mapped_column(String(32), nullable=False)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    policy_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("approval_policies.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending", index=True
    )
    current_step_idx: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("tenant_id", "request_type", "request_id",
                         name="uq_approval_inst_req"),
    )


class ApprovalStep(Base):
    """审批链中的一步:由 resolver 动态解析出可审批用户集合。"""

    __tablename__ = "approval_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instance_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("approval_instances.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    resolver: Mapped[str] = mapped_column(String(16), nullable=False)
    resolver_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending"
    )
    decided_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("instance_id", "idx", name="uq_approval_step_idx"),
    )


class EmployeeSkill(Base):
    """P 批次人才画像 — 员工技能标签(带来源证据 + 待确认护栏)。

    - ``level`` 1-5(对齐 CandidateSkill / CompetencyRatings 刻度)
    - ``source``: resume / interview / appraisal / project / insight / manual / ai
    - ``status``: ``suggested``(AI/规则建议,待 HR/主管确认) / ``confirmed``
      —— AI 建议永不直接落 confirmed,这是画像可信度护栏
    - ``evidence`` JSON: {"kind": "project|interview|insight|appraisal|resume",
      "id": "...", "note": "..."} 记录标签出处,卡片页可穿透
    - 同员工同技能唯一:重复证据只升级 level/补 evidence,不插重复行
    """

    __tablename__ = "employee_skills"

    SOURCES = ("resume", "interview", "appraisal", "project", "insight",
               "manual", "ai")
    STATUSES = ("suggested", "confirmed")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    skill_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    level: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="manual", index=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="suggested", index=True
    )
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("employee_id", "skill_id", name="uq_employee_skills_pair"),
    )


class Project(Base):
    """P 批次 — 轻量项目登记(人才归属载体,不做任务管理)。

    ``required_skills`` JSON: [{"skill_id": "...", "min_level": 3}],
    供 search_talent 的确定性预筛与匹配理由生成。
    状态机: draft → active → done / cancelled;done 触发成员画像更新。
    """

    __tablename__ = "projects"

    STATUSES = ("draft", "active", "done", "cancelled")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft", index=True
    )
    owner_employee_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    required_skills: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ProjectMember(Base):
    """项目成员 — 员工档案「所属项目组」分组的载体。

    ``joined_via``: manual / ai_match —— AI 推荐加入的留痕,
    事后可统计 AI 选人命中率(贡献评价回填后对比)。
    """

    __tablename__ = "project_members"

    VIA = ("manual", "ai_match")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allocation_pct: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="100"
    )
    joined_via: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="manual"
    )
    contribution_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    left_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "employee_id", name="uq_project_members_pair"),
    )


class InsightNote(Base):
    """P 批次 — 画像证据统一输入口。

    kind: meeting(会议纪要,手动上传入口) / project_review(项目复盘) /
          manager_note(主管随手评) / milestone(里程碑)

    新建即触发画像更新管线(P2):LLM 从 body 提取能力证据 →
    EmployeeSkill(source=insight, status=suggested)。

    ``visibility``: manager_only(默认,员工本人不可见) /
    employee_visible —— 主管评价默认不给员工看,避免误伤。
    """

    __tablename__ = "insight_notes"

    KINDS = ("meeting", "project_review", "manager_note", "milestone")
    VISIBILITIES = ("manager_only", "employee_visible")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="manager_note", index=True
    )
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    author_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    source_ref: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    visibility: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="manager_only"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class GeneratedDocument(Base):
    """AI/导出生成的办公文档(P0 文档引擎产物:docx/xlsx/pptx)。

    kind:
    - candidate_report  人才分析报告(docx)
    - exam_paper        笔试卷(docx,可带答案)
    - recruiting_pack   招聘进度汇报(pptx)
    - candidate_list    候选人清单(xlsx)
    - pipeline          管道明细(xlsx)
    - payroll_sheet     薪酬底表(xlsx)

    params: 生成参数快照(candidate_id/question_set_id/period...),审计与复现用。
    """

    __tablename__ = "generated_documents"

    KINDS = (
        "candidate_report",
        "exam_paper",
        "recruiting_pack",
        "candidate_list",
        "pipeline",
        "payroll_sheet",
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    file_name: Mapped[str] = mapped_column(String(200), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime: Mapped[str] = mapped_column(String(120), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(400), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_by_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class RecruitRun(Base):
    """AI 招聘官 SOP 任务链运行实例(P1)。

    由助手工具发起:把「建 JD→解析技能→搜人→打分→面试→汇报」建成
    一条可追踪的 run,每步一个状态;前端步骤链卡片实时呈现。
    底层仍是单 Agent + 工具,但 run/step 提供了可观测的编排状态机。
    """

    __tablename__ = "recruit_runs"

    STATUS = ("running", "done", "failed", "cancelled")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False, default="full_pipeline")
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running", index=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    steps: Mapped[list["RecruitRunStep"]] = relationship(
        "RecruitRunStep",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RecruitRunStep.seq",
    )


class RecruitRunStep(Base):
    """SOP 任务链单步;agent_label 呈现「专项 Agent」分工(编排观感)。"""

    __tablename__ = "recruit_run_steps"

    STATUS = ("pending", "running", "done", "failed", "skipped", "waiting_confirm")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("recruit_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    key: Mapped[str] = mapped_column(String(60), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    agent_label: Mapped[str] = mapped_column(String(60), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    link: Mapped[str | None] = mapped_column(String(300), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    run: Mapped[RecruitRun] = relationship("RecruitRun", back_populates="steps")


class DecisionFeedback(Base):
    """P3 记忆中枢:管道决策埋点 — 从每次通过/淘汰/录用决策中学习。

    JobApplication stage 流转时记录:快照当时的 AI 匹配分(score)与
    逐维评分(criteria_snapshot),供「从你的选择里学到了什么」的
    维度权重建议计算 — 用得越多,建议越贴近真实用人偏好。
    """

    __tablename__ = "decision_feedback"

    DECISIONS = ("advanced", "offered", "hired", "rejected")

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    application_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("job_applications.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decision: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    from_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_stage: Mapped[str] = mapped_column(String(16), nullable=False)
    # 决策时快照:AI 匹配总分 + 逐维锚定评分 [{key,name,weight,level}]
    match_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    criteria_snapshot: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


# ==================== 人才盘点 / 人才池 ====================


class NineBoxAssessment(Base):
    """人才盘点九宫格落位(按盘点周期)。

    auto_perf/auto_potential 是系统初盘算分快照;calibrated=True 时
    perf_band/potential_band 为校准会后的人工覆盖值。band: 1低/2中/3高。
    """

    __tablename__ = "nine_box_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cycle: Mapped[str] = mapped_column(String(32), nullable=False, server_default="")
    auto_perf: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_potential: Mapped[float | None] = mapped_column(Float, nullable=True)
    perf_band: Mapped[int] = mapped_column(Integer, nullable=False, server_default="2")
    potential_band: Mapped[int] = mapped_column(Integer, nullable=False, server_default="2")
    calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_false())
    calibrated_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    calibrated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class TalentPool(Base):
    """人才池(北森式:建池→入池→跟踪培养→出池)。

    kind: succession 继任池 / high_potential 高潜池 / custom。
    """

    __tablename__ = "talent_pools"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, server_default="custom")
    target_designation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class TalentPoolMember(Base):
    """人才池成员。stage: in(入池) → developing(培养中) → ready(就绪) / exited(出池)。

    readiness 为就绪度 1-3(近期可接任/6-12月/1年以上),对标北森梯队阶段。
    """

    __tablename__ = "talent_pool_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pool_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("talent_pools.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(16), nullable=False, server_default="in")
    readiness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    added_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    exited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ==================== 职位管理 / 调薪评审 ====================


class Position(Base):
    """职位(Position)——Workday staffing model 的独立编制实体。

    一个 position = 一个座位,独立于任职人存在:
    status: open(空缺) / filled(在编) / frozen(冻结) / closed(撤销)。
    员工入职占坑(position_id 回填),离职/调动时 vacate 释放。
    """

    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    designation: Mapped[str | None] = mapped_column(String(128), nullable=True)
    org_unit_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("org_units.id", ondelete="SET NULL"), nullable=True, index=True
    )
    salary_band_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("salary_bands.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="open", index=True
    )
    incumbent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True
    )
    opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class CompReviewDecision(Base):
    """调薪评审决策记录(每员工每评审周期一条)。

    suggested_pct 由矩阵(绩效带 × compa-ratio 带)算出;
    final_pct/final_amount 为评审人调整后的值;status: draft / approved。
    """

    __tablename__ = "comp_review_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    tenant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    employee_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cycle: Mapped[str] = mapped_column(String(32), nullable=False)
    perf_band: Mapped[int] = mapped_column(Integer, nullable=False, server_default="2")
    ratio_band: Mapped[str] = mapped_column(String(8), nullable=False, server_default="mid")
    suggested_pct: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    final_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="draft")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "employee_id", "cycle", name="uq_comp_review_emp_cycle"),
    )
