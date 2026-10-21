# 环境现状速查（新会话接手用）

> 本文记录 HROnfire 当前「已部署」和「本地开发」两套环境的真实状态。
> 新会话接手时优先读本文件；搭建原理见 [deploy-guide.md](deploy-guide.md)。

## 一、线上环境（阿里云 ECS）

| 项 | 值 |
|---|---|
| 公网入口 | http://120.55.125.13 （HTTP，无域名/HTTPS） |
| 服务器 | 阿里云 ECS，Ubuntu 22.04，2C4G，root 登录 |
| 代码位置 | `/opt/hronfire`（git clone，origin 指向 `ghfast.top` 镜像） |
| 启动方式 | `cd /opt/hronfire/deploy && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d` |
| 生产配置 | `deploy/docker-compose.prod.yml`（bootstrap 生成）+ `backend/.env` |
| 数据卷 | `pg-data`（Postgres）、`object-data`（简历文件/音频）、`redis-data` |
| 管理员 | `admin@example.com`（密码见服务器 `/root/hronfire-bootstrap.log` 末尾或 `backend/.env` 的 `ADMIN_INITIAL_PASSWORD`） |
| 部署密钥 | `/root/.ssh/github_deploy`（公钥已加入 root authorized_keys） |
| 重建脚本 | `bash /opt/hronfire/deploy/bootstrap.sh`（幂等，可重跑） |

**安全组**：入方向需放行 22（SSH/Actions）、80（HTTP）。443 建议预留。不要放行 5432/6379/8000。

**流量计费**：阿里云只计出方向流量（访客下载页面/文件）。构建时拉包是入方向，不计费。控制台有 20GiB 出方向免费额度。

## 二、本地开发环境（Windows，Docker Desktop）

```powershell
# 1. 启动 Docker Desktop（若未运行）
Start-Process "C:\Users\666\AppData\Local\Programs\DockerDesktop\Docker Desktop.exe"

# 2. 起全部容器（自动加载 docker-compose.override.yml）
cd deploy && docker compose up -d

# 3. 起 Vite 开发服务器（注意代理变量，见下）
cd frontend; $env:VITE_API_PROXY="http://localhost"; npm run dev
```

| 入口 | 地址 |
|---|---|
| 开发用（热更新） | http://localhost:5173 |
| 镜像版前端 | http://localhost （80 端口 nginx） |
| 健康检查 | http://localhost/api/healthz |

**热更新范围**
- `frontend/src/**` 保存即生效（Vite）
- `backend/app/**` 保存即 reload（override 把 `../backend` 挂进容器 + uvicorn `--reload`）
- Celery worker 代码：`docker compose restart worker`
- 新依赖/迁移：`docker compose up -d --build backend worker`

## 三、已知坑（重要）

1. **Vite 代理默认指容器内网**：`vite.config.ts` 里 `/api` 代理默认 `http://backend:8000`，宿主机直连时**必须**设 `VITE_API_PROXY=http://localhost`（nginx 转发），否则 API 请求返回空。可考虑写入 `frontend/.env.local` 固化。
2. **服务器访问 GitHub 受限**：`raw.githubusercontent.com` 直连会被 reset；clone/pull 走 `ghfast.top` 镜像。bootstrap 内已做镜像源容错（apt/pip/npm 走阿里云源）。
3. **本机出站网络受限**：开发机不能直连服务器 SSH（出站被拦），服务器侧操作需在阿里云控制台「远程命令/Cloud Assistant」或 VNC 里执行；但 Actions → 服务器方向是通的。
4. **部署 IP 与控制台显示不一致**：元数据探测 `120.55.125.13`（以此为准），控制台截图曾显示 `120.55.123.13`。
5. **登录页预填演示凭据**：`frontend/src/pages/Login.tsx` 预填了 admin 账号密码方便访客一键登录——仅适合演示，正式使用需移除或换密码。

## 四、CI/CD 自动部署

`.github/workflows/deploy.yml`：push 到 `main` → SSH 上服务器 → `git pull` → **按变更路径选择性重建**：
- `backend/` 或 `deploy/` 变动 → 重建 backend + worker
- `frontend/` 变动 → 重建 frontend
- 仅 docs → 不重建

Secrets（仓库 Settings → Secrets → Actions）：`DEPLOY_HOST`、`DEPLOY_USER`、`DEPLOY_SSH_KEY`（服务器上 `/root/.ssh/github_deploy` 的私钥）。

**已踩过的坑**：`DEPLOY_SSH_KEY` 粘贴不完整会报 `ssh.ParsePrivateKey: ssh: no key found`——必须包含 BEGIN/END 行及全部 5 行 base64。服务器自测：`ssh -i /root/.ssh/github_deploy -o StrictHostKeyChecking=no root@localhost "echo ok"`。

## 五、重置线上 admin 密码（与登录页预填保持一致时）

```bash
cd /opt/hronfire/deploy && docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T backend python -c "from app.infra.db import SessionLocal; from app.infra.security import hash_password; from app.domain.models import User; s=SessionLocal(); u=s.query(User).filter(User.email=='admin@example.com').first(); u.password_hash=hash_password('admin123456'); s.commit(); print('done')"
```
