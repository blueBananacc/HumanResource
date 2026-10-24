# 公网部署指南（一步一步做）

目标：把 HROnfire 部署到一台云服务器，让外网用户通过 `https://你的域名` 访问；
本地开发照常，改完代码 `git push` 后服务器 1~3 分钟自动更新。

> 文档按「你还没有服务器」写。已有服务器可直接跳到第 3 步。

---

## 0. 先在本地确认（已做过可跳过）

```powershell
cd deploy
docker compose up -d        # 会自动加载 docker-compose.override.yml,后端源码热更新生效
```

- 改 `backend/app/**` → 保存后 uvicorn 自动 reload,`http://localhost` 立即生效
- 改 `backend/app/workers/**` → `docker compose restart worker`
- 新增 pip 依赖 / alembic 迁移 → `docker compose up -d --build backend worker`
- 前端开发用 `http://localhost:5173`（Vite 热更新）;`http://localhost` 要看到新样式需 `--build frontend`

## 1. 买服务器

- 配置：**2C4G 起步**（postgres + redis + backend + worker + nginx ≈ 1.5GB 内存;2G 会很紧）
- 系统：Ubuntu 22.04 / 24.04
- 国内云（阿里云/腾讯云轻量）开箱即用，但要备案才能绑域名；
  海外云（Hetzner / DigitalOcean / Vultr）免备案、域名即绑即用
- 控制台放行端口：**22(SSH)、80、443**；其余全部关闭（5432/6379 绝不能对公网开放）

## 2. 域名（可选但推荐）

买一个域名,DNS 加一条 `A 记录 → 服务器公网 IP`。
没有域名也能用 `http://IP` 访问，但无法上 HTTPS，且候选人面试邀请链接会是裸 IP。

## 3. 服务器初始化

```bash
# SSH 登录后:
curl -fsSL https://get.docker.com | sh     # 装 Docker
sudo usermod -aG docker $USER              # 免 sudo(重新登录生效)
```

## 4. 拉取代码

```bash
git clone https://github.com/blueBananacc/HumanResource.git
cd HumanResource
```

仓库若是 private：服务器上配 GitHub deploy key(仓库 Settings → Deploy keys),
或用 PAT:`git clone https://<TOKEN>@github.com/blueBananacc/HumanResource.git`。

## 5. 生产配置（关键,逐项检查）

```bash
cp backend/.env.example backend/.env
nano backend/.env
```

| 项 | 改成 |
|---|---|
| `ENVIRONMENT` | `prod` |
| `JWT_SECRET` | `openssl rand -hex 32` 生成 |
| `BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD` | 你的管理员账号 + 强密码(**只在首次建库时生效**) |
| `LLM_KEY_ENCRYPTION_KEY` | `openssl rand -hex 32` |
| `CORS_ORIGINS` | `["https://你的域名"]` |
| `PUBLIC_BASE_URL` | `https://你的域名`（面试邀请链接的前缀) |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_DEFAULT_MODEL` | 真实大模型 key;留空则跑 mock |

再编辑 `deploy/docker-compose.yml` 里的两处密码:

- `postgres` 服务的 `POSTGRES_PASSWORD`
- `backend`/`worker` 服务的 `DATABASE_URL` 中的密码段 —— **三处必须一致**

`MACHINE_FINGERPRINT_OVERRIDE` 换成你自己的固定随机值（`openssl rand -hex 16`),
不要继续用 `dev-compose-fixed-fingerprint-001`。

## 6. 首次启动

```bash
cd deploy
docker compose -f docker-compose.yml up -d --build
# ↑ 显式 -f 让 compose 不加载 override.yml —— 生产不带源码挂载
docker compose -f docker-compose.yml logs -f backend   # 看到迁移跑完、uvicorn 起来即可
curl http://127.0.0.1/api/healthz                       # 应返回 {"status":"ok"...}
```

浏览器开 `http://服务器IP` 应能看到登录页，用第 5 步的 admin 账号登录。

## 7. HTTPS（有域名时）

最简单的方案是宿主机装 Caddy,自动签/续 Let's Encrypt 证书:

```bash
sudo apt install -y caddy
sudo nano /etc/caddy/Caddyfile
```

Caddyfile 内容:

```
你的域名 {
    reverse_proxy 127.0.0.1:80
}
```

```bash
sudo systemctl reload caddy
```

`https://你的域名` 即开通。

## 8. 自动部署(推 main 即上线)

本地做:

1. 服务器上给 Actions 建一个受限的部署账号或用现有账号,把本地 `~/.ssh/id_rsa.pub` 加到服务器 `~/.ssh/authorized_keys`
2. GitHub 仓库 → Settings → Secrets and variables → Actions,新建:
   - `DEPLOY_HOST` = 服务器 IP
   - `DEPLOY_USER` = SSH 用户名
   - `DEPLOY_SSH_KEY` = 私钥全文
3. 仓库根目录建 `.github/workflows/deploy.yml`:

```yaml
name: deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: SSH deploy
        uses: appleboy/ssh-action@v1.2.0
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script: |
            cd ~/HumanResource
            git pull
            cd deploy
            docker compose -f docker-compose.yml up -d --build
```

之后你本地 `git push` 到 main,服务器几分钟内自动重建。Docker 层有缓存,
只改了后端代码时 backend/worker 重建很快;改了前端才会重新构建前端镜像。

## 9. 备份(至少做一条)

数据在两个 compose 卷里:`pg-data`(数据库)和 `object-data`(简历原文件/音频)。

```bash
# 数据库逻辑备份(可加 crontab 每天跑):
docker compose -f docker-compose.yml exec postgres \
  pg_dump -U freehire freehire > backup-$(date +%F).sql
```

`object-data` 用 `docker run --rm -v free-hire_object-data:/data -v $(pwd):/backup alpine tar czf /backup/objects-$(date +%F).tgz /data` 打包。

## 常见问题

- **改了代码服务器没更新** → 看 GitHub Actions 日志;多半是 secrets 没配或服务器上 `git pull` 冲突(不要在服务器上直接改文件)
- **面试邀请链接指向 localhost** → `PUBLIC_BASE_URL` 没配
- **License 失效/试用到期** → 生产指纹不要再用 dev 固定值;License 页可重新激活
- **邮件发不出** → SMTP 在 UI「SMTP 发件」里配;`EMAIL_SYNC_INTERVAL_SECONDS` 控制收简历频率
