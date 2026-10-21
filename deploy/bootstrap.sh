#!/usr/bin/env bash
# HROnfire 一键部署脚本 —— 云服务器(Ubuntu 22.04)首次部署。
# 用法(在阿里云控制台「远程连接/云助手」里执行):
#   bash <(curl -fsSL https://raw.githubusercontent.com/blueBananacc/HumanResource/main/deploy/bootstrap.sh)
# 或克隆后:bash deploy/bootstrap.sh
# 脚本幂等,可重复执行;再次执行走增量更新(git pull + rebuild)。
set -euo pipefail

LOG=/root/hronfire-bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== HROnfire bootstrap $(date -Is) ==="

APP_DIR=/opt/hronfire
REPO=https://github.com/blueBananacc/HumanResource.git
SERVER_IP=$(curl -fsSL --max-time 10 http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null \
  || curl -fsSL --max-time 10 ifconfig.me 2>/dev/null \
  || hostname -I | awk '{print $1}')
PUBLIC_URL="http://${SERVER_IP}"
step() { echo; echo "---- $* ----"; }

step "1/6 Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker
docker --version
docker compose version

step "2/6 拉取代码"
apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null
mkdir -p "$(dirname "$APP_DIR")"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi
cd "$APP_DIR"

step "3/6 生产配置"
# 密钥只生成一次,存在 deploy/.bootstrap-secrets;重跑脚本复用,避免会话/数据失效。
if [ ! -f deploy/.bootstrap-secrets ]; then
  cat > deploy/.bootstrap-secrets <<EOF
JWT_SECRET=$(openssl rand -hex 32)
LLM_KEY_ENCRYPTION_KEY=$(openssl rand -hex 32)
MACHINE_FP=$(openssl rand -hex 16)
ADMIN_PW=Hr0n-$(openssl rand -hex 4)
PGPW=$(openssl rand -hex 12)
EOF
  chmod 600 deploy/.bootstrap-secrets
fi
# shellcheck disable=SC1091
source deploy/.bootstrap-secrets

cp -f backend/.env.example backend/.env
sed -i \
  -e "s|^ENVIRONMENT=.*|ENVIRONMENT=prod|" \
  -e "s|^JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" \
  -e "s|^BOOTSTRAP_ADMIN_PASSWORD=.*|BOOTSTRAP_ADMIN_PASSWORD=$ADMIN_PW|" \
  -e "s|^LLM_KEY_ENCRYPTION_KEY=.*|LLM_KEY_ENCRYPTION_KEY=$LLM_KEY_ENCRYPTION_KEY|" \
  -e "s|^CORS_ORIGINS=.*|CORS_ORIGINS=[\"$PUBLIC_URL\"]|" \
  -e "s|^PUBLIC_BASE_URL=.*|PUBLIC_BASE_URL=$PUBLIC_URL|" \
  backend/.env
grep -q "^JWT_SECRET=dev" backend/.env && { echo "JWT_SECRET 未替换成功"; exit 1; } || true

step "4/6 生产 compose 覆盖文件(含随机数据库密码,不入库)"
cat > deploy/docker-compose.prod.yml <<EOF
# 由 bootstrap.sh 生成,含生产密钥 —— 不要提交到 git
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: "$PGPW"
  backend:
    environment:
      DATABASE_URL: postgresql+psycopg://freehire:$PGPW@postgres:5432/freehire
      MACHINE_FINGERPRINT_OVERRIDE: "$MACHINE_FP"
  worker:
    environment:
      DATABASE_URL: postgresql+psycopg://freehire:$PGPW@postgres:5432/freehire
      MACHINE_FINGERPRINT_OVERRIDE: "$MACHINE_FP"
EOF
grep -q "docker-compose.prod.yml" .gitignore 2>/dev/null \
  || echo "deploy/docker-compose.prod.yml" >> .git/info/exclude

step "5/6 构建并启动(首次构建约 5~15 分钟)"
cd deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

step "6/6 健康检查"
ok=""
for i in $(seq 1 60); do
  if curl -fsS --max-time 5 http://127.0.0.1/api/healthz >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
[ -n "$ok" ] || { echo "健康检查超时,最近后端日志:"; docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail 40 backend; exit 1; }
curl -fsS http://127.0.0.1/api/healthz; echo

# GitHub Actions 自动部署用的 SSH 密钥(公钥进 authorized_keys,私钥打印出来让你配 secret)
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ ! -f /root/.ssh/github_deploy ]; then
  ssh-keygen -t ed25519 -f /root/.ssh/github_deploy -N "" -C github-deploy -q
fi
grep -q "github-deploy" /root/.ssh/authorized_keys 2>/dev/null \
  || cat /root/.ssh/github_deploy.pub >> /root/.ssh/authorized_keys

echo
echo "================ 部署完成 ================"
echo "访问地址:      $PUBLIC_URL"
echo "管理员邮箱:    admin@example.com"
echo "管理员密码:    $ADMIN_PW"
echo
echo "== GitHub 自动部署 Secrets(仓库 Settings → Secrets and variables → Actions)=="
echo "DEPLOY_HOST = $SERVER_IP"
echo "DEPLOY_USER = root"
echo "DEPLOY_SSH_KEY = 下面这一整行(私钥):"
cat /root/.ssh/github_deploy
echo "=== bootstrap DONE ==="
