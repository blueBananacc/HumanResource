#!/usr/bin/env bash
# HROnfire 一键部署脚本 —— 云服务器(Ubuntu 22.04)首次部署。
# 用法(在阿里云控制台「远程连接/云助手」里执行):
#   bash <(curl -fsSL https://raw.githubusercontent.com/blueBananacc/HumanResource/main/deploy/bootstrap.sh)
# 或克隆后:bash deploy/bootstrap.sh
# 脚本幂等,可重复执行;再次执行走增量更新(git pull + rebuild)。
# 针对国内服务器:GitHub / Docker Hub / get.docker.com 均可能被拦,
# 每个环节都有镜像源或 apt 源兜底,会自动试错,不需要人工干预。
set -euo pipefail

LOG=/root/hronfire-bootstrap.log
echo "=== HROnfire bootstrap $(date -Is) ===" > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "script pid $$ starting"

APP_DIR=/opt/hronfire
REPO=https://github.com/blueBananacc/HumanResource.git
SERVER_IP=$(curl -fsSL --max-time 10 http://100.100.100.200/latest/meta-data/eipv4 2>/dev/null \
  || curl -fsSL --max-time 10 ifconfig.me 2>/dev/null \
  || hostname -I | awk '{print $1}')
PUBLIC_URL="http://${SERVER_IP}"
step() { echo; echo "---- $* ----"; }

step "1/6 Docker"
if ! command -v docker >/dev/null 2>&1; then
  if curl -fsSL --connect-timeout 15 https://get.docker.com -o /tmp/get-docker.sh; then
    sh /tmp/get-docker.sh
  else
    echo "get.docker.com 不可达,改用 Ubuntu 官方源 docker.io"
    apt-get update -qq && apt-get install -y -qq docker.io
  fi
fi
systemctl enable --now docker
docker --version

# compose 插件探测:优先 docker compose(v2),退化 docker-compose(v1)
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  apt-get update -qq
  apt-get install -y -qq docker-compose-v2 || apt-get install -y -qq docker-compose
  if docker compose version >/dev/null 2>&1; then COMPOSE="docker compose"; else COMPOSE="docker-compose"; fi
fi
echo "compose: $COMPOSE"
$COMPOSE version

# Docker Hub 拉取探测:拉不动时自动配 registry 镜像站再重试
if ! timeout 90 docker pull hello-world >/dev/null 2>&1; then
  echo "Docker Hub 直连失败,配置 registry mirrors 后重试"
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.m.daocloud.io",
    "https://docker.xuanyuan.me",
    "https://hub.rat.dev"
  ]
}
EOF
  systemctl restart docker
  sleep 3
  timeout 90 docker pull hello-world >/dev/null 2>&1 \
    && echo "镜像源可用" || echo "警告: 镜像源仍不可用,后续构建可能拉镜像失败"
fi

step "2/6 拉取代码"
apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null
mkdir -p "$(dirname "$APP_DIR")"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only || echo "git pull 失败,沿用当前代码继续部署"
else
  cloned=""
  for src in \
    "$REPO" \
    "https://ghfast.top/$REPO" \
    "https://gh-proxy.com/$REPO" \
    "https://mirror.ghproxy.com/$REPO" ; do
    echo "尝试克隆: $src"
    if git clone --depth 1 "$src" "$APP_DIR"; then cloned=1; break; fi
    rm -rf "$APP_DIR"
  done
  [ -n "$cloned" ] || { echo "所有 GitHub 源均不可达"; exit 1; }
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

step "4/6 生产 compose 覆盖文件(含随机数据库密码 + 国内构建镜像源,不入库)"
cat > deploy/docker-compose.prod.yml <<EOF
# 由 bootstrap.sh 生成,含生产密钥与国内构建镜像源 —— 不要提交到 git
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: "$PGPW"
  backend:
    build:
      args:
        # 阿里云内网源(ECS 免流量);非阿里云环境可换成 mirrors.aliyun.com
        APT_MIRROR: "mirrors.cloud.aliyuncs.com"
        PIP_INDEX: "https://mirrors.aliyun.com/pypi/simple/"
    environment:
      DATABASE_URL: postgresql+psycopg://freehire:$PGPW@postgres:5432/freehire
      MACHINE_FINGERPRINT_OVERRIDE: "$MACHINE_FP"
  worker:
    build:
      args:
        APT_MIRROR: "mirrors.cloud.aliyuncs.com"
        PIP_INDEX: "https://mirrors.aliyun.com/pypi/simple/"
    environment:
      DATABASE_URL: postgresql+psycopg://freehire:$PGPW@postgres:5432/freehire
      MACHINE_FINGERPRINT_OVERRIDE: "$MACHINE_FP"
  frontend:
    build:
      args:
        NPM_REGISTRY: "https://registry.npmmirror.com"
EOF
grep -q "docker-compose.prod.yml" .git/info/exclude 2>/dev/null \
  || echo "deploy/docker-compose.prod.yml" >> .git/info/exclude
# .bootstrap-secrets / .env 同理,防误提交
for f in "deploy/.bootstrap-secrets" "backend/.env"; do
  grep -qxF "$f" .git/info/exclude 2>/dev/null || echo "$f" >> .git/info/exclude
done

step "5/6 构建并启动(首次构建约 5~15 分钟)"
cd deploy
$COMPOSE -f docker-compose.yml -f docker-compose.prod.yml up -d --build

step "6/6 健康检查"
ok=""
for i in $(seq 1 60); do
  if curl -fsS --max-time 5 http://127.0.0.1/api/healthz >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
[ -n "$ok" ] || { echo "健康检查超时,最近后端日志:"; $COMPOSE -f docker-compose.yml -f docker-compose.prod.yml logs --tail 40 backend; exit 1; }
curl -fsS http://127.0.0.1/api/healthz; echo

# GitHub Actions 自动部署用的 SSH 密钥(公钥进 authorized_keys,私钥打印出来让你配 secret)
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ ! -f /root/.ssh/github_deploy ]; then
  ssh-keygen -t ed25519 -f /root/.ssh/github_deploy -N "" -C github-deploy -q
fi
touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
grep -q "github-deploy" /root/.ssh/authorized_keys \
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
