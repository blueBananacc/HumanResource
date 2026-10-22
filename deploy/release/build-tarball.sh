#!/usr/bin/env bash
# Free-Hire 离线发布包构建脚本(维护者本地执行)。
#
# 产物:deploy/release/dist/free-hire-{VERSION}.tar.gz
#       内含 5 个 docker image tarball + docker-compose.yml + install.sh +
#       .env.example + docs/ops/。解压后运行 ./install.sh 即可。
#
# 用法:
#   cd deploy/release
#   ./build-tarball.sh             # VERSION 自动取 git describe
#   ./build-tarball.sh 1.2.3        # 指定版本
#   FREEHIRE_PUSH_DOCKERHUB=1 ./build-tarball.sh   # 顺便 push 到 Docker Hub(可选)
#
# 不依赖外部 CI,在 macOS / Linux 上 docker build + docker save 都跑得了。
# install.sh 见同目录,docs 见 docs/ops/offline-install.md。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

# ---- 版本号 ----
VERSION="${1:-}"
if [ -z "${VERSION}" ]; then
  if VERSION="$(git -C "${REPO}" describe --tags --always --dirty 2>/dev/null)"; then
    :
  else
    VERSION="$(date -u +%Y%m%d-%H%M%S)"
  fi
fi
# 目标架构:部署环境/测试机绝大多数是 linux/amd64。Mac M 系列本地 build 默认产
# arm64 镜像,scp 到 amd64 ECS 会起不来。这里强制 amd64,需要 arm 时显式覆盖:
#   FREEHIRE_TARGET_PLATFORM=linux/arm64 ./build-tarball.sh
TARGET_PLATFORM="${FREEHIRE_TARGET_PLATFORM:-linux/amd64}"

# 国内构建机访问 deb.debian.org / pypi 常 502,可指定镜像源:
#   FREEHIRE_APT_MIRROR=mirrors.tuna.tsinghua.edu.cn ./build-tarball.sh
#   FREEHIRE_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple ./build-tarball.sh
APT_MIRROR="${FREEHIRE_APT_MIRROR:-}"
PIP_INDEX="${FREEHIRE_PIP_INDEX:-}"
# 国内构建机 npm install 常超时,可指定 npmmirror:
#   FREEHIRE_NPM_REGISTRY=https://registry.npmmirror.com ./build-tarball.sh
NPM_REGISTRY="${FREEHIRE_NPM_REGISTRY:-}"
# assistant 基础镜像 node:22-alpine 拉不动时可换国内 mirror:
#   FREEHIRE_ASSISTANT_BASE_IMAGE=docker.m.daocloud.io/library/node:22-alpine
ASSISTANT_BASE_IMAGE="${FREEHIRE_ASSISTANT_BASE_IMAGE:-}"

BACKEND_BUILD_ARGS=()
[ -n "${APT_MIRROR}" ] && BACKEND_BUILD_ARGS+=(--build-arg "APT_MIRROR=${APT_MIRROR}")
[ -n "${PIP_INDEX}" ]  && BACKEND_BUILD_ARGS+=(--build-arg "PIP_INDEX=${PIP_INDEX}")

NODE_BUILD_ARGS=()
[ -n "${NPM_REGISTRY}" ] && NODE_BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")

ASSISTANT_BUILD_ARGS=()
[ -n "${NPM_REGISTRY}" ] && ASSISTANT_BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")
[ -n "${ASSISTANT_BASE_IMAGE}" ] && ASSISTANT_BUILD_ARGS+=(--build-arg "BASE_IMAGE=${ASSISTANT_BASE_IMAGE}")


echo "==> 构建 Free-Hire 离线安装包 v${VERSION} (platform=${TARGET_PLATFORM})"

# ---- 准备 dist 目录 ----
DIST="${HERE}/dist/free-hire-${VERSION}"
rm -rf "${DIST}"
mkdir -p "${DIST}"

# ---- 1. build images ----
# 用 buildx --load 把跨平台镜像塞回本地 docker images,后续 docker save 才能找到。
echo "==> [1/7] 构建 backend 镜像"
docker buildx build \
  --platform "${TARGET_PLATFORM}" \
  --load \
  ${BACKEND_BUILD_ARGS[@]+"${BACKEND_BUILD_ARGS[@]}"} \
  -t "free-hire-backend:${VERSION}" \
  -t "free-hire-backend:latest" \
  -f "${REPO}/deploy/Dockerfile.backend" \
  "${REPO}/backend"

echo "==> [2/7] 构建 frontend 镜像"
docker buildx build \
  --platform "${TARGET_PLATFORM}" \
  --load \
  ${NODE_BUILD_ARGS[@]+"${NODE_BUILD_ARGS[@]}"} \
  -t "free-hire-frontend:${VERSION}" \
  -t "free-hire-frontend:latest" \
  -f "${REPO}/deploy/Dockerfile.frontend" \
  "${REPO}/frontend"

echo "==> [3/7] 构建 assistant 镜像"
docker buildx build \
  --platform "${TARGET_PLATFORM}" \
  --load \
  ${ASSISTANT_BUILD_ARGS[@]+"${ASSISTANT_BUILD_ARGS[@]}"} \
  -t "free-hire-assistant:${VERSION}" \
  -t "free-hire-assistant:latest" \
  -f "${REPO}/deploy/Dockerfile.assistant" \
  "${REPO}/assistant"

# ---- 2. pull 依赖镜像(目标环境可能拿不到 Docker Hub) ----
echo "==> [4/7] 拉取依赖镜像 (postgres / redis) [${TARGET_PLATFORM}]"
docker pull --platform "${TARGET_PLATFORM}" postgres:16-alpine
docker pull --platform "${TARGET_PLATFORM}" redis:7-alpine

# ---- 3. save tarballs ----
echo "==> [5/7] docker save 镜像到 ${DIST}"
docker save "free-hire-backend:${VERSION}" | gzip > "${DIST}/free-hire-backend.tar.gz"
docker save "free-hire-frontend:${VERSION}" | gzip > "${DIST}/free-hire-frontend.tar.gz"
docker save "free-hire-assistant:${VERSION}" | gzip > "${DIST}/free-hire-assistant.tar.gz"
docker save "postgres:16-alpine" | gzip > "${DIST}/postgres-16-alpine.tar.gz"
docker save "redis:7-alpine" | gzip > "${DIST}/redis-7-alpine.tar.gz"

# ---- 4. 复制 compose + install + env + docs ----
echo "==> [6/7] 复制 compose / install / env / docs"

# 发布包里的 docker-compose.yml = 离线版
cp "${REPO}/deploy/docker-compose.offline.yml" "${DIST}/docker-compose.yml"

cp "${HERE}/install.sh" "${DIST}/install.sh"
chmod +x "${DIST}/install.sh"

# 备份 / 还原脚本一起发布
cp "${HERE}/backup.sh" "${DIST}/backup.sh"
chmod +x "${DIST}/backup.sh"
cp "${HERE}/restore.sh" "${DIST}/restore.sh"
chmod +x "${DIST}/restore.sh"

cp "${REPO}/backend/.env.example" "${DIST}/.env.example"

# 运维文档
mkdir -p "${DIST}/docs/ops"
cp "${REPO}/docs/ops/offline-install.md" "${DIST}/docs/ops/" 2>/dev/null || true

# 版本文件 + checksum
echo "${VERSION}" > "${DIST}/VERSION"
(
  cd "${DIST}"
  if command -v sha256sum >/dev/null; then
    sha256sum *.tar.gz > SHA256SUMS
  else
    # macOS fallback: shasum -a 256 输出格式与 sha256sum 一致
    shasum -a 256 *.tar.gz > SHA256SUMS
  fi
)

# ---- 5. 打包整个交付目录 ----
echo "==> [7/7] tar 整个交付目录"
( cd "${HERE}/dist" && tar czf "free-hire-${VERSION}.tar.gz" "free-hire-${VERSION}/" )

OUT="${HERE}/dist/free-hire-${VERSION}.tar.gz"
SIZE=$(du -h "${OUT}" | awk '{print $1}')

echo
echo "✓ 完成: ${OUT} (${SIZE})"
echo
echo "下一步:"
echo "  scp ${OUT} client@target-host:/opt/"
echo "  ssh client@target-host 'cd /opt && tar xzf free-hire-${VERSION}.tar.gz && cd free-hire-${VERSION} && ./install.sh'"
