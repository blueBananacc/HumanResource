import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// 私有化场景下,前端最终走 nginx 反代,API 同域;开发态用 proxy 转发。
// 本机开发时 backend 无独立端口,/api 只能经 docker 网关 localhost:80 进容器,
// 故默认代理 http://localhost;如需改可在 frontend/.env.local 设 VITE_API_PROXY。
export default defineConfig(({ mode }) => {
  const env = { ...process.env, ...loadEnv(mode, __dirname) };
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY ?? "http://localhost",
          changeOrigin: true,
        },
      },
    },
  };
});
