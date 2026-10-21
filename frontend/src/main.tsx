import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource/domine/400.css";
import "@fontsource/domine/600.css";
import "@fontsource/domine/700.css";
import "@fontsource/lato/400.css";
import "@fontsource/lato/700.css";
import "./index.css";
import { router } from "./router";
import { AuthProvider } from "./lib/auth";
import { ConfirmProvider } from "./components/ui/confirm";
import { applyFontSize, getFontSize } from "./lib/fontSize";

// 渲染前应用字号偏好,避免首屏闪一下默认大小
applyFontSize(getFontSize());

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <ConfirmProvider>
          <RouterProvider router={router} />
        </ConfirmProvider>
      </AuthProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
