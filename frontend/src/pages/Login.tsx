import { useEffect, useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { Building2 } from "lucide-react";
import { BrandShape } from "@/components/brand/BrandShape";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchJSON } from "@/lib/api";
import { useAuth, ApiError } from "@/lib/auth";

type SsoPublic = {
  enabled: boolean;
  display_name: string | null;
};

export function Login() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const loc = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [sso, setSso] = useState<SsoPublic | null>(null);

  useEffect(() => {
    fetchJSON<SsoPublic>("/api/sso/public")
      .then(setSso)
      .catch(() => setSso({ enabled: false, display_name: null }));
  }, []);

  if (user) return <Navigate to="/" replace />;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      const next = (loc.state as { from?: string } | null)?.from ?? "/";
      navigate(next, { replace: true });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(
          err.status === 401
            ? "邮箱或密码错误"
            : `登录失败 (HTTP ${err.status})`,
        );
      } else {
        setError("网络错误,请重试");
      }
    } finally {
      setSubmitting(false);
    }
  }

  function startSso() {
    // 后端会 302 跳到 IdP;直接整页跳转,fragment 在回调页处理
    window.location.href = "/api/sso/oidc/start";
  }

  return (
    <main className="relative h-full grid place-items-center bg-[var(--color-brand)] p-4 overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <BrandShape kind="circle" color="#FAA32B" className="-top-28 -left-20 w-[440px] h-[440px]" />
        <BrandShape kind="star" color="#FEBB59" className="top-[8%] left-[34%] w-[150px] h-[150px] rotate-12" />
        <BrandShape kind="free" color="#EE164F" className="bottom-[-80px] left-[6%] w-[340px] h-[340px]" />
        <BrandShape kind="triangle" color="#FEABB0" className="top-[14%] right-[10%] w-[220px] h-[220px] -rotate-12" />
        <BrandShape kind="arc" color="#F36488" className="bottom-[10%] right-[24%] w-[180px] h-[180px]" />
        <BrandShape kind="rect" color="#60032A" className="-bottom-24 -right-16 w-[380px] h-[380px] rotate-6" />
      </div>
      <div className="relative w-[440px] max-w-full bg-white rounded-2xl shadow-[0_12px_40px_rgba(43,10,25,0.35)] p-10 flex flex-col gap-7">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-[12px_12px_12px_3px] bg-[var(--color-accent)] text-white font-body font-bold flex items-center justify-center text-sm">
            Hi
          </div>
          <span className="font-body font-bold text-lg tracking-tight">HROnfire</span>
        </div>
        <div className="flex flex-col gap-1.5">
          <h1 className="font-heading font-semibold text-2xl text-[var(--color-text-primary)]">
            欢迎回来
          </h1>
          <p className="text-sm text-[var(--color-text-secondary)] font-body">
            登录到你的招聘工作台
          </p>
        </div>

        {sso?.enabled && (
          <div className="flex flex-col gap-3">
            <Button
              type="button"
              variant="secondary"
              fullWidth
              onClick={startSso}
            >
              <Building2 className="w-4 h-4" />
              {sso.display_name || "企业统一登录"}
            </Button>
            <div className="flex items-center gap-2 text-[11px] text-[var(--color-text-tertiary)] font-body">
              <span className="flex-1 h-px bg-[var(--color-border-subtle)]" />
              <span>或使用账号密码登录</span>
              <span className="flex-1 h-px bg-[var(--color-border-subtle)]" />
            </div>
          </div>
        )}

        <form onSubmit={onSubmit} className="flex flex-col gap-4">
          <Input
            label="邮箱"
            type="email"
            name="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@company.com"
            required
            autoComplete="email"
          />
          <Input
            label="密码"
            type="password"
            name="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
            autoComplete="current-password"
          />
          <div className="flex items-center justify-between text-[13px] font-body">
            <label className="flex items-center gap-2 text-[var(--color-text-secondary)] cursor-pointer">
              <input
                type="checkbox"
                className="w-3.5 h-3.5 rounded border-[var(--color-border-subtle)]"
              />
              <span>记住我</span>
            </label>
            <a
              href="#"
              className="text-[var(--color-text-primary)] font-medium"
            >
              忘记密码?
            </a>
          </div>
          {error && (
            <div className="text-[13px] text-[var(--color-danger)] bg-[var(--color-danger-soft)] px-3 py-2 rounded-md font-body">
              {error}
            </div>
          )}
          <Button type="submit" fullWidth disabled={submitting}>
            {submitting ? "登录中…" : "登录"}
          </Button>
        </form>
        <p className="text-xs text-[var(--color-text-tertiary)] text-center font-body">
          无账号? 联系系统管理员获取访问权限
        </p>
      </div>
    </main>
  );
}
