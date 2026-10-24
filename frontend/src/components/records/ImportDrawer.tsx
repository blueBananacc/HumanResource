import { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { ApiError, getToken } from "@/lib/api";

type ImportResult = {
  created: number;
  skipped: { row: number; reason: string }[];
  total: number;
};

/** CSV 批量导入抽屉(T5-4)。`endpoint` = /api/talents/import 或 /api/employees/import。 */
export function ImportDrawer({
  endpoint,
  open,
  onClose,
  columnsHint,
  onDone,
}: {
  endpoint: string;
  open: boolean;
  onClose: () => void;
  columnsHint: string;
  onDone?: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImportResult | null>(null);

  async function submit() {
    const f = fileRef.current?.files?.[0];
    if (!f) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { authorization: `Bearer ${getToken() ?? ""}` },
        body: fd,
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new ApiError(resp.status, body?.detail ?? "导入失败");
      setResult(body as ImportResult);
      onDone?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "导入失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Drawer open={open} onClose={onClose} title="批量导入" width={520}>
      <div className="space-y-4">
        <p className="text-sm text-[var(--color-text-muted)]">
          上传 UTF-8 CSV(≤5MB,≤2000 行)。列:{columnsHint}
        </p>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          className="block w-full text-sm text-[var(--color-text)] file:mr-3 file:rounded-lg file:border-0 file:bg-[var(--color-cappuccino)] file:px-3 file:py-1.5 file:text-sm file:text-[var(--color-brand)]"
        />
        {error && <p className="text-sm text-[var(--color-accent)]">{error}</p>}
        <Button onClick={submit} disabled={busy}>
          <Upload className="mr-1.5 h-4 w-4" />
          {busy ? "导入中…" : "开始导入"}
        </Button>
        {result && (
          <div className="rounded-xl border border-[var(--color-border-subtle)] bg-white p-3">
            <p className="text-sm font-medium text-[var(--color-text)]">
              导入完成:新建 {result.created} 条,跳过 {result.skipped.length} 条(共 {result.total} 行)
            </p>
            {result.skipped.length > 0 && (
              <ul className="mt-2 max-h-56 space-y-1 overflow-auto text-xs text-[var(--color-text-muted)]">
                {result.skipped.map((s) => (
                  <li key={s.row}>
                    第 {s.row} 行:{s.reason}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </Drawer>
  );
}
