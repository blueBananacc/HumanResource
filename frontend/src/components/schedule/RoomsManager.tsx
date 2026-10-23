import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { DoorOpen, Package, Pencil, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer } from "@/components/ui/drawer";
import { ApiError, fetchJSON } from "@/lib/api";

/**
 * 会议室 / 资源管理抽屉(T2-2 补齐 — 原只有 API 没有 UI)。
 *
 * manage_settings 用户从排期页进入:增删改会议室与可预约资源,
 * 会议室可绑数据范围(org_unit);有预约记录的对象后端返回 409,
 * 引导停用(is_active=false)而非删除。
 */

export type Room = {
  id: string;
  name: string;
  location: string | null;
  capacity: number;
  equipment: string[];
  org_unit_id: string | null;
  is_active: boolean;
};

export type Resource = {
  id: string;
  name: string;
  type: string;
  is_active: boolean;
};

type OrgNode = { id: string; name: string; children: OrgNode[] };

const RESOURCE_TYPES: Record<string, string> = {
  equipment: "设备",
  phone_booth: "电话亭",
  other: "其他",
};

function flatten(nodes: OrgNode[], depth = 0): { id: string; label: string }[] {
  return nodes.flatMap((n) => [
    { id: n.id, label: `${"— ".repeat(depth)}${n.name}` },
    ...flatten(n.children, depth + 1),
  ]);
}

const inputCls =
  "h-9 px-3 rounded-lg border border-[var(--color-border-subtle)] text-sm font-body focus:outline-none focus:border-[var(--color-accent)]";

export function RoomsManager({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<"rooms" | "resources">("rooms");
  const [error, setError] = useState<string | null>(null);

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["rooms"] });
    qc.invalidateQueries({ queryKey: ["resources"] });
  };

  return (
    <Drawer
      open
      onClose={onClose}
      title="会议室与资源"
      description="管理可预约对象;已有预约记录的只能停用,不能删除"
      width={560}
    >
      <div className="flex flex-col gap-4">
        <div className="flex gap-2">
          {(
            [
              ["rooms", "会议室", DoorOpen],
              ["resources", "资源", Package],
            ] as const
          ).map(([v, label, Icon]) => (
            <button
              key={v}
              type="button"
              onClick={() => setTab(v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-body ${
                tab === v
                  ? "bg-[var(--color-accent)] text-white"
                  : "bg-[var(--color-bg-subtle)] text-[var(--color-text-secondary)]"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>
        {error && (
          <div className="text-[12px] text-[var(--color-error)] font-body">
            {error}
          </div>
        )}
        {tab === "rooms" ? (
          <RoomsTab setError={setError} onChanged={invalidate} />
        ) : (
          <ResourcesTab setError={setError} onChanged={invalidate} />
        )}
      </div>
    </Drawer>
  );
}

type TabProps = {
  setError: (e: string | null) => void;
  onChanged: () => void;
};

function RoomsTab({ setError, onChanged }: TabProps) {
  const [editing, setEditing] = useState<Room | null>(null);
  const [creating, setCreating] = useState(false);
  const query = useQuery({
    queryKey: ["rooms", "all"],
    queryFn: () => fetchJSON<Room[]>("/api/rooms"),
  });
  const orgQuery = useQuery({
    queryKey: ["org-tree"],
    queryFn: () => fetchJSON<OrgNode[]>("/api/org/tree"),
  });
  const orgOptions = flatten(orgQuery.data ?? []);

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/rooms/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      setError(null);
      onChanged();
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : "删除失败"),
  });

  const orgName = (id: string | null) =>
    orgOptions.find((o) => o.id === id)?.label ?? null;

  return (
    <div className="flex flex-col gap-2">
      {(query.data ?? []).map((r) => (
        <div
          key={r.id}
          className="border border-[var(--color-border-subtle)] rounded-lg px-3 py-2.5 flex items-center gap-3"
        >
          <div className="flex-1 min-w-0 text-[12px] font-body">
            <div className="text-[var(--color-text-primary)] font-medium truncate">
              {r.name}
              {!r.is_active && (
                <span className="ml-2 text-[10px] text-[var(--color-text-tertiary)]">
                  (已停用)
                </span>
              )}
            </div>
            <div className="text-[11px] text-[var(--color-text-tertiary)]">
              {r.location || "—"} · {r.capacity} 人
              {(r.equipment?.length ?? 0) > 0 && ` · ${r.equipment.join("/")}`}
              {r.org_unit_id && ` · ${orgName(r.org_unit_id) ?? "指定范围"}`}
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setCreating(false);
              setEditing(r);
            }}
            className="text-[var(--color-text-tertiary)] hover:text-[var(--color-accent)]"
            aria-label="编辑"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => del.mutate(r.id)}
            className="text-[var(--color-text-tertiary)] hover:text-[var(--color-error)]"
            aria-label="删除"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      {(query.data ?? []).length === 0 && (
        <div className="p-6 text-center text-[12px] text-[var(--color-text-tertiary)] border border-dashed border-[var(--color-border-subtle)] rounded-lg">
          还没有会议室
        </div>
      )}

      {creating || editing ? (
        <RoomForm
          key={editing?.id ?? "new"}
          initial={editing}
          orgOptions={orgOptions}
          setError={setError}
          onDone={() => {
            setCreating(false);
            setEditing(null);
            onChanged();
          }}
          onCancel={() => {
            setCreating(false);
            setEditing(null);
          }}
        />
      ) : (
        <Button
          variant="secondary"
          className="self-start !px-3 !py-1.5 !text-[12px]"
          onClick={() => {
            setEditing(null);
            setCreating(true);
          }}
        >
          <Plus className="w-3.5 h-3.5" />
          新增会议室
        </Button>
      )}
    </div>
  );
}

function RoomForm({
  initial,
  orgOptions,
  setError,
  onDone,
  onCancel,
}: {
  initial: Room | null;
  orgOptions: { id: string; label: string }[];
  setError: TabProps["setError"];
  onDone: () => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name ?? "");
  const [location, setLocation] = useState(initial?.location ?? "");
  const [capacity, setCapacity] = useState(initial?.capacity ?? 4);
  const [equipment, setEquipment] = useState(
    (initial?.equipment ?? []).join("\n"),
  );
  const [orgUnitId, setOrgUnitId] = useState(initial?.org_unit_id ?? "");
  const [active, setActive] = useState(initial?.is_active ?? true);

  const save = useMutation({
    mutationFn: () => {
      const body = {
        name: name.trim(),
        location: location.trim() || null,
        capacity,
        equipment: equipment
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
        org_unit_id: orgUnitId || null,
        is_active: active,
      };
      return initial
        ? fetchJSON<Room>(`/api/rooms/${initial.id}`, {
            method: "PUT",
            body: JSON.stringify(body),
          })
        : fetchJSON<Room>("/api/rooms", {
            method: "POST",
            body: JSON.stringify(body),
          });
    },
    onSuccess: () => {
      setError(null);
      onDone();
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  return (
    <div className="border border-[var(--color-border-subtle)] rounded-lg p-4 flex flex-col gap-3">
      <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
        {initial ? "编辑会议室" : "新增会议室"}
      </span>
      <div className="grid grid-cols-2 gap-2.5">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="名称,如 301"
          className={inputCls}
        />
        <input
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder="位置,如 3F 东侧"
          className={inputCls}
        />
        <label className="flex items-center gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]">
          容纳
          <input
            type="number"
            min={1}
            max={500}
            value={capacity}
            onChange={(e) =>
              setCapacity(Math.max(1, Math.min(500, Number(e.target.value) || 4)))
            }
            className={`${inputCls} w-20`}
          />
          人
        </label>
        <select
          value={orgUnitId}
          onChange={(e) => setOrgUnitId(e.target.value)}
          className={inputCls}
        >
          <option value="">全租户可见</option>
          {orgOptions.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      <textarea
        value={equipment}
        onChange={(e) => setEquipment(e.target.value)}
        rows={2}
        placeholder={"设备,每行一个\n如:\n白板\n视频会议一体机"}
        className="rounded-lg border border-[var(--color-border-subtle)] p-2.5 text-[13px] font-body focus:outline-none focus:border-[var(--color-accent)] resize-y"
      />
      <label className="flex items-center gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]">
        <input
          type="checkbox"
          checked={active}
          onChange={(e) => setActive(e.target.checked)}
          className="accent-[var(--color-accent)]"
        />
        启用(停用后不可预约,历史记录保留)
      </label>
      <div className="flex justify-end gap-2">
        <Button variant="secondary" className="!px-3 !py-1.5 !text-[12px]" onClick={onCancel}>
          取消
        </Button>
        <Button
          className="!px-3 !py-1.5 !text-[12px]"
          onClick={() => save.mutate()}
          disabled={save.isPending || !name.trim()}
        >
          保存
        </Button>
      </div>
    </div>
  );
}

function ResourcesTab({ setError, onChanged }: TabProps) {
  const [editing, setEditing] = useState<Resource | null>(null);
  const [creating, setCreating] = useState(false);
  const query = useQuery({
    queryKey: ["resources", "all"],
    queryFn: () => fetchJSON<Resource[]>("/api/resources"),
  });

  const del = useMutation({
    mutationFn: (id: string) =>
      fetchJSON(`/api/resources/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      setError(null);
      onChanged();
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : "删除失败"),
  });

  return (
    <div className="flex flex-col gap-2">
      {(query.data ?? []).map((r) => (
        <div
          key={r.id}
          className="border border-[var(--color-border-subtle)] rounded-lg px-3 py-2.5 flex items-center gap-3"
        >
          <div className="flex-1 min-w-0 text-[12px] font-body">
            <div className="text-[var(--color-text-primary)] font-medium truncate">
              {r.name}
              {!r.is_active && (
                <span className="ml-2 text-[10px] text-[var(--color-text-tertiary)]">
                  (已停用)
                </span>
              )}
            </div>
            <div className="text-[11px] text-[var(--color-text-tertiary)]">
              {RESOURCE_TYPES[r.type] ?? r.type}
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setCreating(false);
              setEditing(r);
            }}
            className="text-[var(--color-text-tertiary)] hover:text-[var(--color-accent)]"
            aria-label="编辑"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => del.mutate(r.id)}
            className="text-[var(--color-text-tertiary)] hover:text-[var(--color-error)]"
            aria-label="删除"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
      {(query.data ?? []).length === 0 && (
        <div className="p-6 text-center text-[12px] text-[var(--color-text-tertiary)] border border-dashed border-[var(--color-border-subtle)] rounded-lg">
          还没有资源
        </div>
      )}

      {creating || editing ? (
        <ResourceForm
          key={editing?.id ?? "new"}
          initial={editing}
          setError={setError}
          onDone={() => {
            setCreating(false);
            setEditing(null);
            onChanged();
          }}
          onCancel={() => {
            setCreating(false);
            setEditing(null);
          }}
        />
      ) : (
        <Button
          variant="secondary"
          className="self-start !px-3 !py-1.5 !text-[12px]"
          onClick={() => {
            setEditing(null);
            setCreating(true);
          }}
        >
          <Plus className="w-3.5 h-3.5" />
          新增资源
        </Button>
      )}
    </div>
  );
}

function ResourceForm({
  initial,
  setError,
  onDone,
  onCancel,
}: {
  initial: Resource | null;
  setError: TabProps["setError"];
  onDone: () => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initial?.name ?? "");
  const [type, setType] = useState(initial?.type ?? "equipment");
  const [active, setActive] = useState(initial?.is_active ?? true);

  const save = useMutation({
    mutationFn: () => {
      const body = { name: name.trim(), type, is_active: active };
      return initial
        ? fetchJSON<Resource>(`/api/resources/${initial.id}`, {
            method: "PUT",
            body: JSON.stringify(body),
          })
        : fetchJSON<Resource>("/api/resources", {
            method: "POST",
            body: JSON.stringify(body),
          });
    },
    onSuccess: () => {
      setError(null);
      onDone();
    },
    onError: (e) => setError(e instanceof ApiError ? e.message : "保存失败"),
  });

  return (
    <div className="border border-[var(--color-border-subtle)] rounded-lg p-4 flex flex-col gap-3">
      <span className="text-[12px] font-medium text-[var(--color-text-secondary)] font-body">
        {initial ? "编辑资源" : "新增资源"}
      </span>
      <div className="grid grid-cols-2 gap-2.5">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="名称,如 视频会议一体机"
          className={inputCls}
        />
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          className={inputCls}
        >
          {Object.entries(RESOURCE_TYPES).map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
      </div>
      <label className="flex items-center gap-1.5 text-[12px] font-body text-[var(--color-text-secondary)]">
        <input
          type="checkbox"
          checked={active}
          onChange={(e) => setActive(e.target.checked)}
          className="accent-[var(--color-accent)]"
        />
        启用(停用后不可预约,历史记录保留)
      </label>
      <div className="flex justify-end gap-2">
        <Button variant="secondary" className="!px-3 !py-1.5 !text-[12px]" onClick={onCancel}>
          取消
        </Button>
        <Button
          className="!px-3 !py-1.5 !text-[12px]"
          onClick={() => save.mutate()}
          disabled={save.isPending || !name.trim()}
        >
          保存
        </Button>
      </div>
    </div>
  );
}
