import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Star } from "lucide-react";
import { fetchJSON } from "@/lib/api";
import { cn } from "@/lib/utils";

type Fav = { id: string; entity: string; entity_id: string };

/** 记录星标(MintHCM Favorites)。entity 白名单:talents/jobs/employees/cases。 */
export function FavoriteStar({ entity, entityId }: { entity: string; entityId: string }) {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["favorites", entity],
    queryFn: () => fetchJSON<Fav[]>(`/api/favorites?entity=${encodeURIComponent(entity)}`),
  });
  const fav = (data ?? []).find((f) => f.entity_id === entityId);

  const toggle = useMutation({
    mutationFn: () =>
      fav
        ? fetchJSON(`/api/favorites/${fav.id}`, { method: "DELETE" })
        : fetchJSON(`/api/favorites`, {
            method: "POST",
            body: JSON.stringify({ entity, entity_id: entityId }),
          }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["favorites", entity] }),
  });

  return (
    <button
      type="button"
      onClick={() => toggle.mutate()}
      disabled={toggle.isPending}
      title={fav ? "取消收藏" : "收藏"}
      aria-label={fav ? "取消收藏" : "收藏"}
      className="rounded-lg p-1 hover:bg-[var(--color-bg-muted)]"
    >
      <Star
        className={cn(
          "h-4.5 w-4.5",
          fav
            ? "fill-[var(--color-orange)] text-[var(--color-orange)]"
            : "text-[var(--color-text-muted)]",
        )}
      />
    </button>
  );
}
