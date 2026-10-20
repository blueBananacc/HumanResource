import { cn } from "@/lib/utils";

/**
 * 品牌有机形状(参照 HiBob Brand Guidelines "Primary shapes"):
 * 圆形 / 矩形 / 三角 / 弧线 / 星形 / 自由形,全部是柔和、不规则的曲线。
 * 纯装饰元素,aria-hidden。
 */
export type ShapeKind = "circle" | "rect" | "triangle" | "arc" | "star" | "free";

const STAR_POINTS = Array.from({ length: 16 }, (_, i) => {
  const r = i % 2 === 0 ? 56 : 16;
  const a = (Math.PI / 8) * i - Math.PI / 2;
  return `${60 + r * Math.cos(a)},${60 + r * Math.sin(a)}`;
}).join(" ");

export function BrandShape({
  kind,
  color,
  className,
}: {
  kind: ShapeKind;
  color: string;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 120 120"
      aria-hidden
      className={cn("pointer-events-none absolute", className)}
    >
      {kind === "circle" && (
        <path
          fill={color}
          d="M62 6C94 4 118 30 114 64 110 96 82 118 52 112 22 106 2 82 8 50 12 24 34 8 62 6Z"
        />
      )}
      {kind === "rect" && (
        <path
          fill={color}
          d="M18 12C44 8 80 10 104 14 112 16 112 22 112 30 110 56 112 84 110 104 108 110 102 112 94 112 68 112 40 114 16 110 10 108 8 102 10 94 12 68 8 40 10 20 10 16 14 12 18 12Z"
        />
      )}
      {kind === "triangle" && (
        <path
          fill={color}
          d="M60 8C68 6 76 16 84 28 98 50 112 72 114 90 116 102 106 106 92 108 70 112 44 112 24 108 10 106 4 98 8 86 14 66 34 40 48 20 52 14 56 9 60 8Z"
        />
      )}
      {kind === "arc" && (
        <path
          fill="none"
          stroke={color}
          strokeWidth="24"
          d="M14 52C28 82 56 96 80 84 100 74 108 50 98 28"
        />
      )}
      {kind === "star" && <polygon fill={color} points={STAR_POINTS} strokeLinejoin="round" />}
      {kind === "free" && (
        <path
          fill={color}
          d="M18 104C6 70 8 20 28 14 44 10 46 52 58 56 64 38 74 18 90 30 116 48 110 78 84 92 62 104 36 110 18 104Z"
        />
      )}
    </svg>
  );
}
