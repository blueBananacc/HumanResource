/** 纯 SVG 雷达图 — 多维评分的标准呈现(对标 Workday/大易的维度可视化)。
 * 零依赖:value 归一化到 [0,1],>=3 维才有意义;维度过多时标签省略采样。
 */
type Dim = { name: string; value: number };

export function RadarChart({
  dims,
  size = 220,
  maxLabel = 8,
}: {
  dims: Dim[];
  size?: number;
  maxLabel?: number;
}) {
  if (dims.length < 3) return null;
  const shown = dims.slice(0, maxLabel);
  const n = shown.length;
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.36;

  const pt = (i: number, r: number) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r] as const;
  };
  const ring = (f: number) =>
    shown.map((_, i) => pt(i, R * f).join(",")).join(" ");
  const poly = shown
    .map((d, i) => pt(i, R * Math.max(0, Math.min(1, d.value))).join(","))
    .join(" ");

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="多维评分雷达图"
      className="shrink-0"
    >
      {[1, 0.75, 0.5, 0.25].map((f) => (
        <polygon
          key={f}
          points={ring(f)}
          fill="none"
          stroke="var(--color-border-subtle)"
          strokeWidth="1"
        />
      ))}
      {shown.map((_, i) => {
        const [x, y] = pt(i, R);
        return (
          <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="var(--color-border-subtle)" strokeWidth="1" />
        );
      })}
      <polygon
        points={poly}
        fill="var(--color-accent)"
        fillOpacity="0.18"
        stroke="var(--color-accent)"
        strokeWidth="1.5"
      />
      {shown.map((d, i) => {
        const [x, y] = pt(i, R * Math.max(0, Math.min(1, d.value)));
        return <circle key={i} cx={x} cy={y} r="2.5" fill="var(--color-accent)" />;
      })}
      {shown.map((d, i) => {
        const [x, y] = pt(i, R + size * 0.11);
        const anchor = Math.abs(x - cx) < size * 0.08 ? "middle" : x < cx ? "end" : "start";
        return (
          <text
            key={i}
            x={x}
            y={y}
            textAnchor={anchor}
            dominantBaseline="middle"
            fontSize={size * 0.045}
            fill="var(--color-text-secondary)"
            style={{ fontFamily: "var(--font-body)" }}
          >
            {d.name}
          </text>
        );
      })}
    </svg>
  );
}
