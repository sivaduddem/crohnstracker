import { useMemo, useState } from "react";
import { mediaUrl } from "../api/client";

type Point = {
  case_id: string;
  x: number;
  y: number;
  gt_label: string;
  pred_label: string;
  confidence: number;
  severity?: string | null;
  thumbnail_url: string;
};

const COLORS: Record<string, string> = {
  normal: "#3ecf9a",
  erythema: "#e07a6a",
  erosion: "#e2b15a",
  ulcer: "#c084fc",
};

type Props = {
  points: Point[];
  method: "umap" | "tsne" | "pca";
  onMethodChange: (m: "umap" | "tsne" | "pca") => void;
  colorBy: "gt" | "pred" | "severity" | "confidence";
  onColorByChange: (c: "gt" | "pred" | "severity" | "confidence") => void;
  onSelect: (caseId: string) => void;
};

export function EmbeddingExplorer({
  points,
  method,
  onMethodChange,
  colorBy,
  onColorByChange,
  onSelect,
}: Props) {
  const [hover, setHover] = useState<Point | null>(null);
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const bounds = useMemo(() => {
    if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [points]);

  const project = (p: Point) => {
    const nx = (p.x - bounds.minX) / (bounds.maxX - bounds.minX || 1);
    const ny = (p.y - bounds.minY) / (bounds.maxY - bounds.minY || 1);
    return {
      left: `${(nx * 80 + 10) * scale + pan.x}%`,
      top: `${((1 - ny) * 80 + 10) * scale + pan.y}%`,
    };
  };

  const colorFor = (p: Point) => {
    if (colorBy === "confidence") {
      const t = Math.round(p.confidence * 255);
      return `rgb(${255 - t}, ${t}, 140)`;
    }
    if (colorBy === "severity") {
      return p.severity === "severe" ? "#e07a6a" : p.severity === "moderate" ? "#e2b15a" : "#3ecf9a";
    }
    const key = colorBy === "gt" ? p.gt_label : p.pred_label;
    return COLORS[key] || "#ccc";
  };

  return (
    <div className="panel fade-up">
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <div>
          <h2>Embedding explorer</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            Precomputed demo projections. Click a point to open the case.
          </p>
        </div>
        <span className="pill mock">Mock embeddings</span>
      </div>
      <div className="controls">
        <label>
          Method
          <select value={method} onChange={(e) => onMethodChange(e.target.value as "umap" | "tsne" | "pca")}>
            <option value="umap">UMAP</option>
            <option value="tsne">t-SNE</option>
            <option value="pca">PCA</option>
          </select>
        </label>
        <label>
          Color by
          <select
            value={colorBy}
            onChange={(e) => onColorByChange(e.target.value as "gt" | "pred" | "severity" | "confidence")}
          >
            <option value="gt">Ground truth</option>
            <option value="pred">Predicted</option>
            <option value="severity">Severity</option>
            <option value="confidence">Confidence</option>
          </select>
        </label>
        <button type="button" onClick={() => setScale((s) => Math.min(2.5, s + 0.15))}>
          Zoom in
        </button>
        <button type="button" onClick={() => setScale((s) => Math.max(0.6, s - 0.15))}>
          Zoom out
        </button>
        <button
          type="button"
          onClick={() => {
            setScale(1);
            setPan({ x: 0, y: 0 });
          }}
        >
          Reset
        </button>
      </div>
      <div
        className="scatter"
        onPointerDown={(e) => {
          const startX = e.clientX;
          const startY = e.clientY;
          const origin = { ...pan };
          const move = (ev: PointerEvent) => {
            setPan({
              x: origin.x + (ev.clientX - startX) / 8,
              y: origin.y + (ev.clientY - startY) / 8,
            });
          };
          const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up);
        }}
      >
        {points.map((p) => {
          const pos = project(p);
          return (
            <button
              key={p.case_id}
              type="button"
              className="dot"
              title={p.case_id}
              style={{ left: pos.left, top: pos.top, background: colorFor(p) }}
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(p.case_id)}
            />
          );
        })}
        {hover && (
          <div className="tooltip" style={{ left: project(hover).left, top: project(hover).top }}>
            <div>
              <strong>{hover.case_id}</strong>
            </div>
            <div>
              GT {hover.gt_label} / pred {hover.pred_label}
            </div>
            <img
              src={mediaUrl(hover.thumbnail_url)}
              alt={hover.case_id}
              style={{ width: 120, marginTop: 4, borderRadius: 6 }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
