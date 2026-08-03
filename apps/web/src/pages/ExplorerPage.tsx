import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchProjection } from "../api/client";
import { EmbeddingExplorer } from "../components/EmbeddingExplorer";

export function ExplorerPage() {
  const [method, setMethod] = useState<"umap" | "tsne" | "pca">("umap");
  const [colorBy, setColorBy] = useState<"gt" | "pred" | "severity" | "confidence">("gt");
  const [points, setPoints] = useState<
    Array<{
      case_id: string;
      x: number;
      y: number;
      gt_label: string;
      pred_label: string;
      confidence: number;
      severity?: string | null;
      thumbnail_url: string;
    }>
  >([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchProjection(method).then((r) => setPoints(r.points || []));
  }, [method]);

  return (
    <EmbeddingExplorer
      points={points}
      method={method}
      onMethodChange={setMethod}
      colorBy={colorBy}
      onColorByChange={setColorBy}
      onSelect={(id) => navigate(`/?case=${id}`)}
    />
  );
}
