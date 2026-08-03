import type { HybridPredictResponse, SimilarCase } from "@crohnstracker/shared";
import { mediaUrl } from "../api/client";

type CaseLite = {
  id: string;
  gt_label: string;
  pred_label?: string | null;
  confidence?: number | null;
  path: string;
};

type Props = {
  current: CaseLite | null;
  result: HybridPredictResponse | null;
  selectedSimilar: SimilarCase | null;
};

function Side({
  title,
  imageUrl,
  gradcamUrl,
  pred,
  gt,
  conf,
  similarity,
}: {
  title: string;
  imageUrl: string;
  gradcamUrl?: string | null;
  pred?: string | null;
  gt?: string | null;
  conf?: number | null;
  similarity?: number | null;
}) {
  return (
    <div className="panel">
      <h3>{title}</h3>
      <img className="thumb" src={imageUrl} alt={title} />
      {gradcamUrl && (
        <>
          <p className="muted" style={{ marginBottom: 0 }}>
            Grad-CAM overlay
          </p>
          <img className="thumb" src={mediaUrl(gradcamUrl)} alt={`${title} gradcam`} />
        </>
      )}
      <ul className="muted">
        <li>Predicted: {pred ?? "—"}</li>
        <li>Ground truth: {gt ?? "—"}</li>
        <li>Confidence: {conf != null ? `${(conf * 100).toFixed(0)}%` : "—"}</li>
        {similarity != null && <li>Similarity: {(similarity * 100).toFixed(0)}%</li>}
      </ul>
    </div>
  );
}

export function CaseComparison({ current, result, selectedSimilar }: Props) {
  if (!current || !result) {
    return <div className="panel muted">Select a case to compare.</div>;
  }
  const similar = selectedSimilar || result.similar_cases[0] || null;
  const currentThumb = mediaUrl(`/media/images/${current.id}.png`);

  return (
    <div className="fade-up">
      <h2>Case comparison</h2>
      <p className="muted">Side-by-side clinical-style review. Demo overlays are mock explanations.</p>
      <div className="compare-row">
        <Side
          title="Current image"
          imageUrl={currentThumb}
          gradcamUrl={`/media/overlays/${current.id}_gradcam.png`}
          pred={result.prediction.label}
          gt={current.gt_label}
          conf={result.prediction.confidence}
        />
        {similar ? (
          <Side
            title="Retrieved similar case"
            imageUrl={mediaUrl(similar.thumbnail_url)}
            gradcamUrl={similar.gradcam_url}
            pred={similar.pred_label}
            gt={similar.gt_label}
            conf={similar.confidence}
            similarity={similar.similarity}
          />
        ) : (
          <div className="panel muted">No similar case above threshold.</div>
        )}
      </div>
      {result.similarity_analysis && (
        <div className="panel" style={{ marginTop: "1rem" }}>
          <strong>Embedding distance (1 − cosine): </strong>
          {result.similarity_analysis.nn_distance.toFixed(3)}
          {" · "}
          <strong>Avg Top-K distance: </strong>
          {result.similarity_analysis.avg_topk_distance.toFixed(3)}
          {" · "}
          <strong>Cluster: </strong>
          {result.similarity_analysis.cluster_id ?? "—"}
        </div>
      )}
    </div>
  );
}
