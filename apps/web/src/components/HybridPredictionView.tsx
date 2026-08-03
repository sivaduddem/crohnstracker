import type { HybridPredictResponse, SimilarCase } from "@crohnstracker/shared";
import { mediaUrl } from "../api/client";

type Props = {
  result: HybridPredictResponse | null;
  loading?: boolean;
  onSelectSimilar?: (caseId: string) => void;
};

export function HybridPredictionView({ result, loading, onSelectSimilar }: Props) {
  if (loading) return <div className="panel muted">Running prediction…</div>;
  if (!result) return <div className="panel muted">No prediction yet.</div>;

  const { prediction, similar_cases, similarity_analysis, is_mock } = result;

  return (
    <div className="panel fade-up">
      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
        <div>
          <h2>Hybrid prediction</h2>
          <p className="muted" style={{ marginTop: 0 }}>
            Classification plus nearest historical cases for context — not a diagnosis.
          </p>
        </div>
        {is_mock && <span className="pill mock">Mock prediction</span>}
      </div>

      <div className="metric-row" style={{ marginBottom: "1rem" }}>
        <div className="metric">
          <div className="muted">Prediction</div>
          <div className="value">{prediction.label}</div>
        </div>
        <div className="metric">
          <div className="muted">Confidence</div>
          <div className="value">{(prediction.confidence * 100).toFixed(0)}%</div>
        </div>
        <div className="metric">
          <div className="muted">NN distance</div>
          <div className="value">{similarity_analysis.nn_distance.toFixed(2)}</div>
        </div>
        <div className="metric">
          <div className="muted">Similarity conf.</div>
          <div className="value">{(similarity_analysis.similarity_confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      {similarity_analysis.novel_or_low_similarity && (
        <div className="banner" style={{ marginBottom: "1rem" }}>
          <strong>Novel or Low Similarity Case</strong>
          <span className="pill warn">Recommended for manual review</span>
        </div>
      )}

      <h3>Most similar cases</h3>
      <div className="grid-3">
        {similar_cases.map((c: SimilarCase) => (
          <button
            key={c.case_id}
            type="button"
            className="case-card"
            onClick={() => onSelectSimilar?.(c.case_id)}
            style={{ textAlign: "left", color: "inherit" }}
          >
            <img className="thumb" src={mediaUrl(c.thumbnail_url)} alt={`Demo case ${c.case_id}`} />
            <div style={{ marginTop: "0.45rem", fontWeight: 700 }}>{c.case_id}</div>
            <div className="muted">
              {c.pred_label} · {(c.similarity * 100).toFixed(0)}% similarity
            </div>
            <div className="muted">
              GT: {c.gt_label} · conf {(c.confidence * 100).toFixed(0)}%
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
