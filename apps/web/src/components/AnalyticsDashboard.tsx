import type { MetricsSummary } from "@crohnstracker/shared";

export function AnalyticsDashboard({ metrics }: { metrics: MetricsSummary | null }) {
  if (!metrics) return <div className="panel muted">Loading metrics…</div>;
  const { classification, retrieval } = metrics;

  return (
    <div className="fade-up">
      <div className="panel" style={{ marginBottom: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
          <div>
            <h2>Error & retrieval analytics</h2>
            <p className="muted" style={{ marginTop: 0 }}>
              Demo metrics are illustrative fixtures until a trained evaluation is run.
            </p>
          </div>
          <span className="pill mock">mode: {metrics.mode}</span>
        </div>
        <div className="metric-row">
          <div className="metric">
            <div className="muted">Accuracy</div>
            <div className="value">{(classification.accuracy * 100).toFixed(0)}%</div>
          </div>
          <div className="metric">
            <div className="muted">Macro F1</div>
            <div className="value">{(classification.macro_f1 * 100).toFixed(0)}%</div>
          </div>
          <div className="metric">
            <div className="muted">Recall@K</div>
            <div className="value">{(retrieval.recall_at_k * 100).toFixed(0)}%</div>
          </div>
          <div className="metric">
            <div className="muted">mAP</div>
            <div className="value">{(retrieval.map * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Per-class F1</h3>
          <ul>
            {Object.entries(classification.per_class_f1).map(([k, v]) => (
              <li key={k}>
                {k}: {(Number(v) * 100).toFixed(0)}%
              </li>
            ))}
          </ul>
          <h3>Confusion matrix</h3>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th />
                {classification.labels.map((l: string) => (
                  <th key={l} className="muted" style={{ fontWeight: 500 }}>
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {classification.confusion_matrix.map((row: number[], i: number) => (
                <tr key={classification.labels[i]}>
                  <td className="muted">{classification.labels[i]}</td>
                  {row.map((cell: number, j: number) => (
                    <td key={`${i}-${j}`} style={{ textAlign: "center", padding: "0.25rem" }}>
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h3>Retrieval performance</h3>
          <ul>
            <li>Precision@K: {(retrieval.precision_at_k * 100).toFixed(0)}%</li>
            <li>NDCG: {(retrieval.ndcg * 100).toFixed(0)}%</li>
            <li>Avg retrieval latency: {retrieval.latency_ms.toFixed(1)} ms</li>
            <li>Avg embedding time: {(retrieval.embedding_ms ?? 0).toFixed(1)} ms</li>
            <li>K: {retrieval.k}</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
