import { useEffect, useState } from "react";
import type { HealthResponse, HybridPredictResponse } from "@crohnstracker/shared";
import { fetchCases, fetchPredict } from "../api/client";
import { HybridPredictionView } from "../components/HybridPredictionView";
import { CaseComparison } from "../components/CaseComparison";

type CaseLite = {
  id: string;
  gt_label: string;
  pred_label?: string | null;
  confidence?: number | null;
  path: string;
  video_id: string;
  frame_idx: number;
  severity?: string | null;
};

export function HomePage({ health }: { health: HealthResponse | null }) {
  const [cases, setCases] = useState<CaseLite[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0);
  const [result, setResult] = useState<HybridPredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [similarId, setSimilarId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCases()
      .then((c) => {
        setCases(c);
        if (c[0]) setSelectedId(c[0].id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setError(null);
    fetchPredict(selectedId, topK, threshold)
      .then((r) => {
        setResult(r);
        setSimilarId(r.similar_cases[0]?.case_id ?? null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedId, topK, threshold]);

  const current = cases.find((c) => c.id === selectedId) || null;
  const selectedSimilar = result?.similar_cases.find((c: { case_id: string }) => c.case_id === similarId) || null;

  return (
    <div>
      <section className="panel fade-up" style={{ marginBottom: "1rem" }}>
        <h1 className="brand" style={{ fontSize: "clamp(2rem, 4vw, 2.8rem)" }}>
          CrohnTracker
        </h1>
        <p className="muted" style={{ maxWidth: "60ch" }}>
          Semantic retrieval complements classification for research and education — similarity is not a
          diagnosis. {health?.mode === "demo" ? "You are viewing demo fixtures." : ""}
        </p>
        <div className="controls">
          <label>
            Case
            <select value={selectedId || ""} onChange={(e) => setSelectedId(e.target.value)}>
              {cases.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ({c.gt_label})
                </option>
              ))}
            </select>
          </label>
          <label>
            Top-K
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </label>
          <label>
            Similarity threshold
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </label>
          <label>
            Distance metric
            <select value="cosine" disabled>
              <option value="cosine">cosine (MVP)</option>
            </select>
          </label>
        </div>
        {error && <p className="pill warn">{error}</p>}
      </section>

      <div style={{ display: "grid", gap: "1rem" }}>
        <HybridPredictionView
          result={result}
          loading={loading}
          onSelectSimilar={(id) => setSimilarId(id)}
        />
        <CaseComparison current={current} result={result} selectedSimilar={selectedSimilar} />
      </div>
    </div>
  );
}
