import { useEffect, useState } from "react";
import type { HybridPredictResponse } from "@crohnstracker/shared";
import { fetchCases, fetchPredict, fetchSequences, mediaUrl, postReviewFlag } from "../api/client";
import { CaseComparison } from "../components/CaseComparison";
import { HybridPredictionView } from "../components/HybridPredictionView";

type CaseLite = {
  id: string;
  gt_label: string;
  pred_label?: string | null;
  confidence?: number | null;
  path: string;
  video_id: string;
  frame_idx: number;
};

type Event = {
  sequence_id: string;
  start_frame: number;
  end_frame: number;
  label: string;
  peak_confidence: number;
  case_ids: string[];
};

export function ReviewPage() {
  const [cases, setCases] = useState<CaseLite[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [result, setResult] = useState<HybridPredictResponse | null>(null);
  const [similarId, setSimilarId] = useState<string | null>(null);
  const [note, setNote] = useState("");

  useEffect(() => {
    Promise.all([fetchCases(), fetchSequences()]).then(([c, e]) => {
      setCases(c);
      setEvents(e);
      if (c[0]) setSelectedId(c[0].id);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    fetchPredict(selectedId, 5, 0).then((r) => {
      setResult(r);
      setSimilarId(r.similar_cases[0]?.case_id ?? null);
      if (r.similarity_analysis.novel_or_low_similarity) {
        void postReviewFlag({
          case_id: selectedId,
          reason: "novel_or_low_similarity",
          note: "Auto-flagged from similarity analysis",
        });
      }
    });
  }, [selectedId]);

  const current = cases.find((c) => c.id === selectedId) || null;
  const selectedSimilar = result?.similar_cases.find((c: { case_id: string }) => c.case_id === similarId) || null;

  return (
    <div className="grid-2">
      <div className="panel fade-up">
        <h2>Clinical Review Mode</h2>
        <p className="muted">
          Select a lesion to retrieve visually similar historical cases. Research/educational use only.
        </p>
        <h3>Lesion events (sequences)</h3>
        {events.map((ev) => (
          <div key={ev.sequence_id} style={{ marginBottom: "0.85rem" }}>
            <div style={{ fontWeight: 700 }}>
              {ev.sequence_id} · {ev.label} · frames {ev.start_frame}–{ev.end_frame}
            </div>
            <div className="grid-3" style={{ marginTop: "0.45rem" }}>
              {ev.case_ids.map((id) => {
                const c = cases.find((x) => x.id === id);
                if (!c) return null;
                return (
                  <button
                    key={id}
                    type="button"
                    className={`case-card ${selectedId === id ? "selected" : ""}`}
                    onClick={() => setSelectedId(id)}
                    style={{ textAlign: "left", color: "inherit" }}
                  >
                    <img className="thumb" src={mediaUrl(`/media/images/${id}.png`)} alt={id} />
                    <div>{id}</div>
                    <div className="muted">
                      {c.gt_label} · f{c.frame_idx}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
        <div className="controls">
          <label style={{ flex: 1 }}>
            Review note
            <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Optional note" />
          </label>
          <button
            type="button"
            onClick={() => {
              if (!selectedId) return;
              void postReviewFlag({
                case_id: selectedId,
                reason: "disagreement",
                note: note || "Manual review",
              });
            }}
          >
            Flag for review
          </button>
        </div>
      </div>
      <div style={{ display: "grid", gap: "1rem" }}>
        <HybridPredictionView
          result={result}
          onSelectSimilar={(id) => setSimilarId(id)}
        />
        <CaseComparison current={current} result={result} selectedSimilar={selectedSimilar} />
      </div>
    </div>
  );
}
