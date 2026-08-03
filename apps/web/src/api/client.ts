import {
  HealthResponseSchema,
  HybridPredictResponseSchema,
  MetricsSummarySchema,
  type HealthResponse,
  type HybridPredictResponse,
  type MetricsSummary,
} from "@crohnstracker/shared";

const API_BASE = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

async function getJson<T>(path: string, parse: (data: unknown) => T): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return parse(await res.json());
}

export function mediaUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return getJson("/health", (d) => HealthResponseSchema.parse(d));
}

export async function fetchPredict(caseId?: string, topK = 5, threshold = 0): Promise<HybridPredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: caseId ?? null,
      top_k: topK,
      similarity_threshold: threshold,
      metric: "cosine",
    }),
  });
  if (!res.ok) throw new Error(`predict failed: ${res.status}`);
  return HybridPredictResponseSchema.parse(await res.json());
}

export async function fetchCases(): Promise<
  Array<{
    id: string;
    gt_label: string;
    pred_label?: string | null;
    confidence?: number | null;
    severity?: string | null;
    path: string;
    video_id: string;
    frame_idx: number;
  }>
> {
  const res = await fetch(`${API_BASE}/cases`);
  if (!res.ok) throw new Error("cases failed");
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsSummary> {
  return getJson("/metrics/summary", (d) => MetricsSummarySchema.parse(d));
}

export async function fetchProjection(method: "umap" | "tsne" | "pca" = "umap") {
  const res = await fetch(`${API_BASE}/embeddings/projection?method=${method}`);
  if (!res.ok) throw new Error("projection failed");
  return res.json();
}

export async function fetchSequences() {
  const res = await fetch(`${API_BASE}/sequences`);
  if (!res.ok) throw new Error("sequences failed");
  return res.json();
}

export async function postReviewFlag(payload: {
  case_id: string;
  reason: "novel_or_low_similarity" | "low_confidence" | "disagreement";
  status?: "open" | "resolved";
  note?: string;
}) {
  const res = await fetch(`${API_BASE}/review/flags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: "open", ...payload }),
  });
  if (!res.ok) throw new Error("flag failed");
  return res.json();
}
