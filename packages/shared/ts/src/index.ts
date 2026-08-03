import { z } from "zod";

export const DISCLAIMER =
  "Research and educational use only. Similarity does not imply the same diagnosis or clinical outcome. Mock predictions are never clinical results.";

export const DEMO_BANNER = "Demo data - mock predictions";

export const TARGET_LABELS = ["normal", "erythema", "erosion", "ulcer"] as const;
export const LABELS = [...TARGET_LABELS, "other_unmapped"] as const;

export const LesionLabelSchema = z.enum(LABELS);
export type LesionLabel = z.infer<typeof LesionLabelSchema>;

export const AppModeSchema = z.enum(["demo", "trained"]);
export type AppMode = z.infer<typeof AppModeSchema>;

export const PredictionResultSchema = z.object({
  label: LesionLabelSchema,
  confidence: z.number().min(0).max(1),
  probs: z.record(z.number()),
});

export const SimilarCaseSchema = z.object({
  case_id: z.string(),
  similarity: z.number(),
  pred_label: LesionLabelSchema,
  gt_label: LesionLabelSchema,
  confidence: z.number().min(0).max(1),
  thumbnail_url: z.string(),
  gradcam_url: z.string().nullable().optional(),
  severity: z.string().nullable().optional(),
});

export const SimilarityAnalysisSchema = z.object({
  nn_distance: z.number(),
  avg_topk_distance: z.number(),
  cluster_id: z.number().nullable().optional(),
  similarity_confidence: z.number().min(0).max(1),
  novel_or_low_similarity: z.boolean(),
});

export const HybridPredictResponseSchema = z.object({
  mode: AppModeSchema,
  disclaimer: z.string(),
  case_id: z.string().nullable().optional(),
  prediction: PredictionResultSchema,
  similarity_analysis: SimilarityAnalysisSchema,
  similar_cases: z.array(SimilarCaseSchema),
  is_mock: z.boolean(),
});

export type HybridPredictResponse = z.infer<typeof HybridPredictResponseSchema>;
export type SimilarCase = z.infer<typeof SimilarCaseSchema>;
export type PredictionResult = z.infer<typeof PredictionResultSchema>;
export type SimilarityAnalysis = z.infer<typeof SimilarityAnalysisSchema>;

export const HealthResponseSchema = z.object({
  status: z.enum(["ok", "degraded"]),
  mode: AppModeSchema,
  model_loaded: z.boolean(),
  faiss_loaded: z.boolean(),
  demo_banner: z.string().nullable().optional(),
  version: z.string(),
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export const EmbeddingPointSchema = z.object({
  case_id: z.string(),
  x: z.number(),
  y: z.number(),
  gt_label: LesionLabelSchema,
  pred_label: LesionLabelSchema,
  confidence: z.number(),
  severity: z.string().nullable().optional(),
  thumbnail_url: z.string(),
});

export const MetricsSummarySchema = z.object({
  mode: AppModeSchema,
  disclaimer: z.string(),
  classification: z.object({
    accuracy: z.number(),
    macro_f1: z.number(),
    per_class_f1: z.record(z.number()),
    confusion_matrix: z.array(z.array(z.number())),
    labels: z.array(z.string()),
  }),
  retrieval: z.object({
    recall_at_k: z.number(),
    precision_at_k: z.number(),
    map: z.number(),
    ndcg: z.number(),
    latency_ms: z.number(),
    embedding_ms: z.number().nullable().optional(),
    k: z.number(),
  }),
});

export type MetricsSummary = z.infer<typeof MetricsSummarySchema>;
