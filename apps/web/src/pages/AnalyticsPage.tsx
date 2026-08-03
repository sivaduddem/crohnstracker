import { useEffect, useState } from "react";
import type { MetricsSummary } from "@crohnstracker/shared";
import { fetchMetrics } from "../api/client";
import { AnalyticsDashboard } from "../components/AnalyticsDashboard";

export function AnalyticsPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  useEffect(() => {
    fetchMetrics().then(setMetrics).catch(console.error);
  }, []);
  return <AnalyticsDashboard metrics={metrics} />;
}
