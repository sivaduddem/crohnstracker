import { DEMO_BANNER, DISCLAIMER } from "@crohnstracker/shared";
import type { HealthResponse } from "@crohnstracker/shared";

export function DemoBanner({ health }: { health: HealthResponse | null }) {
  const isDemo = !health || health.mode === "demo";
  return (
    <div className="banner" role="status">
      <div>
        <strong>{isDemo ? health?.demo_banner || DEMO_BANNER : "Trained model mode"}</strong>
        <div className="disclaimer">{DISCLAIMER}</div>
      </div>
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
        <span className={`pill ${health?.status === "ok" ? "ok" : "warn"}`}>
          API: {health?.status ?? "…"}
        </span>
        <span className={`pill ${isDemo ? "mock" : "ok"}`}>mode: {health?.mode ?? "…"}</span>
        {isDemo && <span className="pill mock">is_mock</span>}
      </div>
    </div>
  );
}
