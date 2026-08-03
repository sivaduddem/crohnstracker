import type { HealthResponse } from "@crohnstracker/shared";

export function HealthPage({ health }: { health: HealthResponse | null }) {
  return (
    <div className="panel fade-up">
      <h2>System health</h2>
      {!health ? (
        <p className="muted">Checking API…</p>
      ) : (
        <ul>
          <li>Status: {health.status}</li>
          <li>Mode: {health.mode}</li>
          <li>Model loaded: {String(health.model_loaded)}</li>
          <li>FAISS loaded: {String(health.faiss_loaded)}</li>
          <li>Version: {health.version}</li>
          {health.demo_banner && <li>Banner: {health.demo_banner}</li>}
        </ul>
      )}
    </div>
  );
}
