import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import type { HealthResponse } from "@crohnstracker/shared";
import { fetchHealth } from "./api/client";
import { DemoBanner } from "./components/DemoBanner";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { ExplorerPage } from "./pages/ExplorerPage";
import { HealthPage } from "./pages/HealthPage";
import { HomePage } from "./pages/HomePage";
import { ReviewPage } from "./pages/ReviewPage";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({
          status: "degraded",
          mode: "demo",
          model_loaded: false,
          faiss_loaded: false,
          demo_banner: "Demo data — mock predictions",
          version: "0.1.0",
        }),
      );
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">CrohnTracker</div>
        <nav className="nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : undefined)}>
            Hybrid
          </NavLink>
          <NavLink to="/review" className={({ isActive }) => (isActive ? "active" : undefined)}>
            Clinical Review
          </NavLink>
          <NavLink to="/explorer" className={({ isActive }) => (isActive ? "active" : undefined)}>
            Embeddings
          </NavLink>
          <NavLink to="/analytics" className={({ isActive }) => (isActive ? "active" : undefined)}>
            Analytics
          </NavLink>
          <NavLink to="/health" className={({ isActive }) => (isActive ? "active" : undefined)}>
            Health
          </NavLink>
        </nav>
      </header>
      <DemoBanner health={health} />
      <Routes>
        <Route path="/" element={<HomePage health={health} />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/explorer" element={<ExplorerPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/health" element={<HealthPage health={health} />} />
      </Routes>
    </div>
  );
}
