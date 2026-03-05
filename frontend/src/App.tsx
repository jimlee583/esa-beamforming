import { useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { computeWeights, computePattern } from "./api";
import type {
  WeightsRequest,
  WeightsResponse,
  PatternRequest,
  PatternResponse,
  LatticeType,
} from "./types";
import "./App.css";

const DEFAULTS: WeightsRequest = {
  freq_hz: 10e9,
  panel_size_m: 0.3,
  lattice: "rectangular",
  element_k_lambda: 0.5,
  steer_az_deg: 0,
  steer_el_deg: 0,
  taper: "uniform",
};

function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [weightsData, setWeightsData] = useState<WeightsResponse | null>(null);
  const [patternData, setPatternData] = useState<PatternResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    (key: keyof WeightsRequest, value: string | number) => {
      setForm((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleWeights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await computeWeights(form);
      setWeightsData(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handlePattern = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const req: PatternRequest = {
        ...form,
        az_range_deg: 90,
        el_range_deg: 90,
        n_points: 601,
      };
      const data = await computePattern(req);
      setPatternData(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handleBoth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [w, p] = await Promise.all([
        computeWeights(form),
        computePattern({
          ...form,
          az_range_deg: 90,
          el_range_deg: 90,
          n_points: 601,
        }),
      ]);
      setWeightsData(w);
      setPatternData(p);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [form]);

  return (
    <div className="app">
      <header>
        <h1>ESA Beamforming Explorer</h1>
      </header>

      <div className="layout">
        {/* ── Controls ── */}
        <section className="controls">
          <h2>Array Parameters</h2>

          <label>
            Frequency (GHz)
            <input
              type="number"
              step="0.1"
              value={form.freq_hz / 1e9}
              onChange={(e) => set("freq_hz", parseFloat(e.target.value) * 1e9)}
            />
          </label>

          <label>
            Panel size (m)
            <input
              type="number"
              step="0.05"
              value={form.panel_size_m}
              onChange={(e) => set("panel_size_m", parseFloat(e.target.value))}
            />
          </label>

          <label>
            Lattice
            <select
              value={form.lattice}
              onChange={(e) => set("lattice", e.target.value as LatticeType)}
            >
              <option value="rectangular">Rectangular</option>
              <option value="triangular">Triangular</option>
            </select>
          </label>

          <label>
            Spacing (k&middot;&lambda;)
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="1.0"
              value={form.element_k_lambda}
              onChange={(e) =>
                set("element_k_lambda", parseFloat(e.target.value))
              }
            />
          </label>

          <h2>Steering</h2>

          <label>
            Azimuth (&deg;)
            <input
              type="number"
              step="1"
              value={form.steer_az_deg}
              onChange={(e) =>
                set("steer_az_deg", parseFloat(e.target.value))
              }
            />
          </label>

          <label>
            Elevation (&deg;)
            <input
              type="number"
              step="1"
              value={form.steer_el_deg}
              onChange={(e) =>
                set("steer_el_deg", parseFloat(e.target.value))
              }
            />
          </label>

          <div className="btn-group">
            <button onClick={handleWeights} disabled={loading}>
              Compute Weights
            </button>
            <button onClick={handlePattern} disabled={loading}>
              Compute Pattern
            </button>
            <button className="primary" onClick={handleBoth} disabled={loading}>
              Compute Both
            </button>
          </div>

          {error && <p className="error">{error}</p>}

          {/* ── Metrics card ── */}
          {(weightsData || patternData) && (
            <div className="metrics">
              <h2>Metrics</h2>
              <table>
                <tbody>
                  {weightsData && (
                    <>
                      <tr>
                        <td>Elements</td>
                        <td>{weightsData.n_elements}</td>
                      </tr>
                      <tr>
                        <td>Spacing</td>
                        <td>{weightsData.spacing_lambda.toFixed(3)} &lambda;</td>
                      </tr>
                    </>
                  )}
                  {patternData && (
                    <>
                      <tr>
                        <td>Peak direction</td>
                        <td>
                          az={patternData.peak_az_deg.toFixed(1)}&deg;, el=
                          {patternData.peak_el_deg.toFixed(1)}&deg;
                        </td>
                      </tr>
                      <tr>
                        <td>Peak gain</td>
                        <td>{patternData.peak_gain_db.toFixed(1)} dB (norm.)</td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* ── Plots ── */}
        <section className="plots">
          {weightsData && (
            <div className="plot-card">
              <h2>Element Phases</h2>
              <Plot
                data={[
                  {
                    x: weightsData.positions.map((p) => p[0] * 1000),
                    y: weightsData.positions.map((p) => p[1] * 1000),
                    mode: "markers",
                    type: "scatter",
                    marker: {
                      color: weightsData.phases_rad,
                      colorscale: "RdBu",
                      size: 6,
                      colorbar: { title: "Phase (rad)" },
                      reversescale: true,
                    },
                  },
                ]}
                layout={{
                  xaxis: { title: "x (mm)", scaleanchor: "y" },
                  yaxis: { title: "y (mm)" },
                  margin: { t: 10, r: 30, b: 50, l: 60 },
                  height: 380,
                }}
                config={{ responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {patternData && (
            <>
              <div className="plot-card">
                <h2>{patternData.az_cut.label}</h2>
                <Plot
                  data={[
                    {
                      x: patternData.az_cut.angles_deg,
                      y: patternData.az_cut.gain_db,
                      type: "scatter",
                      mode: "lines",
                      line: { color: "#2563eb", width: 1.5 },
                    },
                  ]}
                  layout={{
                    xaxis: { title: "Azimuth (°)" },
                    yaxis: { title: "Gain (dB)", range: [-50, 1] },
                    margin: { t: 10, r: 20, b: 50, l: 60 },
                    height: 300,
                  }}
                  config={{ responsive: true }}
                  style={{ width: "100%" }}
                />
              </div>
              <div className="plot-card">
                <h2>{patternData.el_cut.label}</h2>
                <Plot
                  data={[
                    {
                      x: patternData.el_cut.angles_deg,
                      y: patternData.el_cut.gain_db,
                      type: "scatter",
                      mode: "lines",
                      line: { color: "#dc2626", width: 1.5 },
                    },
                  ]}
                  layout={{
                    xaxis: { title: "Elevation (°)" },
                    yaxis: { title: "Gain (dB)", range: [-50, 1] },
                    margin: { t: 10, r: 20, b: 50, l: 60 },
                    height: 300,
                  }}
                  config={{ responsive: true }}
                  style={{ width: "100%" }}
                />
              </div>
            </>
          )}

          {!weightsData && !patternData && (
            <div className="placeholder">
              <p>Configure parameters and click <strong>Compute Both</strong> to see results.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
