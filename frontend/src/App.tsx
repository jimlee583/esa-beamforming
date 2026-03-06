import { useState, useCallback } from "react";
import Plot from "react-plotly.js";
import { computeWeights, computePattern, computeNullWeights } from "./api";
import type {
  WeightsRequest,
  WeightsResponse,
  PatternResponse,
  NullWeightsResponse,
  AzEl,
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
  phase_bits: null,
};

const PHASE_BITS_OPTIONS = [
  { value: "", label: "Continuous (off)" },
  { value: "1", label: "1-bit (2 levels)" },
  { value: "2", label: "2-bit (4 levels)" },
  { value: "3", label: "3-bit (8 levels)" },
  { value: "4", label: "4-bit (16 levels)" },
  { value: "5", label: "5-bit (32 levels)" },
  { value: "6", label: "6-bit (64 levels)" },
  { value: "8", label: "8-bit (256 levels)" },
];

function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [jammers, setJammers] = useState<AzEl[]>([]);
  const [weightsData, setWeightsData] = useState<WeightsResponse | null>(null);
  const [patternData, setPatternData] = useState<PatternResponse | null>(null);
  const [nullData, setNullData] = useState<NullWeightsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = useCallback(
    (key: keyof WeightsRequest, value: string | number) => {
      setForm((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const addJammer = useCallback(() => {
    setJammers((prev) => [...prev, { az_deg: 30, el_deg: 0 }]);
  }, []);

  const removeJammer = useCallback((idx: number) => {
    setJammers((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const updateJammer = useCallback(
    (idx: number, field: keyof AzEl, value: number) => {
      setJammers((prev) =>
        prev.map((j, i) => (i === idx ? { ...j, [field]: value } : j)),
      );
    },
    [],
  );

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
      setNullData(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handleNulling = useCallback(async () => {
    if (jammers.length === 0) {
      setError("Add at least one jammer direction.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [w, p, n] = await Promise.all([
        computeWeights(form),
        computePattern({
          ...form,
          az_range_deg: 90,
          el_range_deg: 90,
          n_points: 601,
        }),
        computeNullWeights({
          ...form,
          jammer_azels: jammers,
        }),
      ]);
      setWeightsData(w);
      setPatternData(p);
      setNullData(n);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [form, jammers]);

  const hasNullOverlay = nullData?.az_cut && nullData?.el_cut;
  const hasQuantOverlay =
    patternData?.quantized_az_cut && patternData?.quantized_el_cut;
  const hasQuantPhases = weightsData?.quantized_phases_rad;

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
              onChange={(e) =>
                set("freq_hz", parseFloat(e.target.value) * 1e9)
              }
            />
          </label>

          <label>
            Panel size (m)
            <input
              type="number"
              step="0.05"
              value={form.panel_size_m}
              onChange={(e) =>
                set("panel_size_m", parseFloat(e.target.value))
              }
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

          <h2>Phase Quantization</h2>

          <label>
            Phase-shifter bits
            <select
              value={form.phase_bits ?? ""}
              onChange={(e) =>
                set(
                  "phase_bits" as keyof WeightsRequest,
                  e.target.value === "" ? null! : parseInt(e.target.value, 10),
                )
              }
            >
              {PHASE_BITS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          {/* ── Jammers ── */}
          <h2>Jammer Nulls</h2>
          {jammers.map((j, i) => (
            <div className="jammer-row" key={i}>
              <input
                type="number"
                step="1"
                placeholder="az"
                value={j.az_deg}
                onChange={(e) =>
                  updateJammer(i, "az_deg", parseFloat(e.target.value))
                }
                title="Jammer azimuth (deg)"
              />
              <input
                type="number"
                step="1"
                placeholder="el"
                value={j.el_deg}
                onChange={(e) =>
                  updateJammer(i, "el_deg", parseFloat(e.target.value))
                }
                title="Jammer elevation (deg)"
              />
              <button
                className="remove-btn"
                onClick={() => removeJammer(i)}
                title="Remove jammer"
              >
                &times;
              </button>
            </div>
          ))}
          <button className="add-btn" onClick={addJammer}>
            + Add Jammer
          </button>

          <div className="btn-group">
            <button className="primary" onClick={handleBoth} disabled={loading}>
              Compute Both
            </button>
            <button
              className="primary nulling"
              onClick={handleNulling}
              disabled={loading || jammers.length === 0}
            >
              Compute Nulling
            </button>
          </div>

          {error && <p className="error">{error}</p>}

          {/* ── Metrics card ── */}
          {(weightsData || patternData || nullData) && (
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
                        <td>
                          {weightsData.spacing_lambda.toFixed(3)} &lambda;
                        </td>
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
                        <td>
                          {patternData.peak_gain_db.toFixed(1)} dB (norm.)
                        </td>
                      </tr>
                    </>
                  )}
                  {nullData && (
                    <tr>
                      <td>LCMV nulls</td>
                      <td>{nullData.constraint_residuals_re_im.length - 1}</td>
                    </tr>
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
              <h2>
                Element Phases{nullData ? " (conventional)" : ""}
              </h2>
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
                    name: "Conventional",
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

          {hasQuantPhases && weightsData && (
            <div className="plot-card">
              <h2>
                Element Phases (quantized, {weightsData.phase_bits}-bit)
              </h2>
              <Plot
                data={[
                  {
                    x: weightsData.positions.map((p) => p[0] * 1000),
                    y: weightsData.positions.map((p) => p[1] * 1000),
                    mode: "markers",
                    type: "scatter",
                    marker: {
                      color: weightsData.quantized_phases_rad!,
                      colorscale: "RdBu",
                      size: 6,
                      colorbar: { title: "Phase (rad)" },
                      reversescale: true,
                    },
                    name: "Quantized",
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

          {nullData && (
            <div className="plot-card">
              <h2>Element Phases (LCMV nulling)</h2>
              <Plot
                data={[
                  {
                    x: nullData.positions.map((p) => p[0] * 1000),
                    y: nullData.positions.map((p) => p[1] * 1000),
                    mode: "markers",
                    type: "scatter",
                    marker: {
                      color: nullData.phases_rad,
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
                <h2>Azimuth Cut</h2>
                <Plot
                  data={[
                    {
                      x: patternData.az_cut.angles_deg,
                      y: patternData.az_cut.gain_db,
                      type: "scatter",
                      mode: "lines",
                      line: { color: "#2563eb", width: 1.5 },
                      name: "Conventional",
                    },
                    ...(hasQuantOverlay
                      ? [
                          {
                            x: patternData.quantized_az_cut!.angles_deg,
                            y: patternData.quantized_az_cut!.gain_db,
                            type: "scatter" as const,
                            mode: "lines" as const,
                            line: {
                              color: "#10b981",
                              width: 1.5,
                              dash: "dash" as const,
                            },
                            name: `Quantized (${patternData.phase_bits}-bit)`,
                          },
                        ]
                      : []),
                    ...(hasNullOverlay
                      ? [
                          {
                            x: nullData.az_cut!.angles_deg,
                            y: nullData.az_cut!.gain_db,
                            type: "scatter" as const,
                            mode: "lines" as const,
                            line: { color: "#f59e0b", width: 2, dash: "dot" as const },
                            name: "LCMV nulling",
                          },
                        ]
                      : []),
                  ]}
                  layout={{
                    xaxis: { title: "Azimuth (deg)" },
                    yaxis: { title: "Gain (dB)", range: [-60, 1] },
                    margin: { t: 10, r: 20, b: 50, l: 60 },
                    height: 320,
                    showlegend: !!(hasNullOverlay || hasQuantOverlay),
                    legend: { x: 0.01, y: 0.99 },
                    shapes: jammers.map((j) => ({
                      type: "line" as const,
                      x0: j.az_deg,
                      x1: j.az_deg,
                      y0: -60,
                      y1: 1,
                      line: { color: "#ef4444", width: 1, dash: "dash" as const },
                    })),
                  }}
                  config={{ responsive: true }}
                  style={{ width: "100%" }}
                />
              </div>
              <div className="plot-card">
                <h2>Elevation Cut</h2>
                <Plot
                  data={[
                    {
                      x: patternData.el_cut.angles_deg,
                      y: patternData.el_cut.gain_db,
                      type: "scatter",
                      mode: "lines",
                      line: { color: "#dc2626", width: 1.5 },
                      name: "Conventional",
                    },
                    ...(hasQuantOverlay
                      ? [
                          {
                            x: patternData.quantized_el_cut!.angles_deg,
                            y: patternData.quantized_el_cut!.gain_db,
                            type: "scatter" as const,
                            mode: "lines" as const,
                            line: {
                              color: "#10b981",
                              width: 1.5,
                              dash: "dash" as const,
                            },
                            name: `Quantized (${patternData.phase_bits}-bit)`,
                          },
                        ]
                      : []),
                    ...(hasNullOverlay
                      ? [
                          {
                            x: nullData.el_cut!.angles_deg,
                            y: nullData.el_cut!.gain_db,
                            type: "scatter" as const,
                            mode: "lines" as const,
                            line: { color: "#f59e0b", width: 2, dash: "dot" as const },
                            name: "LCMV nulling",
                          },
                        ]
                      : []),
                  ]}
                  layout={{
                    xaxis: { title: "Elevation (deg)" },
                    yaxis: { title: "Gain (dB)", range: [-60, 1] },
                    margin: { t: 10, r: 20, b: 50, l: 60 },
                    height: 320,
                    showlegend: !!(hasNullOverlay || hasQuantOverlay),
                    legend: { x: 0.01, y: 0.99 },
                    shapes: jammers.map((j) => ({
                      type: "line" as const,
                      x0: j.el_deg,
                      x1: j.el_deg,
                      y0: -60,
                      y1: 1,
                      line: { color: "#ef4444", width: 1, dash: "dash" as const },
                    })),
                  }}
                  config={{ responsive: true }}
                  style={{ width: "100%" }}
                />
              </div>
            </>
          )}

          {!weightsData && !patternData && (
            <div className="placeholder">
              <p>
                Configure parameters and click{" "}
                <strong>Compute Both</strong> to see results.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
