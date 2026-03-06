/** Mirrors backend Pydantic models. */

export type LatticeType = "rectangular" | "triangular";
export type TaperType = "uniform" | "taylor" | "chebyshev";

export interface WeightsRequest {
  freq_hz: number;
  panel_size_m: number;
  lattice: LatticeType;
  spacing_m?: number | null;
  element_k_lambda: number;
  steer_az_deg: number;
  steer_el_deg: number;
  taper: TaperType;
  phase_bits?: number | null;
}

export interface PatternRequest extends WeightsRequest {
  az_range_deg: number;
  el_range_deg: number;
  n_points: number;
}

export interface WeightsResponse {
  n_elements: number;
  positions: number[][];
  phases_rad: number[];
  weights_re_im: number[][];
  spacing_m: number;
  spacing_lambda: number;
  phase_bits?: number | null;
  quantized_phases_rad?: number[] | null;
  quantized_weights_re_im?: number[][] | null;
}

export interface PatternCut {
  angles_deg: number[];
  gain_db: number[];
  label: string;
}

export interface PatternResponse {
  az_cut: PatternCut;
  el_cut: PatternCut;
  peak_az_deg: number;
  peak_el_deg: number;
  peak_gain_db: number;
  n_elements: number;
  spacing_lambda: number;
  phase_bits?: number | null;
  quantized_az_cut?: PatternCut | null;
  quantized_el_cut?: PatternCut | null;
}

export interface AzEl {
  az_deg: number;
  el_deg: number;
}

export interface NullWeightsRequest {
  freq_hz: number;
  panel_size_m: number;
  lattice: LatticeType;
  spacing_m?: number | null;
  element_k_lambda: number;
  steer_az_deg: number;
  steer_el_deg: number;
  taper: TaperType;
  jammer_azels: AzEl[];
  diag_load?: number;
}

export interface NullWeightsResponse {
  n_elements: number;
  positions: number[][];
  phases_rad: number[];
  weights_re_im: number[][];
  spacing_m: number;
  spacing_lambda: number;
  constraint_residuals_re_im: number[][];
  az_cut: PatternCut | null;
  el_cut: PatternCut | null;
}

export interface NullDepthVsBitsRequest {
  freq_hz: number;
  panel_size_m: number;
  lattice: LatticeType;
  spacing_m?: number | null;
  element_k_lambda: number;
  steer_az_deg: number;
  steer_el_deg: number;
  taper: TaperType;
  jammer_azels: AzEl[];
  diag_load?: number;
  bit_settings: number[];
  include_continuous: boolean;
}

export interface BitSettingResult {
  label: string;
  bits: number | null;
  desired_gain_mag: number;
  jammer_response_mag: number[];
  null_depth_db: number[];
  worst_null_depth_db: number;
}

export interface NullDepthSummary {
  continuous_worst_null_db: number;
  best_quantized_worst_null_db: number | null;
  best_quantized_label: string | null;
}

export interface NullDepthVsBitsResponse {
  results: BitSettingResult[];
  jammer_labels: string[];
  summary: NullDepthSummary;
}
