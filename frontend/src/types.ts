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
}
