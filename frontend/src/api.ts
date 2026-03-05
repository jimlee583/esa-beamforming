import type { WeightsRequest, WeightsResponse, PatternRequest, PatternResponse } from "./types";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function computeWeights(req: WeightsRequest): Promise<WeightsResponse> {
  return post<WeightsResponse>("/weights", req);
}

export function computePattern(req: PatternRequest): Promise<PatternResponse> {
  return post<PatternResponse>("/pattern", req);
}
