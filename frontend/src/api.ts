const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");
const REQUEST_TIMEOUT_MS = 120_000;

export interface ChatMetadata {
  resolution_source: "LIVE_INFERENCE" | "LLM_CACHE";
  model: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  key_id: string | null;
  fallback_attempts: number;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  system_prompt?: string;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ChatMetadata;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export function createSessionId(): string {
  // randomUUID is unavailable on plain-http origins other than localhost.
  if (typeof crypto.randomUUID === "function") return `s-${crypto.randomUUID()}`;
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return `s-${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

function validationMessage(detail: unknown): string {
  const first = Array.isArray(detail) ? detail[0] : undefined;
  const field = Array.isArray(first?.loc) ? first.loc.at(-1) : undefined;
  if (field === "message") {
    return first.type === "string_too_long"
      ? "That message is too long. The limit is 4,000 characters."
      : "Please enter a message before sending.";
  }
  if (field === "system_prompt") {
    return "The system prompt is too long. The limit is 8,000 characters.";
  }
  return "That request wasn't valid. Please check it and try again.";
}

function errorMessage(status: number, detail: unknown): string {
  if (status === 422) return validationMessage(detail);
  if (status === 503) {
    return "All AI providers are busy or unavailable right now. Please try again in a moment.";
  }
  return "Something went wrong on the server. Please try again.";
}

function isChatResponse(data: unknown): data is ChatResponse {
  const candidate = data as ChatResponse | null;
  return typeof candidate?.reply === "string" && typeof candidate?.metadata === "object";
}

export async function sendChat(body: ChatRequest): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    throw new ApiError("Can't reach the server. Check your connection and try again.", 0);
  }
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    throw new ApiError(errorMessage(response.status, detail), response.status);
  }
  if (!isChatResponse(data)) {
    throw new ApiError("The server sent an unexpected response. Please try again.", response.status);
  }
  return data;
}
