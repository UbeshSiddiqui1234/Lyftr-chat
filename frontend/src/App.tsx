import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { ApiError, createSessionId, sendChat } from "./api";
import type { ChatMetadata } from "./api";

const MESSAGE_LIMIT = 4000;

type Item =
  | { id: number; kind: "user"; content: string }
  | { id: number; kind: "assistant"; content: string; metadata: ChatMetadata }
  | { id: number; kind: "error"; content: string; userId: number; text: string; editable: boolean };

function MetadataPanel({ metadata }: { metadata: ChatMetadata }) {
  const cached = metadata.resolution_source === "LLM_CACHE";
  return (
    <aside className="meta" aria-label="Response details">
      <span data-testid="meta-source" className={`source ${cached ? "source-cache" : "source-live"}`}>
        {metadata.resolution_source}
      </span>
      <dl>
        <div className="meta-wide">
          <dt>Model</dt>
          <dd data-testid="meta-model">{metadata.model}</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd data-testid="meta-latency">{metadata.latency_ms} ms</dd>
        </div>
        <div>
          <dt>Key</dt>
          <dd>{metadata.key_id ?? "none"}</dd>
        </div>
        <div>
          <dt>Input tokens</dt>
          <dd data-testid="meta-input-tokens">{metadata.input_tokens}</dd>
        </div>
        <div>
          <dt>Output tokens</dt>
          <dd data-testid="meta-output-tokens">{metadata.output_tokens}</dd>
        </div>
      </dl>
      {metadata.fallback_attempts > 0 && (
        <p className="failover">
          Failed over after {metadata.fallback_attempts} {metadata.fallback_attempts === 1 ? "key" : "keys"}
        </p>
      )}
    </aside>
  );
}

export default function App() {
  const [sessionId] = useState(createSessionId);
  const [items, setItems] = useState<Item[]>([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [pending, setPending] = useState(false);

  const nextId = useRef(1);
  const pendingRef = useRef(false);
  const appliedPrompt = useRef("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [items, pending]);

  const submit = async (text: string) => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    setPending(true);
    const userId = nextId.current++;
    setItems((current) => [...current, { id: userId, kind: "user", content: text }]);

    // The server remembers the prompt, so send it only when it has changed.
    const prompt = systemPrompt.trim();
    const promptChanged = prompt !== appliedPrompt.current;
    try {
      const response = await sendChat({
        session_id: sessionId,
        message: text,
        ...(promptChanged ? { system_prompt: prompt } : {}),
      });
      appliedPrompt.current = prompt;
      const id = nextId.current++;
      setItems((current) => [
        ...current,
        { id, kind: "assistant", content: response.reply, metadata: response.metadata },
      ]);
    } catch (error) {
      const apiError = error instanceof ApiError ? error : null;
      const id = nextId.current++;
      setItems((current) => [
        ...current,
        {
          id,
          kind: "error",
          content: apiError?.message ?? "Something went wrong. Please try again.",
          userId,
          text,
          editable: apiError?.status === 422,
        },
      ]);
    } finally {
      pendingRef.current = false;
      setPending(false);
      inputRef.current?.focus();
    }
  };

  const send = () => {
    const text = input.trim();
    if (!text || pendingRef.current) return;
    setInput("");
    void submit(text);
  };

  const removeFailedTurn = (item: Extract<Item, { kind: "error" }>) => {
    setItems((current) => current.filter((i) => i.id !== item.id && i.id !== item.userId));
  };

  const retry = (item: Extract<Item, { kind: "error" }>) => {
    if (pendingRef.current) return;
    removeFailedTurn(item);
    void submit(item.text);
  };

  const edit = (item: Extract<Item, { kind: "error" }>) => {
    removeFailedTurn(item);
    setInput(item.text);
    inputRef.current?.focus();
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    send();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  };

  const overLimit = input.trim().length > MESSAGE_LIMIT;

  return (
    <div className="app">
      <header className="sidebar">
        <div>
          <h1>Gateway Chat</h1>
          <p className="session" title={sessionId}>
            Session <code>{sessionId.slice(0, 12)}</code>
          </p>
        </div>
        <label className="prompt-field">
          <span>System prompt</span>
          <textarea
            data-testid="system-prompt-input"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            placeholder="Optional. Applies from your next message."
            rows={4}
          />
        </label>
      </header>

      <main className="chat">
        <div className="transcript" role="log" aria-live="polite" aria-label="Conversation">
          <div className="transcript-inner">
            {items.length === 0 && !pending && (
              <div className="empty">
                <h2>Start a conversation</h2>
                <p>Each reply shows where it came from, how long it took and the tokens it used.</p>
              </div>
            )}

            {items.map((item) => {
              if (item.kind === "user") {
                return (
                  <div key={item.id} className="user-row">
                    <div className="user-bubble">{item.content}</div>
                  </div>
                );
              }
              if (item.kind === "assistant") {
                return (
                  <article key={item.id} className="assistant" data-testid="assistant-message">
                    <div className="reply">{item.content}</div>
                    <MetadataPanel metadata={item.metadata} />
                  </article>
                );
              }
              return (
                <div key={item.id} className="error" role="alert" data-testid="chat-error">
                  <span>{item.content}</span>
                  <button type="button" onClick={() => (item.editable ? edit(item) : retry(item))} disabled={pending}>
                    {item.editable ? "Edit message" : "Try again"}
                  </button>
                </div>
              );
            })}

            {pending && (
              <div className="thinking" role="status">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
                <span className="sr-only">Waiting for a reply</span>
              </div>
            )}
            <div ref={endRef} />
          </div>
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <div className="composer-inner">
            <textarea
              ref={inputRef}
              data-testid="chat-input"
              aria-label="Message"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Send a message"
              rows={2}
              autoFocus
            />
            <div className="composer-actions">
              <span className={`hint ${overLimit ? "hint-over" : ""}`}>
                {input.length > MESSAGE_LIMIT - 500
                  ? `${input.trim().length.toLocaleString()} / ${MESSAGE_LIMIT.toLocaleString()}`
                  : "Enter to send, Shift+Enter for a new line"}
              </span>
              <button type="submit" data-testid="send-button" disabled={pending || !input.trim()}>
                Send
              </button>
            </div>
          </div>
        </form>
      </main>
    </div>
  );
}
