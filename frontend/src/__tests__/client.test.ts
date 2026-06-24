import { describe, expect, it, vi } from "vitest";
import { _internals, type SSEHandlers } from "../api/client";

const { parseSSEChunk, dispatchSSE: dispatch } = _internals;

describe("SSE parser", () => {
  it("parses complete events and returns the trailing partial as rest", () => {
    const buf =
      "event: token\ndata: {\"text\": \"hello \"}\n\n" +
      "event: token\ndata: {\"text\": \"world\"}\n\n" +
      "event: done\ndata: {}\n\n" +
      "event: token\ndata: {\"text\": \"partial";
    const { events, rest } = parseSSEChunk(buf);
    expect(events).toHaveLength(3);
    expect(events[0]).toEqual({ event: "token", data: '{"text": "hello "}' });
    expect(events[2].event).toBe("done");
    // The incomplete final block is preserved for the next chunk.
    expect(rest).toContain("partial");
  });

  it("dispatches token/meta/error/done to the right handlers", () => {
    const onToken = vi.fn();
    const onMeta = vi.fn();
    const onError = vi.fn();
    const onDone = vi.fn();
    const handlers: SSEHandlers = { onToken, onMeta, onError, onDone };

    dispatch({ event: "token", data: '{"text":"hi"}' }, handlers);
    dispatch({ event: "meta", data: '{"references":[{"path":"a.py"}]}' }, handlers);
    dispatch({ event: "error", data: '{"message":"boom"}' }, handlers);
    dispatch({ event: "done", data: "{}" }, handlers);

    expect(onToken).toHaveBeenCalledWith("hi");
    expect(onMeta).toHaveBeenCalledWith({ references: [{ path: "a.py" }] });
    expect(onError).toHaveBeenCalledWith("boom");
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it("tolerates malformed JSON in a data line without throwing", () => {
    const onToken = vi.fn();
    expect(() =>
      dispatch({ event: "token", data: "not json" }, { onToken }),
    ).not.toThrow();
    expect(onToken).toHaveBeenCalledWith("");
  });

  it("dispatches heartbeat events", () => {
    const onHeartbeat = vi.fn();
    dispatch({ event: "heartbeat", data: '{"t":123}' }, { onHeartbeat });
    expect(onHeartbeat).toHaveBeenCalledTimes(1);
  });
});
