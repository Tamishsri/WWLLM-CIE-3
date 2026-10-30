// Thin client for the Studio API, including the NDJSON event streams.

export class StreamError extends Error {
  constructor(message, hint = "") {
    super(message);
    this.hint = hint;
  }
}

async function errorFrom(res) {
  try {
    const body = await res.json();
    return new StreamError(body.error || `Request failed (${res.status})`);
  } catch {
    return new StreamError(`Request failed (${res.status} ${res.statusText})`);
  }
}

export async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

export async function postJSON(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

export async function postForm(url, form) {
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

/** Upload with progress (fetch can't report upload progress, XHR can). */
export function uploadFile(file, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.open("POST", "/api/files");
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new StreamError(xhr.response?.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new StreamError("Upload failed — is the server running?"));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(form);
  });
}

/**
 * POST *body* to a streaming endpoint and feed each NDJSON event to *onEvent*.
 * Resolves with the ``result`` payload; rejects with StreamError on an
 * ``error`` event (after passing it to onEvent too).
 */
export async function stream(path, body, onEvent, signal) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw await errorFrom(res);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  const handle = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    onEvent?.(event);
    if (event.event === "error") {
      const err = new StreamError(event.message, event.hint);
      err.fromStream = true; // onEvent has already seen (and logged) it
      throw err;
    }
    if (event.event === "result") result = event.data;
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      handle(line);
    }
  }
  handle(buffer);
  if (!result) throw new StreamError("The server closed the stream without a result.");
  return result;
}
