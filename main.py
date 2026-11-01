import json
import threading

import numpy as np
import onnx_asr
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

app = FastAPI()
MODEL_DIR = "models/canary-1b-v2"
SAMPLE_RATE = 16000
EVAL_EVERY_S = 0.5
END_SILENCE_S = 0.7
MAX_IDLE_S = 3.0

VAD_DIR = "models/silero-vad"

providers = ["CPUExecutionProvider"]
asr = onnx_asr.load_model(
    "nemo-conformer-aed", MODEL_DIR, quantization="int8", providers=providers
)
vad = onnx_asr.load_vad("silero", VAD_DIR, providers=providers)
lock = threading.Lock()
print("STT ready: canary-1b-v2 int8 on CPU", flush=True)


def transcribe(audio: np.ndarray) -> str:
    with lock:
        return str(asr.recognize(audio, sample_rate=SAMPLE_RATE)).strip()


def segments(audio: np.ndarray) -> list[tuple[int, int]]:
    with lock:
        return list(
            next(
                iter(
                    vad.segment_batch(
                        audio[None, :],
                        np.array([len(audio)], dtype=np.int64),
                        SAMPLE_RATE,
                    )
                )
            )
        )


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_pcm(request: Request):
    """One-shot: raw 16 kHz mono s16le PCM body in, transcript out."""
    audio = pcm16_to_float(await request.body())
    return {"text": transcribe(audio) if len(audio) else ""}


@app.post("/stream")
async def stream(request: Request):
    """Chunked raw 16 kHz mono s16le PCM in; one JSON line per utterance out."""

    async def results():
        buf = np.zeros(0, dtype=np.float32)
        carry = b""
        since_eval = 0
        async for chunk in request.stream():
            chunk = carry + chunk
            carry = chunk[len(chunk) // 2 * 2 :]
            samples = pcm16_to_float(chunk[: len(chunk) // 2 * 2])
            buf = np.concatenate([buf, samples])
            since_eval += len(samples)
            if since_eval < EVAL_EVERY_S * SAMPLE_RATE:
                continue
            since_eval = 0
            segs = segments(buf)
            if not segs:
                if len(buf) > MAX_IDLE_S * SAMPLE_RATE:
                    buf = buf[-SAMPLE_RATE:]
                continue
            end = segs[-1][1]
            if len(buf) - end < END_SILENCE_S * SAMPLE_RATE:
                continue
            text = transcribe(buf[segs[0][0] : end])
            buf = buf[end:]
            if text:
                yield json.dumps({"text": text}) + "\n"

    return StreamingResponse(results(), media_type="application/x-ndjson")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Binary frames of raw 16 kHz mono s16le PCM in; one JSON text frame per utterance out."""
    await websocket.accept()
    buf = np.zeros(0, dtype=np.float32)
    carry = b""
    since_eval = 0
    try:
        while True:
            chunk = carry + await websocket.receive_bytes()
            carry = chunk[len(chunk) // 2 * 2 :]
            samples = pcm16_to_float(chunk[: len(chunk) // 2 * 2])
            buf = np.concatenate([buf, samples])
            since_eval += len(samples)
            if since_eval < EVAL_EVERY_S * SAMPLE_RATE:
                continue
            since_eval = 0
            segs = segments(buf)
            if not segs:
                if len(buf) > MAX_IDLE_S * SAMPLE_RATE:
                    buf = buf[-SAMPLE_RATE:]
                continue
            end = segs[-1][1]
            if len(buf) - end < END_SILENCE_S * SAMPLE_RATE:
                continue
            text = transcribe(buf[segs[0][0] : end])
            buf = buf[end:]
            if text:
                await websocket.send_text(json.dumps({"text": text}))
    except WebSocketDisconnect:
        pass
