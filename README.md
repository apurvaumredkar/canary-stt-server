# canary-stt-server

Self-hosted speech-to-text server built on NVIDIA **Canary-1B-v2** (int8 ONNX) with Silero VAD, served over HTTP and WebSocket with FastAPI.

> **Designed to run on the NVIDIA Jetson Orin Nano Super.** It is developed and tested on that device (aarch64, JetPack 6 / L4T r36). Inference runs on the CPU with int8 quantization, which leaves the Orin's GPU free for other workloads such as the companion [kokoro-tts-server](https://github.com/apurvaumredkar/kokoro-tts-server).

## Endpoints (port 8001)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, returns `{"status":"ok"}` |
| POST | `/transcribe` | One-shot. Body is raw PCM, response is `{"text": "..."}` |
| POST | `/stream` | Chunked raw PCM in, one JSON line (NDJSON) per detected utterance out |
| WS | `/ws` | Binary PCM frames in, one JSON text frame per detected utterance out |

All audio is **raw 16 kHz, mono, signed 16-bit little-endian PCM** with no WAV header.

Streaming endpoints use Silero VAD. An utterance is emitted after about 0.7 s of trailing silence.

## Models

Model weights are not stored in this repo. Place them under `models/`:

```
models/
├── canary-1b-v2/      # ONNX export of nvidia/canary-1b-v2, int8 (encoder/decoder-model.int8.onnx, vocab.txt, config.json)
└── silero-vad/        # silero_vad.onnx, config.json
```

The layout is the one `onnx-asr` expects for the `nemo-conformer-aed` model type.

## Run

```bash
docker compose up -d --build
curl localhost:8001/health
```

The `models/` directory is bind-mounted into the container at `/app/models`.

## Example

```bash
# convert any audio to the expected format
ffmpeg -i speech.wav -ar 16000 -ac 1 -f s16le speech.pcm

curl -X POST localhost:8001/transcribe --data-binary @speech.pcm
# {"text":"The quick brown fox jumps over the lazy dog."}
```

## Notes

- Requests are serialized by a lock around the model, so one transcription runs at a time.
- Canary-1B-v2 is multilingual. See the model card for the supported languages and license (CC-BY-4.0).
