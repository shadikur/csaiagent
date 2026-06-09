"""
StyleTTS2 TTS Server — OpenAI-compatible /v1/audio/speech endpoint
Drop-in replacement for kokoro-fastapi-gpu on port 8880.

StyleTTS2 advantages over Kokoro:
  - Dramatically more natural prosody and expressiveness
  - Adaptive style diffusion (5 steps = fast, 10 steps = higher quality)
  - RTF ~0.1-0.15 on T4 — nearly as fast as Kokoro
  - No reference audio needed for default voice (LJSpeech trained)
"""
import io
import re
import os
import struct
import logging
import asyncio
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [styletts2-tts] %(levelname)s %(message)s"
)
logger = logging.getLogger("styletts2_tts")

SAMPLE_RATE = 24000   # StyleTTS2 LJSpeech model output rate

# Pre-downloaded LJSpeech reference voice — avoids network fetch on every inference call
# Source: https://styletts2.github.io/wavs/LJSpeech/OOD/GT/00001.wav
REFERENCE_VOICE_PATH = "/home/compusource/voice-agent/styletts2_reference_voice.wav"

# ── Model globals ──────────────────────────────────────────────────────────────
_tts = None
_device = None
_model_lock = asyncio.Lock()

def _load_model():
    global _tts, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading StyleTTS2 on device: {_device}")

    # PyTorch 2.6 changed torch.load default to weights_only=True which breaks
    # legacy StyleTTS2 checkpoints that use getattr globals. Patch it scoped here.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})
    try:
        from styletts2.tts import StyleTTS2
        # Loads LJSpeech model automatically from HuggingFace cache (~300MB)
        _tts = StyleTTS2()
    finally:
        torch.load = _orig_torch_load  # always restore, even on error

    logger.info("✅ StyleTTS2 model loaded and ready.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)
    yield
    logger.info("StyleTTS2 TTS server shutting down.")

app = FastAPI(title="StyleTTS2 TTS Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sentence splitter ──────────────────────────────────────────────────────────
def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    raw = re.split(r'(?<=[.!?,;])\s+', text.strip())
    chunks, buf = [], ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= min_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return chunks or [text.strip()]

# ── WAV helpers ────────────────────────────────────────────────────────────────
def _make_wav_header(num_samples: int, sample_rate: int = SAMPLE_RATE,
                     num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    byte_rate     = sample_rate * num_channels * bits_per_sample // 8
    block_align   = num_channels * bits_per_sample // 8
    data_size     = num_samples * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1,
        num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )

def _float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()

# ── Synthesis (runs in executor thread) ───────────────────────────────────────
def _synthesize_chunk(text: str, diffusion_steps: int, embedding_scale: float) -> bytes:
    """Synthesize one text chunk; returns raw 16-bit PCM bytes."""
    audio = _tts.inference(
        text,
        target_voice_path=REFERENCE_VOICE_PATH,  # local cache — no network fetch
        diffusion_steps=diffusion_steps,
        embedding_scale=embedding_scale,
        alpha=0.3,    # timbre style weight
        beta=0.7,     # prosody style weight
    )
    # audio is numpy float32 array
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().numpy()
    audio = np.squeeze(np.array(audio, dtype=np.float32))
    return _float_to_pcm16(audio)

# ── OpenAI-compatible speech endpoint ─────────────────────────────────────────
@app.post("/v1/audio/speech")
async def text_to_speech(request: Request):
    try:
        req_json = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    text = req_json.get("input", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Missing 'input' field"})

    voice = req_json.get("voice", "default").lower()

    # Style tuning via voice name hints
    if "expressive" in voice or "emotional" in voice:
        diffusion_steps, embedding_scale = 10, 1.5
    elif "calm" in voice or "neutral" in voice:
        diffusion_steps, embedding_scale = 5, 1.0
    elif "quality" in voice or "slow" in voice:
        diffusion_steps, embedding_scale = 15, 1.3
    else:
        # Default: fast + natural (5 steps is the sweet spot for voice agents)
        diffusion_steps, embedding_scale = 5, 1.3

    sentences = split_sentences(text)
    logger.info(
        f"Synthesizing {len(sentences)} chunk(s) | {len(text)} chars | "
        f"voice='{voice}' steps={diffusion_steps} scale={embedding_scale}"
    )

    async def generate_stream():
        loop = asyncio.get_event_loop()
        all_pcm = b""

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            t0 = loop.time()
            async with _model_lock:
                try:
                    pcm = await loop.run_in_executor(
                        None, _synthesize_chunk, sentence,
                        diffusion_steps, embedding_scale
                    )
                except Exception as e:
                    import traceback
                    logger.error(f"Synthesis error on chunk {i+1}: {e}\n{traceback.format_exc()}")
                    continue
            t1 = loop.time()
            num_samples = len(pcm) // 2  # 16-bit
            audio_dur = num_samples / SAMPLE_RATE
            rtf = (t1 - t0) / max(audio_dur, 0.01)
            logger.info(
                f"  Chunk {i+1}/{len(sentences)}: {len(sentence)} chars → "
                f"{audio_dur:.2f}s audio in {t1-t0:.2f}s (RTF {rtf:.2f})"
            )
            all_pcm += pcm

        total_samples = len(all_pcm) // 2
        header = _make_wav_header(total_samples, SAMPLE_RATE)
        yield header + all_pcm

    return StreamingResponse(generate_stream(), media_type="audio/wav")

# ── Health + voice list ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    vram_free = None
    if torch.cuda.is_available():
        vram_free = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
    return {
        "status": "ok",
        "model": "styletts2",
        "device": _device,
        "model_loaded": _tts is not None,
        "vram_free_gb": vram_free,
    }

@app.get("/v1/audio/voices")
async def list_voices():
    return {"voices": [
        {"id": "default",    "name": "Natural (5-step, fast)"},
        {"id": "expressive", "name": "Expressive (10-step, more emotion)"},
        {"id": "quality",    "name": "High Quality (15-step, slowest)"},
        {"id": "calm",       "name": "Calm/Neutral (5-step, flat style)"},
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "styletts2_tts_server:app",
        host="0.0.0.0",
        port=8880,
        log_level="info",
        workers=1,
    )
