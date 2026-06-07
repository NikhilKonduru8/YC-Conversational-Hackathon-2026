# Jarvis — Real-Time Engineering Co-Pilot for Smart Glasses

Jarvis is a hands-free, vision-grounded assistant for electronics work. You wear
camera glasses wired to a Raspberry Pi in your pocket, say **"Hey Jarvis,"** and
ask about whatever is on your workbench. It *sees* the component through the
camera, looks up its real datasheet specs, reasons over them, and **speaks the
answer back** — in about three seconds, while remembering the conversation so
follow-ups just work.

It's a real-time **co-pilot**: it listens continuously, you can barge in
mid-answer by saying "Hey Jarvis" again to redirect it, and you quit any time by
saying **"Jarvis exit."**

> **Self-contained & portable.** Everything runs on a Raspberry Pi powered by a
> USB power bank — no laptop or external computer tethered. The wake word, voice
> endpointing, state machine, and OLED run on-device; the heavy AI is reached
> over the Pi's own network connection. Put the Pi in your pocket and walk
> around. (The only external piece is a dev speaker; a production build would use
> a bone-conduction earpiece.)

---

## How it works

```
        🎙️ USB Mic                          📷 USB Camera
            │                                     │
            ▼                                     ▼
   openWakeWord ("Hey Jarvis", on-device)   Qwen3-VL-Flash ─► visual scene
            │                                     │           (part numbers,
            ▼  (live audio stream)                │            LEDs, wiring)
   LiveKit STT (streaming) ─► transcript          │
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                  CONTEXT COMPILER  (fuses speech + vision into one query)
                  e.g. sees "LM358" + hears "max voltage"
                               │
                               ▼
                  Moss semantic search ◄── indexed from ── Unsiloed-parsed datasheets
                               │  (matching spec chunks)
                               ▼
        GROUNDING BLOCK = transcript + Qwen vision summary + Moss specs
                               │
                               ▼
                  MiniMax-M3 (grounded reasoning, streaming)
                               │  sentence by sentence
                               ▼
                  LiveKit TTS (Cartesia sonic-3) ─► 🔊 Speaker
                               │
        Every stage updates the OLED · "Hey Jarvis" barges in · "Jarvis exit" quits
```

The orchestrator runs a state machine —
`Sleeping → Listening → Looking → Searching → Thinking → Speaking` — and overlaps
work to stay conversational: vision fires in parallel the moment you start
talking, and transcription streams live as you speak, so when you stop the answer
comes back fast.

---

## Architecture

A single standalone Python app on the Pi, organized as a **ROS-style node graph**.
Each pipeline stage is an independent node communicating over a lightweight
in-process **topic/service bus** (`src/bus.py`, `src/topics.py`) — decoupled,
testable, and portable to ROS 2 later.

| Node | File | Responsibility |
|------|------|----------------|
| Wake | `nodes/wake_node.py` | "Hey Jarvis" detection (openWakeWord, local) |
| STT | `nodes/stt_node.py` | streams mic → LiveKit STT live → transcript |
| Vision | `nodes/vision_node.py` | USB camera (OpenCV) → Qwen3-VL-Flash |
| Retrieval | `nodes/retrieval_node.py` | fuse speech+vision → Moss search |
| Reasoning | `nodes/reasoning_node.py` | MiniMax-M3, grounded, conversation memory |
| TTS | `nodes/tts_node.py` | LiveKit TTS → speaker, interruptible |
| Display | `nodes/display_node.py` | OLED status (SSD1306 over I2C) |
| Orchestrator | `nodes/orchestrator_node.py` | state machine, barge-in, exit |

Supporting modules: `config.py` (all env config), `audio_io.py` (sounddevice mic
+ speaker with rate/format resampling), `wake.py`, `stt.py`, `vision.py`,
`moss_context.py`, `reasoning.py`, `display.py`, `state.py`. `jarvis.py` is the
launch/composition root. `create_index.py` + `map_components.py` build the Moss
index from the component database.

---

## The stack — how each piece is used

- **Unsiloed AI** — parses component **datasheet PDFs into structured, chunked
  text**. This is the knowledge foundation: every spec Jarvis quotes traces back
  to a real datasheet that Unsiloed extracted (not a web guess).
- **Moss** — the **real-time retrieval layer**. Indexes the Unsiloed-parsed
  chunks (one vector per chunk) and serves sub-10 ms semantic search, grounding
  the LLM in the right spec passages for the component in view.
- **Qwen3-VL-Flash** — the **eyes**. Describes the workbench from camera frames
  (chip part numbers, LED states, wiring), run with thinking disabled and in
  parallel with speech so its latency is hidden.
- **MiniMax-M3** — the **brain**. Reasons over the fused context (thinking
  disabled for speed + clean output) and produces a brief, grounded spoken answer.
- **LiveKit Inference** — the **voice I/O**. Streaming STT (Deepgram nova-3) and
  TTS (Cartesia sonic-3), used standalone, wrapping the whole pipeline into a
  natural, interruptible voice conversation.

The product is the **chain**: vision identifies the part, Moss (fed by Unsiloed)
grounds the facts, MiniMax composes the answer, and LiveKit carries it as speech.

---

## Hardware

- Raspberry Pi 4 or 5
- USB microphone, USB camera (the "glasses")
- Speaker (dev) — bone-conduction earpiece in production
- SSD1306 OLED over I2C (status display): VCC→3V3, GND→GND, SDA→GPIO2, SCL→GPIO3
- USB power bank

---

## Setup (Raspberry Pi)

```bash
cd agent-py
bash setup-pi.sh                 # system libs (PortAudio, OpenCV, I2C) + uv + deps + wake model
sudo raspi-config                # Interface Options → I2C → Enable, then reboot

cp .env.example .env.local       # then fill in your keys (see below)
uv run src/check_devices.py      # confirm mic / speaker / camera / OLED

uv run src/map_components.py     # (optional) preview the data folder
uv run src/create_index.py       # build the Moss index from agent-py/data/
uv run src/jarvis.py             # run — say "Hey Jarvis"
```

Drop your component JSON files into `agent-py/data/` (see
`data/example_component.json` for the schema), then re-run `create_index.py`.

### Required API keys (in `agent-py/.env.local` — gitignored)

| Service | Vars | Purpose |
|---------|------|---------|
| LiveKit | `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL` | STT + TTS |
| Moss | `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | semantic search |
| Qwen3-VL | `QWEN_BASE_URL` (workspace MaaS endpoint), `QWEN_API_KEY`, `QWEN_MODEL` | vision |
| MiniMax | `MINIMAX_API_KEY`, `MINIMAX_MODEL` | reasoning |

See `agent-py/.env.example` for the full list (audio device selection, wake
threshold, TTS speed, OLED, etc.). **No keys are stored in this repo.**

---

## Features

- **Wake word** — "Hey Jarvis" (local, no key).
- **Vision-grounded answers** — answers about the component the camera sees.
- **Conversation memory** — follow-ups keep context ("now make it blink").
- **Barge-in** — say "Hey Jarvis" mid-answer to interrupt and redirect.
- **Exit** — say "Jarvis exit" to quit (OLED shows "Okay, done").
- **Low latency** — parallel vision, live streaming STT, thinking-off reasoning.

## Tests

```bash
cd agent-py && uv run pytest      # deterministic unit + integration tests
```
