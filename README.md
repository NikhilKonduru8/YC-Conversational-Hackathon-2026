# Jarvis — Real-Time, Vision-Grounded Engineering Co-Pilot for Smart Glasses

Jarvis is a wearable, hands-free AI assistant for electronics work. You wear a
pair of smart glasses — a **camera**, a **microphone**, and a small **on-glasses
display** — wired to a Raspberry Pi in your pocket. You say **"Hey Jarvis,"** ask
about whatever is on your workbench, and it **sees** the component through the
camera, **retrieves** its real datasheet specifications, **reasons** over them,
and **speaks** the answer back — in roughly three seconds — while remembering the
conversation so natural follow-ups ("now make it blink every three seconds") just
work.

It is a genuine real-time **co-pilot**: it listens continuously, it's **proactive**
(if your request is ambiguous it asks a quick clarifying question instead of
guessing), you can **barge in** mid-answer by saying "Hey Jarvis" again to
redirect it, and you quit any time by saying **"Jarvis exit."**

### 🎯 Who it's for

**Hobbyists and makers** first and foremost — the people hacking on breadboards,
Arduinos, and ESP32s at home who constantly stop to look up "what's the max
voltage on this chip" or "which pin is the output." It's just as useful for
students learning electronics and for engineers who want a hands-free reference
at the bench.

### 🖥️ The on-glasses display (a key differentiator)

Most voice assistants are a black box — you talk and hope. Jarvis puts a small
**OLED display right on the glasses** that shows exactly what the agent is doing
in real time: **Listening → Looking → Searching → Thinking → Speaking**, and
**"Okay, done"** when you exit. Paired with the camera and microphone, that means
the wearer always has glanceable visual feedback about the agent's state — no
guessing whether it heard you, is looking at the board, or is mid-thought. It's a
small piece of hardware that makes the whole interaction feel trustworthy.

---

## 🎒 Wear it and go — it just connects to a speaker

The setup is intentionally minimal. The **glasses go on your head** (camera +
microphone + display), the **Raspberry Pi goes in your pocket** running the whole
application, a **USB power bank** powers it, and it **connects to a speaker — and
that's it.** No laptop, no desktop, no tethered computer anywhere in the loop. You
stand up and walk around the lab with the full pipeline running.

The on-device layer — wake-word detection, voice-activity detection, audio
resampling, the orchestration state machine, and the display — runs **locally on
the Pi** for instant response; the heavy intelligence (speech, vision, retrieval,
reasoning) is reached over the Pi's own network connection, keeping the device
light and battery-friendly.

---

## 🧩 One big loop — every sponsor is connected

This is **not** five tools sitting side by side. They are wired into a single
closed loop, and each one feeds the next. The interesting engineering is in the
**integrations between them**:

```
                    ┌──────────────────────── THE JARVIS LOOP ────────────────────────┐
                    │                                                                  │
   🎙️ USB Mic ──►  openWakeWord            📷 USB Camera ──► OpenCV frame grab         │
   (16 kHz mono)    "Hey Jarvis"  (on-device, ONNX)                │                   │
                         │                                         ▼                   │
                         │ wake                         ┌──────────────────────┐       │
                         ▼                              │   QWEN3-VL-FLASH      │       │
                 ┌───────────────┐                      │  (Alibaba DashScope) │       │
                 │  LiveKit STT  │                      │  "what is the camera │       │
                 │ Deepgram      │                      │   actually seeing?"  │       │
                 │ nova-3        │  transcript          └──────────┬───────────┘       │
                 │ (WebRTC,      │ ───────────┐                    │ scene description │
                 │  streaming)   │            │                    │ (part #s, LEDs)   │
                 └───────────────┘            ▼                    ▼                   │
                                       ┌─────────────────────────────────────┐         │
                                       │       CONTEXT COMPILER              │         │
                                       │  fuses SPEECH + VISION into one      │         │
                                       │  semantic query, e.g.                │         │
                                       │  sees "LM358" + hears "max voltage"  │         │
                                       │  →  "LM358 absolute maximum ratings" │         │
                                       └──────────────────┬──────────────────┘         │
                                                          ▼                            │
                                       ┌─────────────────────────────────────┐         │
   📄 datasheet PDFs ──► UNSILOED AI ─►│            MOSS                     │         │
       (offline, structured            │  real-time semantic search          │         │
        chunk extraction)              │  (<10 ms, Rust/WASM vector index)   │         │
                                       └──────────────────┬──────────────────┘         │
                                                          │ matching spec chunks       │
                                                          ▼                            │
                                  ┌────────────────────────────────────────────┐      │
                                  │   GROUNDING BLOCK                          │      │
                                  │   = transcript + Qwen vision + Moss specs   │      │
                                  └───────────────────┬────────────────────────┘      │
                                                      ▼                               │
                                       ┌─────────────────────────────────────┐         │
                                       │          MINIMAX-M3                 │         │
                                       │  grounded reasoning, thinking off,  │         │
                                       │  streaming, conversation memory     │         │
                                       └──────────────────┬──────────────────┘         │
                                                          │ sentence by sentence       │
                                                          ▼                            │
                                       ┌─────────────────────────────────────┐         │
                                       │  LiveKit TTS — Cartesia sonic-3     │ ──► 🔊  │
                                       │  (WebRTC, streamed, ~1.3× speed)    │         │
                                       └─────────────────────────────────────┘         │
                    │                                                                  │
                    │   every stage → on-glasses display · "Hey Jarvis" barges in ·    │
                    │   "Jarvis exit" ends the loop · question + answer kept in memory │
                    └──────────────────────────────────────────────────────────────────┘
```

**How the integrations chain:**

- **Unsiloed → Moss.** Unsiloed AI parses raw datasheet PDFs into clean,
  structured chunks; those chunks *are* the corpus Moss indexes. Unsiloed makes
  the knowledge real; Moss makes it instantly searchable.
- **Qwen + LiveKit STT → Context Compiler.** The vision model says *what the part
  is*; the speech model says *what you want to know*. The compiler fuses them into
  one query (vision sees `LM358`, you ask "max voltage" → it queries Moss for
  `LM358 absolute maximum ratings`).
- **Moss + Qwen → MiniMax.** The retrieved spec chunks and the visual scene become
  a single grounding block, so MiniMax answers **only** from what is *seen* and
  *verified* — not hallucinated.
- **LiveKit** wraps the whole loop as natural, interruptible voice (ears + mouth).

No single sponsor produces the experience alone — **the product is the loop.**

---

## 🛠️ Technical specifications

### Hardware
| Part | Spec |
|------|------|
| Compute | Raspberry Pi 4 (ARM64, Python 3.11) |
| Microphone | Bluetooth lavalier USB microphone (resampled to 16 kHz mono on-device) |
| Camera | USB webcam via OpenCV `VideoCapture` |
| Display | SSD1306 128×64 OLED over I²C, on the glasses (addr `0x3C`, `SDA=GPIO2`, `SCL=GPIO3`) — live agent-state feedback |
| Audio out | Speaker (USB / HDMI) |
| Power | USB power bank |

### Audio pipeline (on-device + LiveKit)
- **16 kHz mono** internal format (required by the wake word + VAD).
- **`sounddevice` / PortAudio** for capture + playback, with an always-on input
  stream feeding an async queue shared by wake detection and STT.
- **`scipy.signal.resample_poly`** software resampling — captures at the mic's
  native rate (e.g. 48 kHz) and resamples to 16 kHz in; resamples TTS audio to a
  rate the speaker accepts on the way out.
- **Format/channel/rate probing** on output (stereo/mono × int16/float32 × several
  rates, with device fallback) so it "just works" across finicky Pi/HDMI/ALSA
  devices.
- **openWakeWord** — the **"Hey Jarvis" wake word runs entirely on-device** as an
  ONNX model (no cloud, no key, ~80 ms frames), so the glasses idle privately and
  wake instantly.
- **Google WebRTC VAD (`webrtcvad`)** — 20 ms-frame voice-activity detection for
  utterance endpointing (knowing when you've stopped talking).
- **LiveKit Inference STT/TTS** — the cloud ends of the audio pipeline. The mic's
  16 kHz frames are streamed *live* into LiveKit STT as you speak; MiniMax's text
  is streamed into LiveKit TTS and the returned audio frames are resampled and
  played back. LiveKit is what turns raw audio into a conversation.

### Models & services
*(ordered by how central they are to this build)*

1. **LiveKit Agents (WebRTC real-time media framework)** — the backbone of the
   voice experience. We use `inference.STT` and `inference.TTS` **standalone**
   (driven directly with `rtc.AudioFrame`s, no room, via a self-managed HTTP
   session context).
   - **STT:** Deepgram **nova-3**, **streaming** — transcribes *live while you
     speak*, not after.
   - **TTS:** Cartesia **sonic-3**, streamed and sped up (~1.3×) via the
     provider's native speed control (no pitch change).
2. **Moss** — the **real-time retrieval layer, and a centerpiece of this build.**
   Moss is a runtime for semantic search with a **Rust/WASM vector index** that
   delivers **sub-10 ms lookups with no external vector database to stand up**. We
   index one vector per datasheet chunk; at query time it returns the exact spec
   passages for the component in view. It's what lets Jarvis ground every answer
   in verified data *without* adding latency — even as the component library grows
   large, retrieval stays ~10 ms.
3. **Qwen3-VL-Flash** (Alibaba DashScope, workspace MaaS endpoint, OpenAI-compatible)
   — multimodal scene understanding; **thinking disabled** for low latency; frames
   sent as downsampled base64 JPEGs.
4. **MiniMax-M3** (OpenAI-compatible) — grounded reasoning engine; **thinking
   disabled** (direct, fast, clean output) with a `<think>`-stripping safety
   filter; streamed sentence-by-sentence into TTS; carries rolling conversation
   memory.
5. **Unsiloed AI** — datasheet PDF → structured chunk extraction (offline). Builds
   the verified corpus that Moss indexes — the reason answers cite real specs
   instead of guesses.

### Software architecture
- **ROS-style node graph.** Each stage is an independent **node** communicating
  over a lightweight in-process **topic/service bus** (`src/bus.py`,
  `src/topics.py`) — `publish/subscribe` for events, request/response for services.
  Decoupled, individually testable, and portable to ROS 2.
- **`asyncio`** throughout, with heavy parallelism (see latency section).
- **`uv`** for dependency + environment management; **`ruff`** for lint/format;
  **`pytest`** for the test suite (49 tests).

| Node | File | Role |
|------|------|------|
| Wake | `nodes/wake_node.py` | openWakeWord "Hey Jarvis" on the shared mic |
| STT | `nodes/stt_node.py` | streams mic → LiveKit STT live → transcript |
| Vision | `nodes/vision_node.py` | OpenCV capture → Qwen3-VL-Flash |
| Retrieval | `nodes/retrieval_node.py` | fuse speech+vision → Moss query → specs |
| Reasoning | `nodes/reasoning_node.py` | MiniMax-M3, grounded, rolling memory |
| TTS | `nodes/tts_node.py` | LiveKit TTS → speaker, interruptible |
| Display | `nodes/display_node.py` | on-glasses OLED state (luma.oled / SSD1306) |
| Orchestrator | `nodes/orchestrator_node.py` | state machine, barge-in, exit |

State machine: `Sleeping → Listening → Looking → Searching → Thinking → Speaking`.

### Latency engineering
Several deliberate optimizations keep it conversational (~3 s):
- **Moss sub-10 ms retrieval** — semantic search returns in **~10 ms even when
  pulling from a large component database**, so grounding adds virtually nothing
  to the latency budget. This is a big reason the whole loop stays fast; a
  traditional vector-DB round trip would cost far more.
- **Parallel vision** — Qwen fires the instant you start talking, hidden behind
  your speech + transcription instead of adding 2–3 s serially.
- **Live streaming STT** — audio is pushed to STT frame-by-frame as you speak, so
  the transcript is ready almost the moment you stop.
- **Thinking disabled** on both MiniMax-M3 and Qwen3-VL-Flash — no reasoning-token
  generation latency.
- **Sentence-by-sentence TTS** — the first sentence is spoken while the rest is
  still being generated (back-pressured through the bus).
- **On-device wake + VAD + resampling** — zero network round-trips for the parts
  that must feel instant.

### What runs where
| On-device (Raspberry Pi) | Over the network |
|--------------------------|------------------|
| Wake word (openWakeWord/ONNX) | LiveKit STT (Deepgram nova-3) |
| Voice-activity detection (WebRTC VAD) | LiveKit TTS (Cartesia sonic-3) |
| Audio capture/playback + resampling | Qwen3-VL-Flash (vision) |
| Orchestration state machine | Moss (semantic search) |
| On-glasses OLED display, camera capture | MiniMax-M3 (reasoning) |

---

## ✨ Features

- **On-device wake word** — "Hey Jarvis," private and instant, no key.
- **Vision-grounded answers** — answers about the component the camera sees.
- **Verified grounding** — every spec traces to a real datasheet (Unsiloed → Moss).
- **Proactive clarification** — if a request is ambiguous or missing a key detail,
  Jarvis asks a short follow-up question to build understanding instead of
  guessing. It's an active conversational partner, not a passive listener.
- **Conversation memory** — follow-ups keep context across turns.
- **On-glasses status display** — glanceable visual feedback at every stage.
- **Barge-in** — say "Hey Jarvis" mid-answer to interrupt and redirect.
- **Voice exit** — "Jarvis exit" → display shows "Okay, done" and quits.
- **Resilient** — any model/service failure degrades gracefully instead of crashing.

---

## 🔭 Future improvements

- **More compact glasses** — integrate the camera, microphone, and display
  directly into the frame, with a cleaner, smaller display, so it looks and feels
  like ordinary eyewear.
- **Custom compute board instead of a Raspberry Pi** — a smaller, more
  power-efficient board, either built into the glasses themselves or shrunk into a
  tiny pod that slips into a pocket.
- **Bone-conduction earpiece** — replace the external speaker so the glasses are
  the entire interface, fully private audio with nothing else attached.

---

## 🚀 Setup (Raspberry Pi)

```bash
cd agent-py
bash setup-pi.sh                 # PortAudio + OpenCV + I2C libs, uv, deps, wake model
sudo raspi-config                # Interface Options → I2C → Enable, then reboot

cp .env.example .env.local       # fill in your keys (see below)
uv run src/check_devices.py      # verify mic / speaker / camera / OLED

# drop your component JSON into agent-py/data/ (see data/example_component.json)
uv run src/create_index.py       # build the Moss index
uv run src/jarvis.py             # run — say "Hey Jarvis"
```

### API keys (in `agent-py/.env.local`, gitignored — **no keys live in this repo**)

| Service | Variables |
|---------|-----------|
| LiveKit | `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL` |
| Moss | `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` |
| Qwen3-VL | `QWEN_BASE_URL` (workspace MaaS endpoint), `QWEN_API_KEY`, `QWEN_MODEL` |
| MiniMax | `MINIMAX_API_KEY`, `MINIMAX_MODEL` |

See `agent-py/.env.example` for the complete, documented list (audio device
selection, wake threshold, TTS speed, exit phrases, OLED geometry, etc.).

## 🧪 Tests

```bash
cd agent-py && uv run pytest      # 49 deterministic unit + integration tests
```

---

## Project layout

```
agent-py/
├── src/
│   ├── jarvis.py            # launch / composition root
│   ├── bus.py, topics.py    # ROS-style topic/service bus + graph names
│   ├── nodes/               # wake, stt, vision, retrieval, reasoning, tts, display, orchestrator
│   ├── audio_io.py          # sounddevice mic+speaker, resampling, device probing
│   ├── wake.py              # openWakeWord wrapper
│   ├── stt.py / tts.py      # LiveKit Inference STT / TTS wrappers
│   ├── vision.py            # OpenCV camera + Qwen3-VL client
│   ├── moss_context.py      # speech+vision → Moss query compiler
│   ├── reasoning.py         # MiniMax-M3 client (+ <think> filter)
│   ├── config.py, state.py, display.py
│   └── create_index.py, map_components.py   # build the Moss index
├── data/                    # component DB (gitignored; example schema included)
├── setup-pi.sh              # one-shot Raspberry Pi setup
└── pyproject.toml           # uv-managed deps
```

*Built for the YC Conversational AI Hackathon 2026, integrating LiveKit, Moss,
Qwen, MiniMax, and Unsiloed AI.*
