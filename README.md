# Privacy-First Local Voice Personal Assistant

A privacy-centric, fully local voice personal assistant powered by **Qwen 3.5**, **Faster-Whisper**, and **FastAPI**, accompanied by a modern **React + Vite** web dashboard and **1-Click Turn-Key Docker** deployment.

---

## Key Features

-  **100% Local & Privacy-First**: Runs entirely on local hardware with zero telemetry or audio/text data sent to third-party cloud APIs.
-  **Intelligent LLM Function Calling**: Integrated with `qwen3.5:2b` via Ollama for multi-turn conversation and tool calling.
-  **Live Weather Integration**: Real-time forecasts, rain probability, high/low temperatures, and wind conditions via Open-Meteo.
-  **Local Calendar Management**: Appointment creation, scheduling, conflict detection, and event querying via SQLite.
-  **Dual-Mode Voice Interaction**:
  - **Hold-to-Talk**: Press and hold to record; release to send immediately.
  - **Click-to-Toggle**: Click to start recording, click again to stop and submit.
  - **In-Browser Audio Pipeline**: Downsamples microphone audio to 16kHz mono 16-bit PCM RIFF WAV format.
  - **90s Safety Guard**: Real-time countdown timer with 10MB backend payload protection.
-  **Real-Time 3-State Audio Visualizer**: HTML5 Canvas rendering recording frequency bars, assistant speech harmonics, and ambient idle states.
-  **Multi-Session Management**: Sidebar history manager to switch between past conversations or start new sessions.
-  **Modern Clean Slate UI**: Responsive 3-column layout with minimizable sidebars, display-only widget cards, system-aware Light/Dark themes, and comprehensive Settings.
-  **Turn-Key Docker Deployment**: Bundled Ollama with automatic model provisioning on `http://localhost:3000`.

---

## System Architecture

```
+--------------------------------------------------------------------------------------------------+
|                                    APPLICATION ARCHITECTURE                                      |
|                                                                                                  |
|  [ User Browser / Device ] ──► Port 3000                                                         |
|             │                                                                                    |
|             ▼                                                                                    |
|  +─────────────────────────────────────────────────────────────+                                 |
|  | FRONTEND (React + Vite + TypeScript + Tailwind CSS)         |                                 |
|  | • 3-Column Minimizable Dashboard & Settings Modal           |                                 |
|  | • In-Browser 16kHz Mono WAV Encoder & Analyser Visualizer   |                                 |
|  | • Reverse-Proxied by Nginx Alpine                           |                                 |
|  +─────────────────────────────────────────────────────────────+                                 |
|             │                                                                                    |
|             ▼ HTTP REST / WebSocket (/v1/chat, /v1/chat/audio)                                   |
|  +─────────────────────────────────────────────────────────────+                                 |
|  | BACKEND (FastAPI + Python 3.11/3.14)                        |                                 |
|  | • Orchestrator Pipeline & Request Metrics                   |                                 |
|  | • Speech-to-Text: Faster-Whisper (int8 / CPU optimized)     |                                 |
|  | • Text-to-Speech: Piper TTS / pyttsx3 (eSpeak fallback)     |                                 |
|  | • Storage: SQLite Database (assistant.db)                  |                                 |
|  | • Tools: Open-Meteo Weather & SQLite Calendar Adapter       |                                 |
|  +─────────────────────────────────────────────────────────────+                                 |
|             │                                                                                    |
|             ▼ Async HTTP (:11434)                                                                |
|  +─────────────────────────────────────────────────────────────+                                 |
|  | LOCAL LLM ENGINE (Ollama)                                   |                                 |
|  | • Model: qwen3.5:2b (Function Calling & Multi-Turn Context)  |                                 |
|  +─────────────────────────────────────────────────────────────+                                 |
+--------------------------------------------------------------------------------------------------+
```

---

## 📂 Project Structure

```
f:\Projects\Assisstant\
├── docker-compose.yml                    # Turn-key multi-container Docker compose
├── DEPLOYMENT.md                         # Detailed Docker deployment guide
├── README.md                             # Main project documentation
│
├── backend/                              # FastAPI Backend
│   ├── Dockerfile                        # Production Dockerfile with audio dependencies
│   ├── config.yaml                       # Application configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── main.py                           # FastAPI REST & WebSocket endpoints
│   ├── app/
│   │   ├── config.py                     # YAML configuration loader
│   │   ├── logger.py                     # Structured JSON logging
│   │   ├── database/                     # SQLite schema & repository layer
│   │   ├── tools/                        # Open-Meteo & Calendar adapters
│   │   ├── stt/                          # Faster-Whisper STT engine
│   │   ├── tts/                          # Piper & pyttsx3 TTS engines
│   │   ├── llm/                          # Ollama client with tool calling
│   │   └── core/                         # ContextManager & Orchestrator pipeline
│   └── tests/                            # 44/44 Pytest suite
│
└── web/                                  # React (TypeScript) Web Dashboard
    ├── Dockerfile                        # Multi-stage build (Node 20 -> Nginx Alpine)
    ├── nginx.conf                        # Nginx reverse proxy & SPA configuration
    ├── package.json                      # Frontend dependencies & scripts
    ├── vite.config.ts                    # Vite & Vitest configuration
    ├── tailwind.config.js                # Clean Slate design theme tokens
    └── src/
        ├── app/                          # AppLayout & DashboardPage classes
        ├── components/                   # Chat, widgets, session, settings, UI components
        ├── context/                      # Chat, Audio, Theme, Settings, Toast providers
        ├── hooks/                        # Voice recorder (90s limit), audio player, health
        ├── services/                     # Axios API clients
        ├── utils/                        # WAV encoder, date utilities, sanitizers
        └── tests/                        # 15/15 Vitest component tests
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Docker Deployment (Recommended)

Requires only [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine:

```bash
# Clone the repository and navigate to root directory
cd Assisstant

# Launch the entire stack (bundled Ollama + model + backend + frontend)
docker compose up -d --build
```

Open your browser at **`http://localhost:3000`**.

---

### Option 2: Manual Local Development

#### 1. Start Ollama with Qwen 3.5:
```bash
ollama serve
ollama pull qwen3.5:2b
```

#### 2. Start the Backend API:
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate       # On Windows
source venv/bin/activate      # On Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python main.py
```
*Backend runs at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).*

#### 3. Start the Web Dashboard:
```bash
cd web

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```
*Frontend runs at `http://localhost:3000`.*

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `POST` | `/v1/chat` | Send a text message; returns assistant response and audio. |
| `POST` | `/v1/chat/audio` | Send a 16kHz WAV voice recording for STT transcription and response. |
| `GET` | `/v1/sessions` | Retrieve past conversation sessions for the sidebar manager. |
| `GET` | `/v1/history` | Retrieve paginated message turns for a specific session. |
| `POST` | `/v1/session/new` | Create and initialize a new conversation session. |
| `GET` | `/v1/weather/current` | Get live weather data for a specified city (default: Marburg). |
| `GET` | `/v1/calendar/events` | Retrieve upcoming calendar events. |
| `GET` | `/v1/health` | Comprehensive system health check (Ollama, SQLite, model). |
| `WS` | `/v1/ws/chat` | Bidirectional WebSocket connection for streaming chat. |

---

## ⚙️ Configuration (`backend/config.yaml`)

```yaml
assistant:
  name: "GWEN"

llm:
  model_name: "qwen3.5:2b"
  ollama_base_url: "http://localhost:11434"
  temperature: 0.7
  context_window_turns: 10

stt:
  engine: "faster-whisper"
  model_size: "base.en"

tts:
  engine: "pyttsx3"                 # "pyttsx3" | "piper"
  voice: "en_US-lessac-medium"

weather:
  provider: "open-meteo"
  units:
    temperature: "celsius"

calendar:
  provider: "sqlite"

locale:
  timezone: "Europe/Berlin"

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
```

---

## Automated Testing

### Run Backend Tests (Pytest)
```bash
cd backend
pytest -v
```
*Result: **44 / 44 tests passed** (100% pass rate).*

### Run Frontend Tests (Vitest)
```bash
cd web
npm test
```
*Result: **15 / 15 tests passed** across 8 test suites.*

### Run Production Build Validation
```bash
cd web
npm run build
```

---

##  Project Roadmap

- [x] **Phase 1: Core Engine**: FastAPI server, Faster-Whisper STT, Piper/pyttsx3 TTS, Qwen 3.5 LLM function calling, Open-Meteo weather tool, and SQLite calendar tool.
- [x] **Phase 2: Web Application**: Modern React + TypeScript SPA, 3-column layout, dual-mode voice recorder, 3-state canvas audio visualizer, minimizable panels, and settings manager.
- [x] **Docker Containerization**: Self-contained 4-tier Docker Compose deployment with automated model provisioning and Nginx reverse proxy.
- [ ] **Phase 3: Mobile Application**: Cross-platform native mobile application for iOS and Android.

---

## License
This project is open-source and licensed under the MIT License.
