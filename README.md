# AI VTuber Persona System

An AI-powered VTuber system that combines a local LLM with a 3D VRM avatar, text-to-speech, and live chat integration.

![Version](https://img.shields.io/badge/version-5.8-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- **Local LLM Integration** - Uses LM Studio for privacy-focused, offline AI responses
- **3D VRM Avatar** - Animated character with facial expressions and emotions
- **Text-to-Speech** - Natural voice output with lip sync (Piper TTS)
- **Live Chat Integration** - Connects to Kick.com chat for stream interaction
- **Voice Input** - Speech-to-text for collaborations (Whisper — supports openai-whisper and faster-whisper)
- **AI-to-AI Peer Link** - Synchronize two AI instances for collaborative conversations (no talking over each other)
- **Idle Chatter** - Fill dead air automatically using topic prompt files
- **3-Tier Memory System** - Per-user logs, stream highlights, and vector database for long-term semantic memory
- **Emotion Detection** - AI responses include emotion tags for avatar expressions
- **Audio Output Routing** - Choose which audio device TTS plays through (useful for OBS virtual cables)
- **OBS Ready** - Transparent background stage for easy streaming setup

---
## 🎥 Demo Video

<div align="center">
  <a href="PASTE_YOUR_VIDEO_URL_HERE">
    <img src="<iframe width="560" height="315" src="https://www.youtube.com/embed/2-DD4iH9QDQ?si=Fh_HNvPuUfkIBNX9" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>" 
         alt="AI VTuber Demo" width="640">
  </a>
  <p><strong>Watch the full demo (with voice, chat, and avatar in action)</strong></p>
</div>
---
## 🚀 Quick Start

### 1. Download & Extract
```bash
https://github.com/au79g/Idiots-AI-vtuber-project.git
cd ai-vtuber
```

### 2. Run Setup
Double-click `start.bat` and select **[5] First Time Setup**

Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start LM Studio
- Download [LM Studio](https://lmstudio.ai/)
- Load a model (recommended: **Qwen 2.5 3B Instruct**)
- Start the local server (default port: 1234)

### 4. Run the AI
Double-click `start.bat` and select **[1] Start AI + Open Stage**

Or:
```bash
venv\Scripts\activate
python ai_persona.py
```

---

## 📁 Project Structure

```
ai-vtuber/
├── ai_persona.py           # Main AI persona script
├── animation_organizer.py  # Tool to sort/organize VRMA animation files
├── vector_db_manager.py    # Vector database import tool
├── streaming_stage.html    # OBS-ready 3D viewer
├── vrm_viewer_v3.html      # Testing/development viewer
│
├── start.bat               # Launcher menu
├── run.bat                 # Quick start (AI only)
│
├── character.json          # SillyTavern character card (active)
├── character.example.json  # Example character card for reference
├── lorebook.json           # SillyTavern lorebook (active)
├── lorebook.example.json   # Example lorebook for reference
├── requirements.txt        # Python dependencies
│
├── *.vrm                   # VRM model files (place your .vrm models here)
│
├── scripts/
│   ├── intro.txt           # Stream intro script
│   └── outro.txt           # Stream outro script
│
├── animations/             # VRMA animation files
│   ├── idle/               # Default/neutral poses
│   ├── happy/              # Joy, excitement
│   ├── sad/                # Sadness, disappointment
│   ├── angry/              # Anger, frustration
│   ├── surprised/          # Shock, surprise
│   ├── talking/            # General speaking gestures
│   ├── greeting/           # Waves, hellos
│   └── general/            # Uncategorized
│
├── memories/               # Conversation & memory storage
│   ├── users/              # Per-user conversation logs
│   ├── stream_highlights.txt
│   └── voice_sessions/     # Voice transcripts
│
├── memory_db/              # Supplemental memory/database storage
│
├── topics/                 # Idle chatter topic prompt files
│   └── *.txt               # One prompt per line — AI uses these to fill dead air
│
├── vector_db/              # ChromaDB vector storage
│
├── venv/                   # Python virtual environment (auto-created by setup)
│
└── piper/                  # TTS engine
    ├── piper.exe
    └── en_US-hfc_female-medium.onnx
```

---

## 📦 Installation

### Core Dependencies
```bash
pip install langchain langchain-openai websockets playsound==1.2.2
pip install g2p-en nltk
```

### Memory System (Vector Database)
```bash
pip install langchain-chroma chromadb
pip install langchain-huggingface sentence-transformers
```

### Voice Input (Optional)
Choose one Whisper backend:
```bash
# Option A: openai-whisper (CPU/GPU, easier to install)
pip install openai-whisper sounddevice numpy

# Option B: faster-whisper (GPU optimized, faster)
pip install faster-whisper sounddevice numpy
```
Also requires [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

### Kick.com Chat (Optional)
```bash
pip install KickApi
```

### Vector Database Manager Tool (Optional but Recommended)
```bash
pip install pymupdf      # PDF support
pip install psutil       # Memory monitoring
pip install numpy        # Semantic chunking
```

### All at Once
```bash
pip install langchain langchain-openai langchain-chroma chromadb
pip install langchain-huggingface sentence-transformers
pip install websockets playsound==1.2.2 g2p-en nltk
pip install openai-whisper sounddevice numpy pymupdf psutil KickApi
```

---

## 🗄️ Vector Database Manager

The **Vector Database Manager** (`vector_db_manager.py`) lets you import documents into the AI's long-term memory. This allows your VTuber to reference rulebooks, lore documents, world-building content, and more during conversations.

### Running the Tool
```bash
python vector_db_manager.py
```

### Features
- **GUI file/folder selection** - Easy import via file picker
- **PDF support** - Import PDF rulebooks and documents
- **Progress bars with ETA** - See exactly how long imports will take
- **Memory monitoring** - Track RAM usage during large imports
- **Three chunking modes:**

| Mode | Best For | Speed |
|------|----------|-------|
| **Fixed** | General text, large documents | Fast |
| **Semantic** | Rulebooks, reference material, structured content | Slower |
| **Sentence** | Articles, narratives | Medium |

---

## 🎞️ Animation Organizer

The **Animation Organizer** (`animation_organizer.py`) is a utility tool for sorting and managing VRMA animation files into the correct emotion subfolders. Run it to batch-organize loose animation files before starting a stream.

```bash
python animation_organizer.py
```

---

## ⚙️ Configuration

Edit the `Config` class in `ai_persona.py`:

### Essential Settings
```python
# Your admin username (can use admin commands)
ADMIN_USERS = ["your_username"]

# LM Studio connection
LLM_BASE_URL = "http://localhost:1234/v1"
```

### Kick.com Chat (Optional)
```python
KICK_CHANNEL = "your_kick_username"
```

### Voice Input (Optional)
```python
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE = "cuda"       # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" for CPU
```

### Whisper Transcription Buffering
```python
# Combines split transcription fragments before sending to LLM
VOICE_BUFFER_ENABLED = True
VOICE_BUFFER_TIMEOUT = 2.0   # Seconds of silence before combining & sending
```

### Audio Output Routing
```python
# Route TTS audio to a specific output device (e.g. OBS virtual cable)
# Use !audio devices to list available devices, then set the index
AUDIO_OUTPUT_DEVICE = None   # Set to device index number
```

### Idle Chatter
```python
IDLE_CHATTER_ENABLED = False     # Enable with !idle chatter on
IDLE_TIMEOUT_SECONDS = 45.0      # Seconds of silence before rambling starts
IDLE_RAMBLE_INTERVAL = 5.0       # Minimum seconds between rambles
IDLE_MAX_RAMBLES_PER_TOPIC = 3   # Max rambles per topic before switching
IDLE_CHATTER_TOKENS = 200        # Token limit for idle chatter responses
IDLE_TOPICS_DIR = Path("./topics")
```

### AI-to-AI Peer Link
```python
PEER_LINK_ENABLED = False       # Enable with !peer on host / !peer on client
PEER_LINK_PORT = 9876
PEER_LINK_MODE = "host"         # "host" or "client"
PEER_LINK_HOST = "127.0.0.1"   # IP of the host machine
PEER_LINK_NAME = "AI_1"         # Friendly name for this instance
PEER_LINK_WAIT_TIMEOUT = 30.0
PEER_LINK_TURN_DELAY = 1.0      # Natural pause after peer finishes speaking
```

### Memory Settings
```python
MAX_USER_HISTORY_LINES = 10   # Lines of chat history per user
MAX_VECTOR_RESULTS = 2        # Vector DB results to include
MAX_LORE_ENTRIES = 2          # Lorebook entries to include
```

---

## 🎮 Commands

### Public Commands
| Command | Description |
|---------|-------------|
| `user <name>` | Set your username |
| `help` | Show commands |
| `exit` | Quit |

### Admin Commands (prefix with `!`)
| Command | Description |
|---------|-------------|
| `!stream on/off` | Toggle streaming mode |
| `!viewers <n>` | Set viewer count |
| `!intro` | Play intro script |
| `!outro` | Play outro script |
| `!idle` | Return to idle state |
| `!setidle <name>` | Set default idle animation |
| `!emotion <e>` | Test emotion |
| `!anim <name>` | Play animation |
| `!anim list` | List all animations |
| `!reload` | Reload character/lorebook/animations |
| `!addadmin <name>` | Add admin for this session |

### Audio Output Commands
| Command | Description |
|---------|-------------|
| `!audio devices` | List available audio output devices |
| `!audio device <index>` | Route TTS to a specific device |

### Voice Input Commands
| Command | Description |
|---------|-------------|
| `!voice on/off` | Enable/disable voice input |
| `!voice test` | Test microphone |
| `!voice devices` | List audio input devices |
| `!voice ptt` | Push-to-talk (record once) |
| `!voice listen` | Continuous listening (VAD) |
| `!voice stop` | Stop listening |

### Kick Chat Commands
| Command | Description |
|---------|-------------|
| `!kick` | Show connection status |
| `!kick connect` | Connect to your channel |
| `!kick connect <channel>` | Connect to a specific channel |
| `!kick disconnect` | Disconnect from Kick |
| `!kick poll` | Start polling chat (no auto-respond) |
| `!kick process` | Start polling + auto-responding |
| `!kick stop` | Stop polling and processing |
| `!kick next` | Process one message manually |
| `!kick clear` | Clear the message queue |

### Idle Chatter Commands
| Command | Description |
|---------|-------------|
| `!idle chatter` | Show idle chatter status |
| `!idle chatter on/off` | Enable/disable idle chatter |
| `!idle topics` | List topic files and prompt counts |
| `!idle timeout <seconds>` | Set silence timeout before rambling |
| `!idle ramble` | Force a single ramble immediately (testing) |

### AI-to-AI Peer Link Commands
| Command | Description |
|---------|-------------|
| `!peer` | Show peer link status |
| `!peer on host` | Start as host (listens for peer connection) |
| `!peer on client [ip]` | Connect to a host AI instance |
| `!peer off` | Disconnect peer link |
| `!peer name <name>` | Set this instance's display name |

---

## 💭 Idle Chatter Setup

The idle chatter system keeps your stream engaging when chat is quiet. The AI picks random prompts from `.txt` files in the `topics/` folder and generates unprompted conversation.

### Creating Topic Files
Create any `.txt` file in the `topics/` folder with one prompt per line:
```
# topics/games.txt
Talk about your favorite video game genre
Share your opinion on a recent game release
Ask viewers what games they're playing lately
```

```
# topics/random.txt
Say something philosophical about existence
Share a fun fact about something you find interesting
Talk about what you'd do if you had a free day
```

Lines beginning with `#` are treated as comments and ignored. Then enable it during a stream:
```
!stream on
!idle chatter on
```

---

## 🤝 AI-to-AI Peer Link Setup

The Peer Link allows two AI VTuber instances on the same local network to take turns speaking without talking over each other — useful for AI collab streams.

**Instance 1 (host machine):**
```
!peer on host
```

**Instance 2 (client machine):**
```
!peer on client 192.168.1.100
```

Each instance broadcasts its state (idle / listening / processing / speaking) to the other. When one is speaking, the other waits before responding.

---

## 🎭 Character Customization

### Using SillyTavern Cards
Place your character card as `character.json` (see `character.example.json` for structure):
```json
{
  "name": "Your Character",
  "description": "Character description...",
  "personality": "Personality traits...",
  "first_mes": "Hello! I'm your AI VTuber!",
  "mes_example": "<START>\n{{user}}: Hi!\n{{char}}: Hey there!"
}
```

### Using a Lorebook
Place your lorebook as `lorebook.json` (see `lorebook.example.json` for structure). Lore entries activate based on keywords in the conversation.

### Using Vector Memory
Import reference documents with `vector_db_manager.py` for the AI to search during conversations. Great for RPG rulebooks, world lore, character backstories, and game mechanics references.

---

## 📺 OBS Setup

1. Add **Browser Source** in OBS
2. Set URL to: `file:///C:/path/to/ai-vtuber/streaming_stage.html`
3. Set dimensions: 1920x1080 (or your stream resolution)
4. Enable: "Shutdown source when not visible"
5. The background is transparent — layer your game/content behind it

### Hotkeys
- Press `H` in the stage to hide/show UI elements

---

## 📊 TTS Setup (Piper)

1. Download [Piper](https://github.com/rhasspy/piper/releases)
2. Extract `piper.exe` to the `piper/` folder
3. Download a voice model from [Piper Voices](https://github.com/rhasspy/piper/blob/master/VOICES.md)
4. Place the `.onnx` and `.onnx.json` files in `piper/`
5. Update `Config.VOICE_MODEL` path if needed

To route TTS to a specific audio device (e.g., an OBS virtual audio cable), use `!audio devices` to find the device index and set `AUDIO_OUTPUT_DEVICE` in Config or use `!audio device <index>` at runtime.

---

## 🎤 Voice Input Setup (Whisper)

```bash
# openai-whisper (easier, CPU/GPU)
pip install openai-whisper sounddevice numpy

# OR faster-whisper (GPU optimized)
pip install faster-whisper sounddevice numpy
```

Also requires [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

The system auto-detects which backend is installed. Test with `!voice test` after starting the AI.

---

## 🤖 Recommended LLM Models

| Model | Size | Notes |
|-------|------|-------|
| Qwen 2.5 3B Instruct | ~2GB | Best balance of speed/quality |
| Phi-3.5 Mini | ~2.5GB | Good instruction following |
| Gemma 2 2B | ~1.5GB | Fast, lightweight |
| Llama 3.2 3B | ~2GB | Good general purpose |

For embedding (vector database), LM Studio can also load embedding models, or use the default HuggingFace model (`all-MiniLM-L6-v2`).

---

## 🎬 Animations

Place VRMA animation files in the appropriate emotion folders, or use `animation_organizer.py` to batch-sort them:

| Folder | Emotions/Triggers |
|--------|-------------------|
| `idle/` | Default state, neutral |
| `happy/` | Joy, excitement, laughter |
| `sad/` | Sadness, disappointment |
| `angry/` | Anger, frustration |
| `surprised/` | Shock, amazement |
| `talking/` | General speaking gestures |
| `greeting/` | Waves, hellos, intros |
| `general/` | Fallback/uncategorized |

The AI automatically selects animations based on detected emotion in responses.

---

## 🛠 Troubleshooting

### "LM Studio not connected"
- Make sure LM Studio is running with local server enabled
- Check the port matches `LLM_BASE_URL` (default: 1234)

### "No audio output"
- Check Piper is installed in `piper/` folder
- Verify the voice model `.onnx` file exists
- Try `pip install playsound==1.2.2` (specific version required)
- Use `!audio devices` to check which device TTS is routing to

### "WebSocket not connecting"
- Make sure the Python script is running
- Check firewall isn't blocking port 8765
- Refresh the browser/OBS source

### "Voice input not working"
- Run `!voice test` to check microphone
- Use `!voice devices` to list available microphones
- Install FFmpeg if using openai-whisper or faster-whisper
- If using faster-whisper on CPU, set `WHISPER_COMPUTE_TYPE = "int8"`

### "Idle chatter won't start"
- Make sure `!stream on` is active before enabling idle chatter
- Verify `.txt` files exist in the `topics/` folder
- Use `!idle topics` to confirm topic files are being detected

### "Peer Link not connecting"
- Ensure host is started first with `!peer on host`
- Check that both machines are on the same network
- Verify port 9876 is not blocked by firewall
- The client should specify the host's local IP: `!peer on client 192.168.x.x`

### "Vector database import is slow"
- This is normal for large documents, especially with semantic chunking
- Use Fixed chunking mode for faster imports
- Press Ctrl+C to cancel gracefully (partial progress is saved)

---

## 📊 Memory System

The AI uses a 3-tier memory system:

| Tier | Storage | Purpose |
|------|---------|---------|
| **1** | `memories/users/<name>.txt` | Per-user conversation history |
| **2** | `memories/stream_highlights.txt` | Notable stream moments |
| **3** | `vector_db/` | Semantic search over imported documents |

The vector database (Tier 3) allows the AI to search through imported documents and find relevant context based on meaning, not just keywords.

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.8 | Feb 2025 | AI-to-AI Peer Link, Whisper buffering, TTS audio locking, audio output device routing |
| v5.7 | Feb 2025 | TTS audio improvements |
| v5.6 | Feb 2025 | Idle chatter system, topics folder, fill dead air |
| v5.5 | Feb 2025 | Kick.com integration, voice input, vector DB manager |
| v5.3 | Feb 2025 | TTS-friendly output cleaning, admin authentication system |
| v5.2 | Feb 2025 | Clearer prompts, admin commands, animation fixes |
| v5.1 | Feb 2025 | Output cleaning, username handling, viseme stop |
| v5.0 | Feb 2025 | Dynamic tokens, chat queue, streaming stage |
| v4.0 | Feb 2025 | Character cards, lorebook, idle return |
| v3.0 | Feb 2025 | Animation system, WebSocket animations |
| v2.0 | Feb 2025 | Memory system, TTS, basic viewer |
| v1.0 | Feb 2025 | Initial prototype |

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---

## 🙏 Credits

- [LM Studio](https://lmstudio.ai/) - Local LLM inference
- [Three.js](https://threejs.org/) - 3D rendering
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [Whisper](https://github.com/openai/whisper) - Speech-to-text
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - GPU-optimized speech-to-text
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://langchain.com/) - LLM framework
- [KickApi](https://pypi.org/project/KickApi/) - Kick.com chat integration

---

Made with ❤️ for the VTuber community
