# AI VTuber Persona System

An AI-powered VTuber system that combines a local LLM with a 3D VRM avatar, text-to-speech, and live chat integration.

![Version](https://img.shields.io/badge/version-5.5-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- **Local LLM Integration** - Uses LM Studio for privacy-focused, offline AI responses
- **3D VRM Avatar** - Animated character with facial expressions and emotions
- **Text-to-Speech** - Natural voice output with lip sync (Piper TTS)
- **Live Chat Integration** - Connects to Kick.com chat for stream interaction
- **Voice Input** - Speech-to-text for collaborations (Whisper)
- **3-Tier Memory System** - Per-user logs, stream highlights, and vector database for long-term semantic memory
- **Emotion Detection** - AI responses include emotion tags for avatar expressions
- **OBS Ready** - Transparent background stage for easy streaming setup

---

## 🚀 Quick Start

### 1. Download & Extract
```bash
git clone https://github.com/yourusername/ai-vtuber.git
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
├── vector_db_manager.py    # Vector database import tool
├── streaming_stage.html    # OBS-ready 3D viewer
├── vrm_viewer_v3.html      # Testing/development viewer
│
├── start.bat               # Launcher menu
├── run.bat                 # Quick start (AI only)
│
├── character.json          # SillyTavern character card
├── lorebook.json           # SillyTavern lorebook
├── requirements.txt        # Python dependencies
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
├── vector_db/              # ChromaDB vector storage
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
```bash
pip install openai-whisper sounddevice numpy
```
Also requires [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

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
pip install openai-whisper sounddevice numpy pymupdf psutil
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

### Chunking Modes Explained

**Fixed Chunking** splits text into equal-sized pieces. Fast but may cut concepts mid-sentence.

**Semantic Chunking** uses AI embeddings to detect topic changes, keeping complete concepts together. For example, an entire spell description stays in one chunk instead of being split. Best for RPG rulebooks and reference material.

**Sentence Chunking** groups sentences together. A middle ground between speed and coherence.

### Example Usage
```
Choice: 2
📁 Opening folder selector...
Selected: C:\Documents\RPG_Rulebooks

🔧 Chunking Mode:
   [1] FIXED    - Fast, fixed-size chunks
   [2] SEMANTIC - Smart topic-aware chunks (slower)
   [3] SENTENCE - Sentence-based chunks

Choice: 2

📂 Processing 15 files with SEMANTIC chunking...
📄 Reading files: |████████████████████| 15/15 (100%) [0.5/s] ETA: 0s
🔢 Embedding:     |██████████░░░░░░░░░░| 500/1000 (50%) [12.3/s] ETA: 41s
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
WHISPER_MODEL_SIZE = "base"  # tiny, base, small, medium, large
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
| `!reload` | Reload character/lorebook |

### Voice Commands
| Command | Description |
|---------|-------------|
| `!voice on/off` | Enable/disable voice input |
| `!voice test` | Test microphone |
| `!voice devices` | List audio devices |
| `!voice ptt` | Push-to-talk (record once) |
| `!voice listen` | Continuous listening (VAD) |
| `!voice stop` | Stop listening |

### Kick Chat Commands
| Command | Description |
|---------|-------------|
| `!kick` | Show connection status |
| `!kick connect` | Connect to your channel |
| `!kick process` | Start auto-responding |
| `!kick stop` | Stop auto-responding |
| `!kick next` | Process one message manually |

---

## 🎭 Character Customization

### Using SillyTavern Cards
Place your character card as `character.json`:
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
Place your lorebook as `lorebook.json` for world-building and lore entries that activate based on keywords.

### Using Vector Memory
Import reference documents with `vector_db_manager.py` for the AI to search during conversations. Great for:
- RPG rulebooks
- World lore documents
- Character backstories
- Game mechanics references

---

## 📺 OBS Setup

1. Add **Browser Source** in OBS
2. Set URL to: `file:///C:/path/to/ai-vtuber/streaming_stage.html`
3. Set dimensions: 1920x1080 (or your stream resolution)
4. Enable: "Shutdown source when not visible"
5. The background is transparent - layer your game/content behind it

### Hotkeys
- Press `H` in the stage to hide/show UI elements

---

## 🔊 TTS Setup (Piper)

1. Download [Piper](https://github.com/rhasspy/piper/releases)
2. Extract `piper.exe` to the `piper/` folder
3. Download a voice model from [Piper Voices](https://github.com/rhasspy/piper/blob/master/VOICES.md)
4. Place the `.onnx` and `.onnx.json` files in `piper/`
5. Update `Config.VOICE_MODEL` path if needed

---

## 🎤 Voice Input Setup (Whisper)

```bash
pip install openai-whisper sounddevice numpy
```

Also requires [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

Test with `!voice test` after starting the AI.

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

Place VRMA animation files in the appropriate emotion folders:

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

### "WebSocket not connecting"
- Make sure the Python script is running
- Check firewall isn't blocking port 8765
- Refresh the browser/OBS source

### "Voice input not working"
- Run `!voice test` to check microphone
- Use `!voice devices` to list available microphones
- Install FFmpeg if using openai-whisper

### "Vector database import is slow"
- This is normal for large documents, especially with semantic chunking
- Use Fixed chunking mode for faster imports
- The progress bar shows ETA - let it run
- Check memory usage with psutil installed

### "Import seems stuck"
- Check the progress bar - if it's updating, it's working
- Large PDFs can take several minutes
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
| v5.5 | Feb 2025 | Kick.com integration, voice input, vector DB manager |
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
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://langchain.com/) - LLM framework

---

Made with ❤️ for the VTuber community
