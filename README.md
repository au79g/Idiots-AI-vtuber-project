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
- **Memory System** - Remembers viewers across sessions
- **Emotion Detection** - AI responses include emotion tags for avatar expressions
- **OBS Ready** - Transparent background stage for easy streaming setup

## 🚀 Quick Start

### 1. Download & Extract
```
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
python ai_persona_v5_5.py
```

## 📁 Project Structure

```
ai-vtuber/
├── ai_persona_v5_5.py      # Main Python script
├── streaming_stage.html    # OBS-ready 3D viewer
├── start.bat               # Launcher with menu
├── run.bat                 # Quick start (AI only)
│
├── character.json          # SillyTavern character card
├── lorebook.json           # SillyTavern lorebook
│
├── animations/             # VRMA animation files
│   ├── idle/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── surprised/
│   ├── talking/
│   └── greeting/
│
├── memories/               # Conversation logs
│   ├── users/              # Per-user history
│   └── voice_sessions/     # Voice transcripts
│
├── scripts/                # Intro/outro scripts
│   ├── intro.txt
│   └── outro.txt
│
├── piper/                  # TTS engine (optional)
│   ├── piper.exe
│   └── en_US-hfc_female-medium.onnx
│
└── vector_db/              # ChromaDB memory storage
```

## ⚙️ Configuration

Edit the `Config` class in `ai_persona_v5_5.py`:

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
| `!emotion <e>` | Test emotion |
| `!anim <name>` | Play animation |

### Voice Commands
| Command | Description |
|---------|-------------|
| `!voice on/off` | Enable/disable voice input |
| `!voice test` | Test microphone |
| `!voice ptt` | Push-to-talk (record once) |
| `!voice listen` | Continuous listening (VAD) |
| `!voice stop` | Stop listening |

### Kick Chat Commands
| Command | Description |
|---------|-------------|
| `!kick` | Show status |
| `!kick connect` | Connect to your channel |
| `!kick process` | Start auto-responding |
| `!kick stop` | Stop |
| `!kick next` | Process one message |

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
Place your lorebook as `lorebook.json` for world-building and lore entries.

## 📺 OBS Setup

1. Add **Browser Source** in OBS
2. Set URL to: `file:///C:/path/to/ai-vtuber/streaming_stage.html`
3. Set dimensions: 1920x1080 (or your stream resolution)
4. Enable: "Shutdown source when not visible"
5. The background is transparent - layer your game/content behind it

### Hotkeys
- Press `H` in the stage to hide/show UI elements

## 🔊 TTS Setup (Piper)

1. Download [Piper](https://github.com/rhasspy/piper/releases)
2. Extract `piper.exe` to the `piper/` folder
3. Download a voice model from [Piper Voices](https://github.com/rhasspy/piper/blob/master/VOICES.md)
4. Place the `.onnx` file in `piper/`
5. Update `Config.VOICE_MODEL` path if needed

## 🎤 Voice Input Setup (Whisper)

```bash
pip install openai-whisper sounddevice numpy
```

Also requires [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH.

## 🤖 Recommended LLM Models

| Model | Size | Notes |
|-------|------|-------|
| Qwen 2.5 3B Instruct | ~2GB | Best balance of speed/quality |
| Phi-3.5 Mini | ~2.5GB | Good instruction following |
| Gemma 2 2B | ~1.5GB | Fast, lightweight |
| Llama 3.2 3B | ~2GB | Good general purpose |

## 🎬 Animations

Place VRMA animation files in the appropriate folders:
- `animations/idle/` - Default/neutral poses
- `animations/happy/` - Joy, excitement
- `animations/sad/` - Sadness, disappointment
- `animations/angry/` - Anger, frustration
- `animations/surprised/` - Shock, surprise
- `animations/talking/` - General speaking gestures
- `animations/greeting/` - Waves, hellos

The AI automatically selects animations based on detected emotion.

## 🐛 Troubleshooting

### "LM Studio not connected"
- Make sure LM Studio is running with local server enabled
- Check the port matches `LLM_BASE_URL` (default: 1234)

### "No audio output"
- Check Piper is installed in `piper/` folder
- Verify the voice model `.onnx` file exists
- Try `pip install playsound==1.2.2` (specific version)

### "WebSocket not connecting"
- Make sure the Python script is running
- Check firewall isn't blocking port 8765
- Refresh the browser/OBS source

### "Voice input not working"
- Run `!voice test` to check microphone
- Use `!voice devices` to list available microphones
- Install FFmpeg if using openai-whisper

## 📝 License

MIT License - Feel free to use, modify, and distribute.

## 🙏 Credits

- [LM Studio](https://lmstudio.ai/) - Local LLM inference
- [Three.js](https://threejs.org/) - 3D rendering
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [Whisper](https://github.com/openai/whisper) - Speech-to-text
- [KickApi](https://github.com/Enmn/KickApi) - Kick.com integration

---

Made with ❤️ for the VTuber community
