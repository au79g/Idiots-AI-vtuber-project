"""
AI VTuber Persona System v5.8
Fixes:
- Clearer role separation in prompts (fixes "You are Schizo Chair" confusion)
- Aggressive output cleaning (removes |>system, role confusion, etc.)
- Admin commands: intro, outro, idle, setidle
- Better animation handling
- v5.3: TTS-friendly output cleaning
  - Removes emotion tag leakage (EMOTION: EXCITED!)
  - Removes asterisk roleplay (*action*)
  - Removes parenthetical actions (sighs)
  - Removes emoji
  - Improved system prompt for TTS clarity
- v5.3: Admin authentication system
  - Admin commands require '!' prefix (e.g., !intro, !outro)
  - Only users in ADMIN_USERS list can run admin commands
  - Prevents chat messages from accidentally triggering commands
  - Runtime admin management with !addadmin
- v5.4: Whisper voice input integration
  - Supports openai-whisper and faster-whisper backends
  - Push-to-talk mode (!voice ptt)
  - VAD continuous listening mode (!voice listen)
  - Auto-pause during TTS to prevent feedback loops
  - Voice session logging
- v5.5: Kick.com chat integration
  - Connect to your Kick channel chat (!kick connect)
  - Polls chat messages and adds to queue
  - Auto-processing with configurable response chance
  - Priority keywords and user cooldowns
  - Commands: !kick connect, !kick poll, !kick process
- v5.6: Idle Chatter system (Fill dead air)
  - When chat is quiet, AI talks about random topics
  - Topics loaded from .txt files in topics/ folder
  - Keeps stream engaging even without viewer interaction
  - Chat always takes priority over rambling
  - Commands: !idle chatter on/off, !idle topics, !idle ramble
- v5.7: TTS Audio Improvements
- v5.8: AI-to-AI Synchronization & Whisper Buffering
  - Peer Link: TCP-based turn-taking between two AI instances
  - Whisper transcription buffering (combines split fragments into one input)
  - Peer state broadcasting (idle/listening/processing/speaking)
  - Prevents AIs from talking over each other on local network
  - TTS Locking: Prevents audio overlap when multiple sources try to speak
  - If TTS is playing, new requests wait in queue (no overlap/interruption)
  - Audio Output Device: Choose which audio device TTS plays through
  - Commands: !audio devices, !audio device <index>
  - Useful for routing audio to OBS virtual cables, etc.
"""

import os
import warnings
import json
import re
import time
import wave
import threading
import asyncio
import subprocess
import tempfile
import platform
import random
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set
from collections import deque
from dataclasses import dataclass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("\n[!] Missing: pip install langchain-huggingface\n")
    raise

try:
    from langchain_chroma import Chroma
except ImportError:
    print("\n[!] Missing: pip install langchain-chroma chromadb\n")
    raise

# ============================================================================
# TTS AUDIO LOCK (Prevents overlapping audio playback)
# ============================================================================
# This lock ensures only one TTS output plays at a time.
# When chat messages and idle chatter both try to speak, 
# the lock makes them wait in queue rather than overlap.

_tts_lock = threading.Lock()
_is_speaking = False  # Flag to check if TTS is currently playing

def is_tts_speaking() -> bool:
    """Check if TTS is currently playing audio"""
    return _is_speaking

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    LLM_BASE_URL = "http://localhost:1234/v1"
    LLM_MODEL_NAME = "your-model-name"
    LLM_TEMPERATURE = 0.8
    
    DEFAULT_USERNAME = "Viewer"
    
    # Admin system - add usernames here to grant admin access
    # These users can run commands like !intro, !outro, !idle, etc.
    ADMIN_USERS = ["au79"]  # Add your username(s) here
    ADMIN_PREFIX = "!"  # Commands must start with this (e.g., !intro)
    
    TOKEN_SETTINGS = {
        (0, 5):     (200, 1.0),
        (6, 20):    (150, 0.8),
        (21, 50):   (100, 0.5),
        (51, 100):  (80, 0.3),
        (101, 500): (60, 0.15),
        (501, 9999): (50, 0.05),
    }
    DEFAULT_MAX_TOKENS = 150
    
    MAX_QUEUE_SIZE = 100
    MAX_MESSAGES_PER_CYCLE = 5
    MESSAGE_COOLDOWN_SECONDS = 30
    
    MEMORY_DIR = Path("./memories")
    USER_LOGS_DIR = MEMORY_DIR / "users"
    HIGHLIGHTS_FILE = MEMORY_DIR / "stream_highlights.txt"
    VECTOR_DB_DIR = Path("./vector_db")
    
    CHARACTER_CARD_FILE = Path("./character.json")
    LOREBOOK_FILE = Path("./lorebook.json")
    
    # Admin script files
    INTRO_FILE = Path("./scripts/intro.txt")
    OUTRO_FILE = Path("./scripts/outro.txt")
    
    MAX_USER_HISTORY_LINES = 10
    MAX_VECTOR_RESULTS = 2
    MAX_HIGHLIGHTS_CHARS = 1000
    MAX_LORE_ENTRIES = 2
    
    PIPER_EXECUTABLE = Path("./piper/piper.exe").resolve()
    VOICE_MODEL = Path("./piper/en_US-hfc_female-medium.onnx").resolve()
    
    # ─────────────────────────────────────────────────
    # TTS AUDIO OUTPUT SETTINGS
    # ─────────────────────────────────────────────────
    # Audio output device index (None = system default)
    # Use !audio devices to list available output devices
    AUDIO_OUTPUT_DEVICE = None  # Set to device index to route audio
    
    # ─────────────────────────────────────────────────
    # VOICE INPUT (Whisper Speech-to-Text)
    # ─────────────────────────────────────────────────
    # Disabled by default - enable with !voice on
    VOICE_INPUT_ENABLED = False
    
    # Two backends supported (auto-detected):
    #   1. openai-whisper: pip install openai-whisper
    #      - More compatible, works on CPU/GPU
    #      - Slower but reliable
    #   2. faster-whisper: pip install faster-whisper  
    #      - GPU optimized (requires CUDA)
    #      - Much faster but harder to install on Windows
    
    # Whisper model size: tiny, base, small, medium, large-v2, large-v3
    # Smaller = faster, larger = more accurate
    # Approximate speed on CPU: tiny ~1x, base ~1x, small ~2x, medium ~5x
    WHISPER_MODEL_SIZE = "base"
    
    # Device/compute settings (only used by faster-whisper backend)
    # openai-whisper auto-detects GPU/CPU
    WHISPER_DEVICE = "cuda"  # "cuda" or "cpu"
    WHISPER_COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" or "float32" for CPU
    
    # Audio input device index (None = system default)
    # Use !voice devices to list available devices
    VOICE_INPUT_DEVICE = None
    
    # Voice activation settings
    VOICE_ACTIVATION_MODE = "push_to_talk"  # "push_to_talk" or "vad" (voice activity detection)
    VOICE_VAD_THRESHOLD = 0.5  # For VAD mode: 0.0-1.0, higher = less sensitive
    VOICE_SILENCE_DURATION = 1.0  # Seconds of silence before processing (VAD mode)
    
    # Recording settings
    VOICE_SAMPLE_RATE = 16000  # Whisper expects 16kHz
    VOICE_RECORD_SECONDS = 10  # Max recording length for push-to-talk
    
    # Who is speaking? (for memory logging)
    VOICE_SPEAKER_NAME = "Collaborator"
    
    # Voice session logging
    VOICE_SESSIONS_DIR = Path("./memories/voice_sessions")
    
    # ─────────────────────────────────────────────────
    # WHISPER TRANSCRIPTION BUFFERING
    # ─────────────────────────────────────────────────
    # When receiving audio from another AI (or a long speaker), Whisper may
    # split a single utterance into multiple transcriptions. This buffer
    # collects them and combines into one input before responding.
    
    VOICE_BUFFER_ENABLED = True   # Enable transcription buffering
    VOICE_BUFFER_TIMEOUT = 2.0    # Seconds of silence before combining & sending
    # Increase this if the other AI's responses are getting split a lot
    # Decrease if you want faster response times
    
    # ─────────────────────────────────────────────────
    # PEER LINK (AI-to-AI Synchronization)
    # ─────────────────────────────────────────────────
    # Allows two AI instances on the same network to coordinate turn-taking.
    # One instance runs as HOST, the other as CLIENT.
    # They exchange state (idle/listening/processing/speaking) so neither
    # talks over the other.
    
    PEER_LINK_ENABLED = False     # Enable with !peer on
    PEER_LINK_PORT = 9876         # TCP port for peer communication
    PEER_LINK_MODE = "host"       # "host" or "client"
    PEER_LINK_HOST = "127.0.0.1"  # IP of the host (client uses this to connect)
    PEER_LINK_NAME = "AI_1"       # Friendly name for this instance
    
    # How long to wait for peer to finish before giving up (seconds)
    PEER_LINK_WAIT_TIMEOUT = 30.0
    # Extra delay after peer finishes speaking before we respond (seconds)
    # This creates a natural conversational pause
    PEER_LINK_TURN_DELAY = 1.0
    
    # ─────────────────────────────────────────────────
    # KICK.COM CHAT INTEGRATION
    # ─────────────────────────────────────────────────
    # Connects to Kick.com chat for live stream integration
    # Install: pip install KickApi
    
    # Your Kick channel username (what appears in kick.com/username)
    KICK_CHANNEL = "your_channel_name"
    
    # Chat polling settings
    KICK_POLL_INTERVAL = 2.0  # Seconds between chat polls (don't set too low)
    
    # Chat processing settings
    KICK_PRIORITY_KEYWORDS = ["@", "?"]  # Messages with these get higher priority
    KICK_IGNORE_USERS = ["bot", "nightbot"]  # Ignore these usernames (lowercase)
    KICK_IGNORE_PREFIXES = ["!", "/"]  # Ignore messages starting with these (commands)
    
    # Rate limiting
    KICK_MIN_RESPONSE_INTERVAL = 3.0  # Minimum seconds between AI responses
    
    WEBSOCKET_PORT = 8765
    ANIMATIONS_DIR = Path("./animations")
    RETURN_TO_IDLE_DELAY = 0.3
    
    # ─────────────────────────────────────────────────
    # IDLE CHATTER (Fill dead air when no chat)
    # ─────────────────────────────────────────────────
    # When enabled, the AI will talk about random topics if chat is quiet
    # This keeps the stream engaging even with no viewer interaction
    
    IDLE_CHATTER_ENABLED = False  # Enable with !idle chatter on
    
    # How long to wait (seconds) with no activity before starting to ramble
    # Shorter = AI starts talking faster when chat is quiet
    IDLE_TIMEOUT_SECONDS = 45.0
    
    # Minimum time between rambles (seconds)
    # This gives the AI time to finish speaking before starting next topic
    IDLE_RAMBLE_INTERVAL = 5.0
    
    # Maximum rambles in a row before picking a new topic file
    # Prevents getting stuck on one topic too long
    IDLE_MAX_RAMBLES_PER_TOPIC = 3
    
    # Token limit for idle chatter (can be different from chat responses)
    IDLE_CHATTER_TOKENS = 200
    
    # Directory containing topic files
    IDLE_TOPICS_DIR = Path("./topics")
    
    # Emotions to use for idle chatter (randomly selected)
    IDLE_EMOTIONS = ['happy', 'playful', 'excited', 'neutral', 'confused']

# ============================================================================
# CHAT QUEUE
# ============================================================================

@dataclass
class ChatMessage:
    id: str
    username: str
    content: str
    timestamp: float
    processed: bool = False
    
    @classmethod
    def create(cls, username: str, content: str) -> 'ChatMessage':
        return cls(
            id=str(uuid.uuid4())[:8],
            username=username,
            content=content,
            timestamp=time.time(),
            processed=False
        )

class ChatQueue:
    def __init__(self):
        self.queue: deque = deque(maxlen=Config.MAX_QUEUE_SIZE)
        self.processed_ids: Set[str] = set()
        self.user_last_response: Dict[str, float] = {}
        self.viewer_count: int = 1
        self._lock = threading.Lock()
        
        # Message rate tracking
        self.message_timestamps: deque = deque(maxlen=500)  # Track last 500 messages
        self.last_mpm_calc: float = 0
        self.cached_mpm: float = 0
        self.last_response_time: float = 0
    
    def set_viewer_count(self, count: int):
        self.viewer_count = max(1, count)
    
    def get_token_settings(self) -> tuple:
        for (min_v, max_v), (tokens, chance) in Config.TOKEN_SETTINGS.items():
            if min_v <= self.viewer_count <= max_v:
                return tokens, chance
        return Config.DEFAULT_MAX_TOKENS, 0.5
    
    def get_queue_status(self) -> dict:
        max_tokens, chance = self.get_token_settings()
        return {
            'viewers': self.viewer_count,
            'tokens': max_tokens,
            'chance': f"{chance*100:.0f}%"
        }
    
    def add_message(self, username: str, content: str) -> Optional[ChatMessage]:
        """Add a message from chat to the queue"""
        # Ignore empty messages
        if not content or not content.strip():
            return None
        
        # Ignore bot users
        if username.lower() in [u.lower() for u in Config.KICK_IGNORE_USERS]:
            return None
        
        # Ignore command messages
        if any(content.strip().startswith(prefix) for prefix in Config.KICK_IGNORE_PREFIXES):
            return None
        
        with self._lock:
            # Check user cooldown
            now = time.time()
            if username in self.user_last_response:
                if now - self.user_last_response[username] < Config.MESSAGE_COOLDOWN_SECONDS:
                    return None  # User on cooldown
            
            # Create and add message
            msg = ChatMessage.create(username, content.strip())
            self.queue.append(msg)
            self.message_timestamps.append(now)
            
            # Reset idle chatter - chat activity detected
            # (idle_chatter may not be initialized yet during startup)
            try:
                idle_chatter.reset_activity()
            except NameError:
                pass
            
            return msg
    
    def get_next_message(self) -> Optional[ChatMessage]:
        """Get a message to respond to (random selection from queue)"""
        with self._lock:
            # Filter unprocessed messages
            unprocessed = [m for m in self.queue if m.id not in self.processed_ids]
            
            if not unprocessed:
                return None
            
            # Check response rate limit
            now = time.time()
            if now - self.last_response_time < Config.KICK_MIN_RESPONSE_INTERVAL:
                return None
            
            # Check response chance based on viewer count
            _, chance = self.get_token_settings()
            if random.random() > chance:
                # Still mark as processed so we don't keep checking
                msg = random.choice(unprocessed)
                self.processed_ids.add(msg.id)
                return None
            
            # Prioritize messages with keywords
            priority_msgs = [m for m in unprocessed 
                          if any(kw.lower() in m.content.lower() for kw in Config.KICK_PRIORITY_KEYWORDS)]
            
            if priority_msgs:
                msg = random.choice(priority_msgs)
            else:
                msg = random.choice(unprocessed)
            
            # Mark as processed
            self.processed_ids.add(msg.id)
            self.user_last_response[msg.username] = now
            self.last_response_time = now
            
            return msg
    
    def get_mpm(self) -> float:
        """Calculate messages per minute (with caching)"""
        now = time.time()
        
        # Use cached value if recent
        if now - self.last_mpm_calc < 60:
            return self.cached_mpm
        
        with self._lock:
            # Count messages in last 5 minutes
            cutoff = now - 300
            recent = [t for t in self.message_timestamps if t > cutoff]
            
            if len(recent) < 2:
                self.cached_mpm = 0
            else:
                span_minutes = (now - min(recent)) / 60
                self.cached_mpm = len(recent) / span_minutes if span_minutes > 0 else 0
            
            self.last_mpm_calc = now
            return self.cached_mpm
    
    def get_pending_count(self) -> int:
        """Get number of unprocessed messages"""
        with self._lock:
            return len([m for m in self.queue if m.id not in self.processed_ids])
    
    def clear(self):
        """Clear the queue"""
        with self._lock:
            self.queue.clear()
            self.processed_ids.clear()

chat_queue = ChatQueue()

# ============================================================================
# TEXT CLEANING
# ============================================================================

def ensure_complete_sentence(text: str) -> str:
    """Ensure response ends with a complete sentence."""
    if not text:
        return text
    
    # Preserve emotion tag
    emotion_match = re.match(r'^\[EMOTION:\s*\w+\]\s*', text)
    emotion_tag = emotion_match.group(0) if emotion_match else ""
    content = text[len(emotion_tag):].strip() if emotion_tag else text.strip()
    
    if not content:
        return text
    
    # Already complete
    if content[-1] in '.!?':
        return emotion_tag + content
    
    # Find last complete sentence
    matches = list(re.finditer(r'[.!?]+(?:\s|$)', content))
    if matches:
        last_match = matches[-1]
        complete = content[:last_match.end()].strip()
        return emotion_tag + complete
    
    # No sentence ending - add period
    return emotion_tag + content + '.'

def clean_llm_response(response: str, username: str, char_name: str) -> str:
    """
    Aggressively clean LLM output to remove:
    - Role labels and markers
    - Model-specific tokens
    - Self-referential confusion
    - Instruction leakage
    """
    text = response
    
    # Remove model-specific tokens/markers (common across many models)
    bad_tokens = [
        r'\|>system\b',
        r'\|>user\b', 
        r'\|>assistant\b',
        r'\|>human\b',
        r'<\|.*?\|>',           # <|anything|>
        r'\[INST\].*?\[/INST\]',
        r'<<SYS>>.*?<</SYS>>',
        r'###\s*(Human|Assistant|System|User):?',
        r'<s>|</s>',
        r'\[/INST\]',
        r'\[INST\]',
    ]
    for pattern in bad_tokens:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove role labels at start of lines
    role_labels = [
        r'^(system|user|assistant|human|ai|bot)\s*:\s*',
        r'\n(system|user|assistant|human|ai|bot)\s*:\s*',
        rf'^{re.escape(char_name)}\s*:\s*',
        rf'\n{re.escape(char_name)}\s*:\s*',
    ]
    for pattern in role_labels:
        text = re.sub(pattern, '\n', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove lines where AI is confused about identity
    # Pattern: "You are [entity/being/etc] [known as] CharName"
    confused_lines = [
        rf'[Yy]ou are (an? )?(entity|being|character|chair|AI)?\s*(known as|called)?\s*{re.escape(char_name)}[^.!?]*[.!?]?',
        rf'[Ii] remember you.*?[Yy]ou are[^.!?]*[.!?]?',
        r'[Yy]ou are an? sentient[^.!?]*[.!?]?',
    ]
    for pattern in confused_lines:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    # Remove any fake user/viewer responses (AI roleplaying both sides)
    fake_user = [
        rf'\n\s*{re.escape(username)}\s*:.*$',
        r'\n\s*(User|Viewer|Human)\s*:.*$',
    ]
    for pattern in fake_user:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Replace username placeholders
    placeholders = ['{{user}}', '{{USER}}', '{user}', '{USER}', '<<user>>', '<user>', '[user]', '{{char}}', '{{CHAR}}']
    for p in placeholders:
        if 'char' in p.lower():
            text = text.replace(p, char_name)
        else:
            text = text.replace(p, username)
    
    # "the user" -> username
    text = re.sub(r'\bthe user\b', username, text, flags=re.IGNORECASE)
    
    # Remove instruction-like lines
    instruction_phrases = [
        'you must', 'you should respond', 'your task is', 'instructions:',
        'valid emotions:', 'emotion tag', '[format]', 'respond as',
        'do not', 'remember to', 'make sure', 'always end', 'keep it brief'
    ]
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(phrase in line_lower for phrase in instruction_phrases):
            continue
        if line.strip():
            clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    
    # =========================================
    # TTS-FRIENDLY CLEANING (v5.3 additions)
    # =========================================
    
    # Remove emotion tags that leaked outside proper format
    # Catches: "EMOTION: EXCITED!" "Emotion: happy" "FEELING: sad" etc.
    text = re.sub(r'(?i)\b(EMOTION|FEELING|MOOD)\s*[:=]\s*\w+[!.]*\s*', '', text)
    
    # Remove bracketed/tagged emotions that aren't our format
    # Catches: [HAPPY], [excited], {angry}, <sad>
    text = re.sub(r'[\[{<][\w\s]+[\]}>]\s*', '', text)
    
    # Remove asterisk roleplay actions: *action here*
    # This prevents TTS from saying "asterisk glows RGB eyes asterisk"
    text = re.sub(r'\*[^*]+\*', '', text)
    
    # Remove parenthetical actions: (sighs), (laughs nervously), etc.
    text = re.sub(r'\([^)]*(?:sigh|laugh|smile|grin|chuckle|nod|wave|shrug|wink|cry|sob|gasp|groan|moan|hum|pause|think|ponder|consider)[^)]*\)', '', text, flags=re.IGNORECASE)
    
    # Remove emoji (TTS will try to say "face with tears of joy" etc.)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    
    # Clean up multiple spaces (from removed content)
    text = re.sub(r' {2,}', ' ', text)
    
    # =========================================
    # END TTS-FRIENDLY CLEANING
    # =========================================
    
    # Clean whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    
    return text

# ============================================================================
# CHARACTER CARD
# ============================================================================

class CharacterCard:
    def __init__(self, file_path: Path = None):
        self.file_path = file_path or Config.CHARACTER_CARD_FILE
        self.name = "AI"
        self.description = ""
        self.personality = ""
        self.scenario = ""
        self._load()
    
    def _load(self):
        if not self.file_path.exists():
            print(f"⚠ Character card not found: {self.file_path}")
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            card_data = data.get('data', data) if 'data' in data else data
            
            self.name = card_data.get('name', self.name)
            self.description = card_data.get('description', '')
            self.personality = card_data.get('personality', '')
            self.scenario = card_data.get('scenario', '')
            
            print(f"✓ Character: {self.name}")
        except Exception as e:
            print(f"⚠ Character card error: {e}")
    
    def reload(self):
        print("🔄 Reloading character...")
        self._load()

# ============================================================================
# LOREBOOK
# ============================================================================

class LorebookEntry:
    def __init__(self, data: dict):
        self.keys = [k.lower() for k in data.get('key', [])]
        self.content = data.get('content', '')
        self.constant = data.get('constant', False)
        self.disabled = data.get('disable', False)

class Lorebook:
    def __init__(self, file_path: Path = None):
        self.file_path = file_path or Config.LOREBOOK_FILE
        self.entries: List[LorebookEntry] = []
        self.constant_entries: List[LorebookEntry] = []
        self._load()
    
    def _load(self):
        self.entries = []
        self.constant_entries = []
        
        if not self.file_path.exists():
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, entry_data in data.get('entries', {}).items():
                entry = LorebookEntry(entry_data)
                if entry.disabled:
                    continue
                if entry.constant:
                    self.constant_entries.append(entry)
                else:
                    self.entries.append(entry)
            
            print(f"✓ Lorebook: {len(self.entries)} entries")
        except Exception as e:
            print(f"⚠ Lorebook error: {e}")
    
    def reload(self):
        print("🔄 Reloading lorebook...")
        self._load()
    
    def get_lore_context(self, text: str) -> str:
        text_lower = text.lower()
        parts = [e.content for e in self.constant_entries]
        
        count = 0
        for entry in self.entries:
            if count >= Config.MAX_LORE_ENTRIES:
                break
            if any(key in text_lower for key in entry.keys):
                parts.append(entry.content)
                count += 1
        
        return '\n'.join(parts) if parts else ""

# ============================================================================
# ANIMATION MANAGER
# ============================================================================

class AnimationManager:
    EMOTION_TO_CATEGORIES = {
        'happy': ['happy', 'greeting', 'talking', 'general'],
        'excited': ['happy', 'surprised', 'greeting', 'general'],
        'sad': ['sad', 'idle', 'general'],
        'tired': ['sad', 'idle', 'general'],
        'angry': ['angry', 'talking', 'general'],
        'surprised': ['surprised', 'happy', 'general'],
        'confused': ['idle', 'talking', 'general'],
        'neutral': ['idle', 'talking', 'general'],
        'sarcastic': ['talking', 'idle', 'general'],
        'playful': ['happy', 'greeting', 'talking', 'general'],
    }
    
    def __init__(self):
        self.animations: Dict[str, List[Path]] = {}
        self.all_animations: List[Path] = []
        self.idle_animation: Optional[Path] = None
        self._scan()
    
    def _scan(self):
        self.animations = {}
        self.all_animations = []
        
        if not Config.ANIMATIONS_DIR.exists():
            Config.ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
            return
        
        for file_path in Config.ANIMATIONS_DIR.rglob("*.vrma"):
            self.all_animations.append(file_path)
            relative = file_path.relative_to(Config.ANIMATIONS_DIR)
            category = relative.parts[0].lower() if len(relative.parts) > 1 else 'general'
            
            if category not in self.animations:
                self.animations[category] = []
            self.animations[category].append(file_path)
        
        # Set default idle
        if 'idle' in self.animations and self.animations['idle']:
            self.idle_animation = self.animations['idle'][0]
        elif self.all_animations:
            self.idle_animation = self.all_animations[0]
        
        print(f"✓ Animations: {len(self.all_animations)}")
    
    def reload(self):
        print("🔄 Reloading animations...")
        self._scan()
    
    def set_idle_animation(self, name: str) -> bool:
        anim = self.get_by_name(name)
        if anim:
            self.idle_animation = anim
            return True
        return False
    
    def get_for_emotion(self, emotion: str) -> Optional[Path]:
        categories = self.EMOTION_TO_CATEGORIES.get(emotion.lower(), ['general', 'idle'])
        for cat in categories:
            if cat in self.animations and self.animations[cat]:
                return random.choice(self.animations[cat])
        return random.choice(self.all_animations) if self.all_animations else None
    
    def get_idle(self) -> Optional[Path]:
        return self.idle_animation
    
    def get_by_name(self, name: str) -> Optional[Path]:
        name_lower = name.lower()
        for p in self.all_animations:
            if name_lower in p.stem.lower():
                return p
        return None
    
    def get_base64(self, file_path: Path) -> Optional[str]:
        try:
            return base64.b64encode(file_path.read_bytes()).decode('utf-8')
        except:
            return None
    
    def list_all(self) -> List[str]:
        return [p.stem for p in self.all_animations]

# ============================================================================
# VOICE INPUT MANAGER (Whisper Speech-to-Text)
# ============================================================================

# Check for voice input dependencies
VOICE_INPUT_AVAILABLE = False
WHISPER_AVAILABLE = False
WHISPER_BACKEND = None  # "openai" or "faster"
SOUNDDEVICE_AVAILABLE = False

# Try to import Whisper - supports multiple backends
# Priority: faster-whisper (GPU optimized) > openai-whisper (more compatible)

# First try faster-whisper (better performance if it works)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    WHISPER_AVAILABLE = True
    WHISPER_BACKEND = "faster"
except Exception:
    # Catches ImportError, FileNotFoundError (ROCm issue), etc.
    pass

# Fall back to openai-whisper if faster-whisper failed
if not WHISPER_AVAILABLE:
    try:
        import whisper as openai_whisper
        WHISPER_AVAILABLE = True
        WHISPER_BACKEND = "openai"
    except Exception:
        pass

# Check for audio recording
try:
    import sounddevice as sd
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    pass

VOICE_INPUT_AVAILABLE = WHISPER_AVAILABLE and SOUNDDEVICE_AVAILABLE

class VoiceInputManager:
    """
    Handles voice input via microphone using Whisper for transcription.
    
    Usage:
        voice_manager = VoiceInputManager()
        voice_manager.start()  # Load model
        
        # Push-to-talk mode:
        text = voice_manager.record_and_transcribe()
        
        # VAD mode:
        voice_manager.start_listening(callback)
    """
    
    def __init__(self):
        self.model = None
        self.is_enabled = False
        self.is_listening = False
        self._listen_thread = None
        self._stop_listening = threading.Event()
        self._callback = None
        self._is_paused = False  # Track if listening is temporarily paused
        
        # Session logging
        self.session_file = None
        
        # Transcription buffer (combines split transcriptions into one)
        self._transcription_buffer = []
        self._buffer_lock = threading.Lock()
        self._buffer_timer = None
        
    def _check_dependencies(self) -> tuple:
        """Check what's available and return status messages"""
        issues = []
        if not WHISPER_AVAILABLE:
            issues.append("Whisper not installed. Try: pip install openai-whisper")
            issues.append("  (or pip install faster-whisper for GPU acceleration)")
        if not SOUNDDEVICE_AVAILABLE:
            issues.append("sounddevice not installed (pip install sounddevice numpy)")
        return len(issues) == 0, issues
    
    def start(self) -> bool:
        """Load the Whisper model. Call this before using voice input."""
        ready, issues = self._check_dependencies()
        if not ready:
            for issue in issues:
                print(f"    ⚠ {issue}")
            return False
        
        if self.model is not None:
            print(f"    ✓ Whisper already loaded ({WHISPER_BACKEND})")
            return True
        
        try:
            device_info = Config.WHISPER_DEVICE if WHISPER_BACKEND == "faster" else "auto"
            print(f"    🔄 Loading Whisper {Config.WHISPER_MODEL_SIZE} ({WHISPER_BACKEND} backend)...")
            
            if WHISPER_BACKEND == "faster":
                # faster-whisper backend
                self.model = FasterWhisperModel(
                    Config.WHISPER_MODEL_SIZE,
                    device=Config.WHISPER_DEVICE,
                    compute_type=Config.WHISPER_COMPUTE_TYPE
                )
            else:
                # openai-whisper backend (auto-detects GPU/CPU)
                self.model = openai_whisper.load_model(Config.WHISPER_MODEL_SIZE)
            
            print(f"    ✓ Whisper ready ({Config.WHISPER_MODEL_SIZE} on {WHISPER_BACKEND})")
            
            # Create session log directory
            Config.VOICE_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"    ❌ Failed to load Whisper: {e}")
            return False
    
    def enable(self) -> bool:
        """Enable voice input"""
        if not self.start():
            return False
        self.is_enabled = True
        
        # Start new session log
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = Config.VOICE_SESSIONS_DIR / f"session_{ts}.txt"
        
        return True
    
    def disable(self):
        """Disable voice input"""
        self.stop_listening()
        self.is_enabled = False
        self.session_file = None
    
    def list_devices(self) -> List[dict]:
        """List available audio input devices"""
        if not SOUNDDEVICE_AVAILABLE:
            return []
        
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:  # Input device
                    devices.append({
                        'index': i,
                        'name': dev['name'],
                        'channels': dev['max_input_channels'],
                        'default': dev.get('default_samplerate', 44100)
                    })
        except:
            pass
        return devices
    
    def set_device(self, device_index: int) -> bool:
        """Set the audio input device"""
        devices = self.list_devices()
        valid_indices = [d['index'] for d in devices]
        if device_index in valid_indices:
            Config.VOICE_INPUT_DEVICE = device_index
            return True
        return False
    
    def record_audio(self, duration: float = None) -> Optional[np.ndarray]:
        """Record audio from microphone"""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        
        duration = duration or Config.VOICE_RECORD_SECONDS
        
        try:
            print(f"    🎤 Recording ({duration}s)... ", end="", flush=True)
            audio = sd.rec(
                int(duration * Config.VOICE_SAMPLE_RATE),
                samplerate=Config.VOICE_SAMPLE_RATE,
                channels=1,
                dtype='float32',
                device=Config.VOICE_INPUT_DEVICE
            )
            sd.wait()
            print("done")
            return audio.flatten()
        except Exception as e:
            print(f"failed: {e}")
            return None
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper (supports both backends)"""
        if self.model is None or audio is None:
            return None
        
        try:
            if WHISPER_BACKEND == "faster":
                # faster-whisper backend
                segments, info = self.model.transcribe(
                    audio,
                    beam_size=5,
                    language="en",
                    vad_filter=True,  # Filter out non-speech
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
                text = " ".join([seg.text for seg in segments]).strip()
            else:
                # openai-whisper backend
                # Needs to save to temp file or use numpy array directly
                # openai-whisper can take numpy array if it's float32 [-1, 1]
                result = self.model.transcribe(
                    audio,
                    language="en",
                    fp16=False  # Use float32 for CPU compatibility
                )
                text = result.get("text", "").strip()
            
            return text if text else None
            
        except Exception as e:
            print(f"    ❌ Transcription error: {e}")
            return None
    
    def record_and_transcribe(self, duration: float = None) -> Optional[str]:
        """Record audio and transcribe it (push-to-talk mode)"""
        if not self.is_enabled:
            print("    ⚠ Voice input not enabled. Use !voice on")
            return None
        
        audio = self.record_audio(duration)
        if audio is None:
            return None
        
        print("    🔄 Transcribing... ", end="", flush=True)
        text = self.transcribe(audio)
        
        if text:
            print(f"'{text}'")
            self._log_transcription(text)
        else:
            print("(no speech detected)")
        
        return text
    
    def _log_transcription(self, text: str):
        """Log transcription to session file"""
        if self.session_file:
            try:
                ts = datetime.now().strftime("%H:%M:%S")
                with open(self.session_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}] {Config.VOICE_SPEAKER_NAME}: {text}\n")
            except:
                pass
    
    # ─────────────────────────────────────────────────
    # VAD (Voice Activity Detection) Continuous Listening
    # ─────────────────────────────────────────────────
    
    def start_listening(self, callback):
        """
        Start continuous listening with VAD.
        callback(text) is called when speech is detected and transcribed.
        """
        if not self.is_enabled:
            print("    ⚠ Voice input not enabled. Use !voice on")
            return False
        
        if self.is_listening:
            print("    ⚠ Already listening")
            return False
        
        self._callback = callback
        self._stop_listening.clear()
        self.is_listening = True
        
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        
        print(f"    🎤 Listening... (VAD threshold: {Config.VOICE_VAD_THRESHOLD})")
        return True
    
    def stop_listening(self):
        """Stop continuous listening"""
        if self.is_listening or self._is_paused:
            self._stop_listening.set()
            self.is_listening = False
            self._is_paused = False
            if self._listen_thread:
                self._listen_thread.join(timeout=2.0)
            print("    🔇 Stopped listening")
    
    def pause_listening(self):
        """Temporarily pause listening (e.g., during TTS playback)"""
        if self.is_listening:
            self._stop_listening.set()
            self.is_listening = False
            self._is_paused = True
            if self._listen_thread:
                self._listen_thread.join(timeout=2.0)
            # Don't print anything - this is a temporary internal pause
    
    def resume_listening(self):
        """Resume listening after a pause"""
        if getattr(self, '_is_paused', False) and self._callback:
            self._is_paused = False
            self._stop_listening.clear()
            self.is_listening = True
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
            # Don't print anything - seamless resume
    
    def _listen_loop(self):
        """Background thread for VAD listening"""
        if not SOUNDDEVICE_AVAILABLE:
            return
        
        import numpy as np
        
        chunk_duration = 0.5  # seconds per chunk
        chunk_samples = int(Config.VOICE_SAMPLE_RATE * chunk_duration)
        silence_chunks_needed = int(Config.VOICE_SILENCE_DURATION / chunk_duration)
        
        audio_buffer = []
        silence_counter = 0
        is_speaking = False
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_counter, is_speaking
            
            if self._stop_listening.is_set():
                raise sd.CallbackAbort()
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(indata**2))
            is_voice = rms > Config.VOICE_VAD_THRESHOLD * 0.1  # Scale threshold
            
            if is_voice:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = []
                audio_buffer.append(indata.copy())
                silence_counter = 0
            elif is_speaking:
                audio_buffer.append(indata.copy())
                silence_counter += 1
                
                if silence_counter >= silence_chunks_needed:
                    # Speech ended, process it
                    is_speaking = False
                    if len(audio_buffer) > 2:  # At least 1 second of audio
                        full_audio = np.concatenate(audio_buffer).flatten()
                        # Process in separate thread to not block audio
                        threading.Thread(
                            target=self._process_vad_audio,
                            args=(full_audio,),
                            daemon=True
                        ).start()
                    audio_buffer = []
                    silence_counter = 0
        
        try:
            with sd.InputStream(
                samplerate=Config.VOICE_SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=chunk_samples,
                device=Config.VOICE_INPUT_DEVICE,
                callback=audio_callback
            ):
                while not self._stop_listening.is_set():
                    time.sleep(0.1)
        except Exception as e:
            if not self._stop_listening.is_set():
                print(f"    ❌ Listening error: {e}")
        
        self.is_listening = False
    
    def _process_vad_audio(self, audio: np.ndarray):
        """Process audio detected by VAD - buffers transcriptions before sending"""
        text = self.transcribe(audio)
        if text and self._callback:
            self._log_transcription(text)
            
            if Config.VOICE_BUFFER_ENABLED:
                # Add to buffer and reset the flush timer
                self._buffer_transcription(text)
            else:
                # No buffering - send immediately (original behavior)
                self._callback(text)
    
    def _buffer_transcription(self, text: str):
        """
        Add transcription to buffer and reset the flush timer.
        
        When Whisper splits a single utterance into multiple transcriptions
        (common when receiving TTS audio from another AI), this collects
        all fragments and combines them into one input after a quiet period.
        """
        with self._buffer_lock:
            self._transcription_buffer.append(text)
            print(f"    📝 Buffered: '{text}' ({len(self._transcription_buffer)} fragment{'s' if len(self._transcription_buffer) > 1 else ''})")
            
            # Cancel existing timer
            if self._buffer_timer is not None:
                self._buffer_timer.cancel()
            
            # Start new timer - when it fires, flush the buffer
            self._buffer_timer = threading.Timer(
                Config.VOICE_BUFFER_TIMEOUT,
                self._flush_buffer
            )
            self._buffer_timer.daemon = True
            self._buffer_timer.start()
    
    def _flush_buffer(self):
        """
        Flush the transcription buffer - combines all fragments and sends
        as a single input to the callback.
        """
        with self._buffer_lock:
            if not self._transcription_buffer:
                return
            
            # Combine all buffered transcriptions
            combined = " ".join(self._transcription_buffer)
            fragment_count = len(self._transcription_buffer)
            self._transcription_buffer.clear()
            self._buffer_timer = None
        
        if fragment_count > 1:
            print(f"    📦 Combined {fragment_count} fragments: '{combined}'")
        
        # Send the combined text to the callback
        if self._callback and combined.strip():
            self._callback(combined.strip())
    
    def clear_buffer(self):
        """Clear the transcription buffer without sending"""
        with self._buffer_lock:
            self._transcription_buffer.clear()
            if self._buffer_timer is not None:
                self._buffer_timer.cancel()
                self._buffer_timer = None
    
    def test(self) -> bool:
        """Test voice input with a short recording"""
        if not self.start():
            return False
        
        # Temporarily enable for test
        was_enabled = self.is_enabled
        self.is_enabled = True
        
        print("    🎤 Testing voice input...")
        print("    Speak now (3 seconds)...")
        
        text = self.record_and_transcribe(duration=3.0)
        
        self.is_enabled = was_enabled
        
        if text:
            print(f"    ✓ Test successful: \"{text}\"")
            return True
        else:
            print("    ⚠ No speech detected. Check your microphone.")
            return False

# Create global voice manager instance
voice_manager = VoiceInputManager()

# ============================================================================
# PEER LINK MANAGER (AI-to-AI Synchronization)
# ============================================================================
# Allows two AI instances to coordinate turn-taking over a local network.
# Uses a simple TCP connection where both sides broadcast their state.
#
# States:
#   IDLE       - Not doing anything, ready to listen/respond
#   LISTENING  - Whisper is actively capturing audio
#   PROCESSING - LLM is generating a response
#   SPEAKING   - TTS is playing audio output
#
# Usage:
#   Instance 1 (host):  !peer on host
#   Instance 2 (client): !peer on client <host_ip>
#
# When one AI is SPEAKING or PROCESSING, the other will buffer its
# voice input and wait until the peer returns to IDLE before responding.

import socket

class PeerLinkManager:
    """
    Manages AI-to-AI synchronization for turn-taking coordination.
    
    Two instances connect via TCP and exchange state updates.
    This prevents them from talking over each other.
    """
    
    # Possible states
    STATE_IDLE = "idle"
    STATE_LISTENING = "listening"
    STATE_PROCESSING = "processing"
    STATE_SPEAKING = "speaking"
    
    def __init__(self):
        self.is_enabled = False
        self.is_connected = False
        self.mode = None  # "host" or "client"
        self.name = Config.PEER_LINK_NAME
        
        # Connection
        self._server_socket = None
        self._peer_socket = None
        self._recv_thread = None
        self._accept_thread = None
        self._stop_event = threading.Event()
        
        # State tracking
        self._my_state = self.STATE_IDLE
        self._peer_state = self.STATE_IDLE
        self._peer_name = "Unknown"
        self._state_lock = threading.Lock()
        
        # Turn coordination
        self._peer_idle_event = threading.Event()
        self._peer_idle_event.set()  # Assume peer is idle initially
        
        # Callbacks for state change notifications
        self._on_peer_state_change = None
    
    @property
    def my_state(self):
        return self._my_state
    
    @property
    def peer_state(self):
        return self._peer_state
    
    @property
    def peer_name(self):
        return self._peer_name
    
    def is_peer_busy(self) -> bool:
        """Check if peer is currently speaking or processing"""
        if not self.is_enabled or not self.is_connected:
            return False
        return self._peer_state in (self.STATE_SPEAKING, self.STATE_PROCESSING)
    
    def wait_for_peer_idle(self, timeout: float = None) -> bool:
        """
        Block until peer is idle. Returns True if peer became idle,
        False if timeout was reached.
        """
        if not self.is_enabled or not self.is_connected:
            return True
        
        if not self.is_peer_busy():
            return True
        
        timeout = timeout or Config.PEER_LINK_WAIT_TIMEOUT
        print(f"    ⏳ Waiting for {self._peer_name} to finish ({self._peer_state})...")
        result = self._peer_idle_event.wait(timeout=timeout)
        
        if result:
            # Add natural pause after peer finishes
            time.sleep(Config.PEER_LINK_TURN_DELAY)
        else:
            print(f"    ⚠ Timed out waiting for {self._peer_name}")
        
        return result
    
    def set_state(self, state: str):
        """Update our state and broadcast to peer"""
        with self._state_lock:
            if state == self._my_state:
                return
            self._my_state = state
        
        self._send_state_update()
    
    def start_host(self, port: int = None) -> bool:
        """Start as host (listens for incoming connection)"""
        port = port or Config.PEER_LINK_PORT
        
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(1.0)
            self._server_socket.bind(('0.0.0.0', port))
            self._server_socket.listen(1)
            
            self.mode = "host"
            self.is_enabled = True
            self._stop_event.clear()
            
            # Start accept thread
            self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self._accept_thread.start()
            
            print(f"    ✓ Peer Link: Hosting on port {port}")
            print(f"    ⏳ Waiting for peer to connect...")
            return True
            
        except Exception as e:
            print(f"    ❌ Peer Link host failed: {e}")
            return False
    
    def start_client(self, host: str = None, port: int = None) -> bool:
        """Start as client (connects to host)"""
        host = host or Config.PEER_LINK_HOST
        port = port or Config.PEER_LINK_PORT
        
        try:
            self._peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._peer_socket.settimeout(5.0)
            self._peer_socket.connect((host, port))
            self._peer_socket.settimeout(None)
            
            self.mode = "client"
            self.is_enabled = True
            self.is_connected = True
            self._stop_event.clear()
            
            # Start receive thread
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            
            # Send hello
            self._send_message({"type": "hello", "name": self.name, "state": self._my_state})
            
            print(f"    ✓ Peer Link: Connected to {host}:{port}")
            return True
            
        except Exception as e:
            print(f"    ❌ Peer Link connection failed: {e}")
            print(f"      Make sure the other AI has !peer on host running")
            return False
    
    def stop(self):
        """Disconnect and stop peer link"""
        self._stop_event.set()
        self.is_enabled = False
        self.is_connected = False
        
        # Send goodbye
        try:
            self._send_message({"type": "goodbye", "name": self.name})
        except:
            pass
        
        # Close sockets
        for sock in [self._peer_socket, self._server_socket]:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        self._peer_socket = None
        self._server_socket = None
        self._peer_state = self.STATE_IDLE
        self._peer_idle_event.set()
        
        print("    🔌 Peer Link: Disconnected")
    
    def get_status(self) -> dict:
        """Get current peer link status"""
        return {
            'enabled': self.is_enabled,
            'connected': self.is_connected,
            'mode': self.mode,
            'my_name': self.name,
            'my_state': self._my_state,
            'peer_name': self._peer_name,
            'peer_state': self._peer_state,
        }
    
    # ─── Internal Methods ────────────────────────────
    
    def _accept_loop(self):
        """Host: wait for client connection"""
        while not self._stop_event.is_set():
            try:
                conn, addr = self._server_socket.accept()
                self._peer_socket = conn
                self.is_connected = True
                
                print(f"\n    ✓ Peer connected from {addr[0]}:{addr[1]}")
                
                # Send hello
                self._send_message({"type": "hello", "name": self.name, "state": self._my_state})
                
                # Start receive thread
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                
                return  # Only accept one peer
                
            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"    ❌ Accept error: {e}")
                return
    
    def _recv_loop(self):
        """Receive messages from peer"""
        buffer = ""
        
        while not self._stop_event.is_set():
            try:
                if not self._peer_socket:
                    break
                
                self._peer_socket.settimeout(1.0)
                
                try:
                    data = self._peer_socket.recv(4096)
                except socket.timeout:
                    continue
                
                if not data:
                    # Peer disconnected
                    print(f"\n    ⚠ Peer {self._peer_name} disconnected")
                    self.is_connected = False
                    self._peer_state = self.STATE_IDLE
                    self._peer_idle_event.set()
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages (newline-delimited JSON)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            self._handle_message(msg)
                        except json.JSONDecodeError:
                            pass
                            
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"    ❌ Peer recv error: {e}")
                break
        
        self.is_connected = False
    
    def _handle_message(self, msg: dict):
        """Handle incoming peer message"""
        msg_type = msg.get("type", "")
        
        if msg_type == "hello":
            self._peer_name = msg.get("name", "Unknown")
            peer_state = msg.get("state", self.STATE_IDLE)
            self._update_peer_state(peer_state)
            print(f"    🤝 Peer Link: {self._peer_name} connected (state: {peer_state})")
            
        elif msg_type == "state":
            new_state = msg.get("state", self.STATE_IDLE)
            self._update_peer_state(new_state)
            
        elif msg_type == "goodbye":
            print(f"\n    👋 {self._peer_name} disconnected")
            self.is_connected = False
            self._peer_state = self.STATE_IDLE
            self._peer_idle_event.set()
    
    def _update_peer_state(self, new_state: str):
        """Update tracked peer state and manage coordination events"""
        old_state = self._peer_state
        self._peer_state = new_state
        
        if new_state == self.STATE_IDLE:
            self._peer_idle_event.set()
        else:
            self._peer_idle_event.clear()
        
        # Log state transitions (only meaningful ones)
        if old_state != new_state:
            state_icons = {
                self.STATE_IDLE: "💤",
                self.STATE_LISTENING: "👂",
                self.STATE_PROCESSING: "🤔",
                self.STATE_SPEAKING: "🔊",
            }
            icon = state_icons.get(new_state, "❓")
            # Only print speaking/idle transitions to avoid spam
            if new_state in (self.STATE_SPEAKING, self.STATE_IDLE) and old_state != new_state:
                status = f"{icon} {self._peer_name}: {new_state}"
                print(f"    [Peer] {status}")
        
        # Notify callback if set
        if self._on_peer_state_change:
            try:
                self._on_peer_state_change(old_state, new_state)
            except:
                pass
    
    def _send_message(self, msg: dict):
        """Send a message to the peer"""
        if not self._peer_socket or not self.is_connected:
            return
        
        try:
            data = json.dumps(msg) + '\n'
            self._peer_socket.sendall(data.encode('utf-8'))
        except Exception:
            pass  # Don't crash on send failures
    
    def _send_state_update(self):
        """Broadcast current state to peer"""
        self._send_message({
            "type": "state",
            "name": self.name,
            "state": self._my_state
        })

# Create global peer link manager
peer_link = PeerLinkManager()

# ============================================================================
# KICK.COM CHAT MANAGER
# ============================================================================

# Check for KickApi dependency
KICK_AVAILABLE = False

try:
    from kickapi import KickAPI
    KICK_AVAILABLE = True
except Exception:
    pass

class KickChatManager:
    """
    Manages Kick.com chat integration using the KickApi package.
    Polls chat messages and feeds them to ChatQueue.
    
    Usage:
        kick_manager.connect("channel_name")  # Start connection
        kick_manager.disconnect()  # Stop
    """
    
    def __init__(self):
        self.api = None
        self.channel = None
        self.channel_id = None
        self.is_connected = False
        self.is_polling = False
        self._poll_thread = None
        self._stop_event = threading.Event()
        self._seen_messages = set()  # Track seen message IDs to avoid duplicates
        self._last_poll_time = None
    
    def _check_dependencies(self) -> tuple:
        """Check what's available"""
        issues = []
        if not KICK_AVAILABLE:
            issues.append("KickApi not installed. Run: pip install KickApi")
        return len(issues) == 0, issues
    
    def connect(self, channel_name: str = None) -> bool:
        """Connect to a Kick channel"""
        ready, issues = self._check_dependencies()
        if not ready:
            for issue in issues:
                print(f"    ⚠ {issue}")
            return False
        
        if self.is_connected:
            print(f"    ✓ Already connected to {self.channel.username if self.channel else 'unknown'}")
            return True
        
        channel_name = channel_name or Config.KICK_CHANNEL
        
        try:
            print(f"    🔄 Connecting to Kick channel: {channel_name}...")
            self.api = KickAPI()
            self.channel = self.api.channel(channel_name)
            self.channel_id = self.channel.id
            
            self.is_connected = True
            print(f"    ✓ Connected to Kick!")
            print(f"      Channel: {self.channel.username}")
            print(f"      Followers: {self.channel.followers}")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Failed to connect: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Kick"""
        self.stop_polling()
        self.is_connected = False
        self.channel = None
        self.channel_id = None
        self._seen_messages.clear()
        print("    📴 Disconnected from Kick")
    
    def start_polling(self) -> bool:
        """Start polling for chat messages"""
        if not self.is_connected:
            print("    ⚠ Not connected. Use '!kick connect' first")
            return False
        
        if self.is_polling:
            print("    ⚠ Already polling")
            return False
        
        self._stop_event.clear()
        self.is_polling = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        
        print(f"    📡 Polling chat every {Config.KICK_POLL_INTERVAL}s...")
        return True
    
    def stop_polling(self):
        """Stop polling for chat messages"""
        if self.is_polling:
            self._stop_event.set()
            self.is_polling = False
            if self._poll_thread:
                self._poll_thread.join(timeout=3.0)
            print("    ⏹ Stopped polling")
    
    def _poll_loop(self):
        """Background thread that polls chat messages"""
        from datetime import datetime, timedelta
        
        # Start from now
        poll_time = datetime.utcnow()
        
        while not self._stop_event.is_set():
            try:
                # Format time for Kick API
                time_str = poll_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                
                # Fetch chat messages
                chat = self.api.chat(self.channel_id, time_str)
                
                if chat and hasattr(chat, 'messages'):
                    for message in chat.messages:
                        # Create unique ID for deduplication
                        msg_id = f"{message.sender.username}:{message.text[:50]}:{getattr(message, 'created_at', '')}"
                        
                        if msg_id not in self._seen_messages:
                            self._seen_messages.add(msg_id)
                            
                            username = message.sender.username
                            content = message.text
                            
                            # Add to queue
                            msg = chat_queue.add_message(username, content)
                            
                            if msg:
                                # Show in console
                                display = content[:50] + "..." if len(content) > 50 else content
                                print(f"    💬 {username}: {display}")
                    
                    # Keep seen messages set from growing too large
                    if len(self._seen_messages) > 1000:
                        # Remove oldest half
                        self._seen_messages = set(list(self._seen_messages)[-500:])
                
                # Update poll time for next iteration
                poll_time = datetime.utcnow()
                
                # Wait before next poll
                time.sleep(Config.KICK_POLL_INTERVAL)
                
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"    ⚠ Poll error: {e}")
                time.sleep(Config.KICK_POLL_INTERVAL * 2)  # Back off on error
        
        self.is_polling = False
    
    def get_status(self) -> dict:
        """Get connection status"""
        return {
            'available': KICK_AVAILABLE,
            'connected': self.is_connected,
            'polling': self.is_polling,
            'channel': self.channel.username if self.channel else None,
            'channel_id': self.channel_id,
            'followers': self.channel.followers if self.channel else 0,
            'pending_messages': chat_queue.get_pending_count(),
            'mpm': chat_queue.get_mpm()
        }

# Create global Kick manager instance
kick_manager = KickChatManager()

# ============================================================================
# CHAT PROCESSING LOOP
# ============================================================================

class ChatProcessor:
    """
    Background processor that handles queued chat messages.
    Runs in a separate thread, picks messages from queue, and generates responses.
    """
    
    def __init__(self):
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self.is_streaming = False
    
    def start(self):
        """Start the chat processor"""
        if self.is_running:
            return
        
        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("    ✓ Chat processor started")
    
    def stop(self):
        """Stop the chat processor"""
        if not self.is_running:
            return
        
        self._stop_event.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("    ⏹ Chat processor stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        while not self._stop_event.is_set():
            try:
                # Wait if TTS is currently playing (don't process during playback)
                if is_tts_speaking():
                    time.sleep(0.5)
                    continue
                
                # Wait if peer is speaking/processing (don't talk over them)
                if peer_link.is_peer_busy():
                    time.sleep(0.5)
                    continue
                
                # Get next message to respond to
                msg = chat_queue.get_next_message()
                
                if msg:
                    # Reset idle chatter (chat takes priority)
                    idle_chatter.reset_activity()
                    
                    # Wait for peer to be idle before responding
                    peer_link.wait_for_peer_idle()
                    
                    # Broadcast processing state
                    peer_link.set_state(PeerLinkManager.STATE_PROCESSING)
                    
                    # Process the message
                    print(f"\n    📨 Processing: {msg.username}: {msg.content[:40]}...")
                    print("    🤔...")
                    
                    response = process_message(msg.username, msg.content, self.is_streaming)
                    
                    if response:
                        display_text = clean_response(response)
                        print(f"\n{character.name}: {display_text}")
                        speak_text(response, extract_emotion(response))
                    else:
                        peer_link.set_state(PeerLinkManager.STATE_IDLE)
                    
                    # Reset activity again after responding
                    idle_chatter.reset_activity()
                
                # Small delay between checks
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Chat processor error: {e}")
                peer_link.set_state(PeerLinkManager.STATE_IDLE)
                time.sleep(1)

# Create global chat processor instance
chat_processor = ChatProcessor()

# ============================================================================
# IDLE CHATTER SYSTEM
# ============================================================================

class IdleChatterManager:
    """
    Manages autonomous "rambling" when chat is quiet.
    
    When no chat activity occurs for IDLE_TIMEOUT_SECONDS, the AI will:
    1. Pick a random topic file from the topics/ folder
    2. Generate a response about that topic
    3. Continue rambling until chat activity resumes
    
    Topic files are simple .txt files containing prompts/topics, one per line.
    Example: "Talk about your favorite video game and why you love it"
    """
    
    def __init__(self):
        self.is_enabled = Config.IDLE_CHATTER_ENABLED
        self.is_rambling = False
        self._stop_event = threading.Event()
        self._thread = None
        
        # Activity tracking
        self.last_activity_time = time.time()
        self._activity_lock = threading.Lock()
        
        # Topic management
        self.current_topic_file = None
        self.current_topics = []
        self.rambles_on_current_topic = 0
        self.used_prompts = set()  # Track used prompts to avoid immediate repeats
        
        # State
        self.is_streaming = False
        
        # Ensure topics directory exists
        Config.IDLE_TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    
    def enable(self):
        """Enable idle chatter"""
        self.is_enabled = True
        Config.IDLE_CHATTER_ENABLED = True
        self.reset_activity()
        print("    ✓ Idle chatter enabled")
    
    def disable(self):
        """Disable idle chatter"""
        self.is_enabled = False
        Config.IDLE_CHATTER_ENABLED = False
        self.stop()
        print("    ✓ Idle chatter disabled")
    
    def reset_activity(self):
        """Reset the activity timer (call when chat message received)"""
        with self._activity_lock:
            self.last_activity_time = time.time()
            # If we were rambling, stop
            if self.is_rambling:
                self.is_rambling = False
    
    def get_idle_duration(self) -> float:
        """Get seconds since last activity"""
        with self._activity_lock:
            return time.time() - self.last_activity_time
    
    def should_ramble(self) -> bool:
        """Check if we should start/continue rambling"""
        if not self.is_enabled:
            return False
        if not self.is_streaming:
            return False
        # Check if we've been idle long enough
        return self.get_idle_duration() >= Config.IDLE_TIMEOUT_SECONDS
    
    def get_topic_files(self) -> List[Path]:
        """Get all topic files from the topics directory"""
        if not Config.IDLE_TOPICS_DIR.exists():
            return []
        
        files = []
        for ext in ['*.txt', '*.md']:
            files.extend(Config.IDLE_TOPICS_DIR.glob(ext))
        return sorted(files)
    
    def load_topic_file(self, file_path: Path) -> List[str]:
        """Load prompts from a topic file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            # Split by lines, filter empty lines and comments
            prompts = []
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
            return prompts
        except Exception as e:
            print(f"    ⚠ Error loading {file_path.name}: {e}")
            return []
    
    def pick_topic_file(self) -> Optional[Path]:
        """Pick a random topic file"""
        files = self.get_topic_files()
        if not files:
            return None
        return random.choice(files)
    
    def get_ramble_prompt(self) -> Optional[str]:
        """Get a prompt to ramble about"""
        
        # Check if we need a new topic file
        need_new_file = (
            self.current_topic_file is None or
            not self.current_topics or
            self.rambles_on_current_topic >= Config.IDLE_MAX_RAMBLES_PER_TOPIC
        )
        
        if need_new_file:
            # Pick a new topic file
            self.current_topic_file = self.pick_topic_file()
            if self.current_topic_file:
                self.current_topics = self.load_topic_file(self.current_topic_file)
                self.rambles_on_current_topic = 0
                # Clear used prompts for new file
                self.used_prompts.clear()
        
        if not self.current_topics:
            return None
        
        # Try to pick an unused prompt first
        available = [p for p in self.current_topics if p not in self.used_prompts]
        
        if not available:
            # All prompts used, reset and pick any
            self.used_prompts.clear()
            available = self.current_topics
        
        prompt = random.choice(available)
        self.used_prompts.add(prompt)
        self.rambles_on_current_topic += 1
        
        return prompt
    
    def start(self):
        """Start the idle chatter background thread"""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._ramble_loop, daemon=True)
        self._thread.start()
        print("    ✓ Idle chatter monitor started")
    
    def stop(self):
        """Stop the idle chatter thread"""
        self._stop_event.set()
        self.is_rambling = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _ramble_loop(self):
        """Background loop that checks for idle state and generates rambles"""
        while not self._stop_event.is_set():
            try:
                # Check if we should be rambling
                if self.should_ramble():
                    # Check if TTS is currently playing (don't interrupt)
                    if is_tts_speaking():
                        time.sleep(0.5)
                        continue
                    
                    # Check if peer is busy (don't talk over them)
                    if peer_link.is_peer_busy():
                        time.sleep(0.5)
                        continue
                    
                    # Check if there are chat messages waiting (they take priority)
                    if chat_queue.get_pending_count() > 0:
                        # Chat has priority, reset activity
                        self.reset_activity()
                        time.sleep(1)
                        continue
                    
                    # Get a topic to ramble about
                    prompt = self.get_ramble_prompt()
                    
                    if prompt:
                        self.is_rambling = True
                        
                        # Generate and speak the ramble
                        self._generate_ramble(prompt)
                        
                        # Wait before next ramble
                        time.sleep(Config.IDLE_RAMBLE_INTERVAL)
                    else:
                        # No topics available
                        time.sleep(5)
                else:
                    # Not time to ramble yet, check periodically
                    time.sleep(1)
                    
            except Exception as e:
                print(f"    ❌ Idle chatter error: {e}")
                time.sleep(2)
    
    def _generate_ramble(self, prompt: str):
        """Generate a single ramble response"""
        # Double-check that chat hasn't become active
        if chat_queue.get_pending_count() > 0:
            self.reset_activity()
            return
        
        try:
            # Build a special system prompt for rambling
            char = character.name
            
            emotion = random.choice(Config.IDLE_EMOTIONS)
            
            system_prompt = f"""You are {char}, a VTuber who is currently streaming.

About {char}:
{character.description[:300] if character.description else "A friendly AI VTuber."}

You are currently on stream and chat has been quiet. Fill the dead air by talking about the following topic naturally and engagingly. Speak as if you're sharing your thoughts with your audience.

Keep it conversational and entertaining. Don't ask too many questions - just share your thoughts, tell a story, or ramble about the topic. 

Start your response with [EMOTION: {emotion}]

Topic/Prompt: {prompt}

RULES:
- Speak naturally as {char} in first person
- Be engaging and entertaining
- Don't mention that you're "filling time" or that chat is quiet
- Keep the energy up
- 2-4 sentences is perfect"""

            # Use the idle chatter token limit
            response = get_llm(Config.IDLE_CHATTER_TOKENS).invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Share your thoughts on: {prompt}")
            ]).content
            
            # Clean and speak
            response = clean_llm_response(response, "audience", character.name)
            response = ensure_complete_sentence(response)
            
            # Check one more time for chat activity
            if chat_queue.get_pending_count() > 0:
                self.reset_activity()
                return
            
            display_text = clean_response(response)
            detected_emotion = extract_emotion(response)
            
            # Show what topic inspired this
            topic_name = self.current_topic_file.stem if self.current_topic_file else "unknown"
            print(f"\n    💭 [Idle: {topic_name}]")
            print(f"{character.name}: {display_text}")
            
            speak_text(response, detected_emotion)
            
            # Reset activity time after speaking (to space out rambles)
            # But set it to be just before the threshold so next ramble happens at interval
            with self._activity_lock:
                self.last_activity_time = time.time() - Config.IDLE_TIMEOUT_SECONDS + Config.IDLE_RAMBLE_INTERVAL
                
        except Exception as e:
            print(f"    ❌ Ramble generation error: {e}")
    
    def get_status(self) -> dict:
        """Get status info"""
        topic_files = self.get_topic_files()
        return {
            'enabled': self.is_enabled,
            'rambling': self.is_rambling,
            'idle_seconds': round(self.get_idle_duration(), 1),
            'timeout': Config.IDLE_TIMEOUT_SECONDS,
            'topic_files': len(topic_files),
            'current_topic': self.current_topic_file.stem if self.current_topic_file else None,
            'rambles_on_topic': self.rambles_on_current_topic
        }
    
    def list_topics(self) -> List[str]:
        """List all available topic files"""
        return [f.stem for f in self.get_topic_files()]

# Create global idle chatter instance
idle_chatter = IdleChatterManager()

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠ websockets not installed")

class WebSocketManager:
    def __init__(self):
        self.connections = set()
        self.loop = None
        self.server_thread = None
    
    def start(self):
        if not WEBSOCKETS_AVAILABLE:
            return
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        time.sleep(0.5)
    
    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def handler(websocket):
            self.connections.add(websocket)
            print(f"✓ Viewer connected ({len(self.connections)})")
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get('type') == 'viewer_count':
                            chat_queue.set_viewer_count(data.get('count', 1))
                    except:
                        pass
            except:
                pass
            finally:
                self.connections.discard(websocket)
                print(f"✗ Viewer disconnected ({len(self.connections)})")
        
        async def serve():
            async with websockets.serve(handler, "localhost", Config.WEBSOCKET_PORT):
                print(f"✓ WebSocket: ws://localhost:{Config.WEBSOCKET_PORT}")
                await asyncio.Future()
        
        self.loop.run_until_complete(serve())
    
    def send(self, data: dict) -> bool:
        if not WEBSOCKETS_AVAILABLE or self.loop is None:
            return False
        
        async def broadcast():
            if not self.connections:
                return False
            message = json.dumps(data)
            disconnected = set()
            for ws in self.connections.copy():
                try:
                    await ws.send(message)
                except:
                    disconnected.add(ws)
            self.connections -= disconnected
            return len(self.connections) > 0
        
        try:
            future = asyncio.run_coroutine_threadsafe(broadcast(), self.loop)
            return future.result(timeout=5.0)
        except:
            return False
    
    def send_emotion(self, emotion: str):
        self.send({'type': 'emotion', 'emotion': emotion})
    
    def send_animation(self, file_path: Path) -> bool:
        if not file_path or not file_path.exists():
            return False
        anim_data = anim_manager.get_base64(file_path)
        if not anim_data:
            return False
        return self.send({
            'type': 'animation_file',
            'name': file_path.stem,
            'data': anim_data,
            'play': True
        })
    
    def send_sync_playback(self, emotion: str, anim_path: Optional[Path], 
                           visemes: list, duration: float):
        data = {
            'type': 'synchronized_playback',
            'emotion': emotion,
            'visemes': visemes,
            'audio_duration': duration,
            'animation': None
        }
        if anim_path and anim_path.exists():
            anim_data = anim_manager.get_base64(anim_path)
            if anim_data:
                data['animation'] = {'name': anim_path.stem, 'data': anim_data}
        self.send(data)
    
    def send_stop_visemes(self):
        self.send({'type': 'stop_visemes'})
    
    def send_idle(self):
        idle_anim = anim_manager.get_idle()
        if idle_anim:
            self.send_animation(idle_anim)
        self.send_emotion('neutral')

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

character = CharacterCard()
lorebook = Lorebook()
anim_manager = AnimationManager()
ws_manager = WebSocketManager()
ws_manager.start()

# ============================================================================
# VISEME GENERATION
# ============================================================================

import nltk
try:
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download('cmudict', quiet=True)
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()

try:
    from g2p_en import G2p
    G2P_MODEL = G2p()
except ImportError:
    G2P_MODEL = None

PHONEME_TO_VISEME = {
    'AA': 'aa', 'AO': 'aa', 'AH': 'aa', 'AE': 'aa',
    'IY': 'ih', 'IH': 'ih', 'EY': 'eh', 'EH': 'eh',
    'OW': 'oh', 'OY': 'oh', 'UW': 'oh', 'UH': 'oh',
    'AY': 'aa', 'AW': 'aa', 'ER': 'oh',
    'B': 'PP', 'M': 'PP', 'P': 'PP', 'F': 'FF', 'V': 'FF',
    'T': 'DD', 'D': 'DD', 'N': 'DD', 'L': 'DD',
    'S': 'SS', 'Z': 'SS', 'SH': 'SS', 'ZH': 'SS',
    'CH': 'CH', 'JH': 'CH', 'TH': 'TH', 'DH': 'TH',
    'K': 'kk', 'G': 'kk', 'NG': 'kk',
    'R': 'RR', 'W': 'oh', 'Y': 'ih', 'HH': 'neutral',
}

def text_to_visemes(text: str) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    phonemes = []
    for word in words:
        if word in CMU_DICT:
            phonemes.extend([p.rstrip('012') for p in CMU_DICT[word][0]])
        elif G2P_MODEL:
            try:
                phonemes.extend([p for p in G2P_MODEL(word) if p.isalpha()])
            except:
                pass
        phonemes.append('SIL')
    
    visemes = [PHONEME_TO_VISEME.get(p.upper(), 'neutral') for p in phonemes]
    filtered = [visemes[0]] if visemes else []
    for v in visemes[1:]:
        if v != filtered[-1]:
            filtered.append(v)
    return filtered

# ============================================================================
# MEMORY SYSTEM
# ============================================================================

class MemoryManager:
    def __init__(self):
        Config.USER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        Config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        if not Config.HIGHLIGHTS_FILE.exists():
            Config.HIGHLIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            Config.HIGHLIGHTS_FILE.write_text("STREAM HIGHLIGHTS:\n", encoding='utf-8')
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=str(Config.VECTOR_DB_DIR),
            embedding_function=self.embeddings,
            collection_name="vtuber_memory"
        )
        print(f"✓ Memory ready")
    
    def _get_user_file(self, username: str) -> Path:
        safe_name = re.sub(r'[^\w\-]', '_', username)
        return Config.USER_LOGS_DIR / f"{safe_name}.txt"
    
    def is_known_user(self, username: str) -> bool:
        return self._get_user_file(username).exists()
    
    def get_user_history(self, username: str) -> str:
        file_path = self._get_user_file(username)
        if not file_path.exists():
            return ""
        try:
            lines = file_path.read_text(encoding='utf-8').splitlines()
            return '\n'.join(lines[-Config.MAX_USER_HISTORY_LINES:])
        except:
            return ""
    
    def log_interaction(self, username: str, user_msg: str, ai_msg: str):
        if username == Config.DEFAULT_USERNAME:
            return
        file_path = self._get_user_file(username)
        ts = datetime.now().strftime("%m-%d %H:%M")
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] {username}: {user_msg}\n[{ts}] AI: {ai_msg}\n\n")
            self.vector_db.add_texts([f"{username}: {user_msg} -> AI: {ai_msg}"])
        except:
            pass
    
    def search(self, query: str) -> str:
        try:
            results = self.vector_db.similarity_search(query, k=Config.MAX_VECTOR_RESULTS)
            return '\n'.join([r.page_content for r in results])
        except:
            return ""

memory = MemoryManager()

# ============================================================================
# LLM SETUP
# ============================================================================

VALID_EMOTIONS = ['neutral', 'happy', 'excited', 'sad', 'angry', 'surprised', 
                  'confused', 'sarcastic', 'playful', 'tired']

def get_llm(max_tokens: int):
    return ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        api_key="not-needed",
        model=Config.LLM_MODEL_NAME,
        max_tokens=max_tokens,
        temperature=Config.LLM_TEMPERATURE,
    )

def build_system_prompt(username: str, user_message: str, is_streaming: bool = False) -> str:
    """
    Build a VERY clear system prompt.
    Key: Make it absolutely clear that the AI IS the character.
    """
    
    char = character.name
    
    # Start with unmistakable identity
    prompt = f"""You are {char}. You speak as {char} in first person.

The person chatting with you is named {username}. Talk TO them, not about yourself.

About {char}:
"""
    
    if character.description:
        desc = character.description[:250]
        prompt += f"{desc}\n"
    
    if character.personality:
        prompt += f"Traits: {character.personality[:100]}\n"
    
    # Lore
    lore = lorebook.get_lore_context(user_message)
    if lore:
        prompt += f"\nWorld info:\n{lore[:300]}\n"
    
    # User info
    if memory.is_known_user(username):
        history = memory.get_user_history(username)
        if history:
            prompt += f"\nYou've chatted with {username} before:\n{history[-150:]}\n"
    
    if is_streaming:
        prompt += "\n[Currently live streaming]\n"
    
    # Response format
    prompt += f"""
RESPOND LIKE THIS:
[EMOTION: X] Your response here

Where X is one of: neutral, happy, excited, sad, angry, surprised, confused, playful

Rules:
- Talk directly to {username}
- Use first person (I, me, my)
- 1-3 sentences max
- Complete sentences only
- Your text will be spoken by TTS - write ONLY dialogue

NEVER USE:
- Asterisks for actions (*waves*, *laughs*)
- Parentheses for actions (sighs, smiles)
- Emojis or special characters
- "EMOTION:" outside the bracket format
- Stage directions or narration
- "User:", "system:", role labels

Express emotion through your WORDS and punctuation, not actions."""
    
    return prompt

# ============================================================================
# AUDIO
# ============================================================================

AUDIO_METHOD = None
_pygame_initialized = False

# Try to import audio libraries
try:
    from playsound import playsound
    AUDIO_METHOD = "playsound"
except ImportError:
    pass

# Prefer pygame if available (supports device selection)
try:
    import pygame
    AUDIO_METHOD = "pygame"
except ImportError:
    pass

# Also check for sounddevice (best for device selection)
try:
    import sounddevice as sd_output
    import soundfile as sf_output
    if AUDIO_METHOD != "pygame":  # pygame already handles devices well
        AUDIO_METHOD = "sounddevice"
except ImportError:
    pass

def init_audio_output(device_index: int = None) -> bool:
    """Initialize audio output with optional device selection"""
    global _pygame_initialized
    
    if AUDIO_METHOD == "pygame":
        try:
            import pygame
            pygame.mixer.quit()  # Reset if already initialized
            
            if device_index is not None:
                # pygame doesn't directly support device index, but we can try
                # For proper device selection, use sounddevice
                pygame.mixer.init()
            else:
                pygame.mixer.init()
            
            _pygame_initialized = True
            return True
        except Exception as e:
            print(f"    ⚠ Failed to init pygame audio: {e}")
            return False
    
    return AUDIO_METHOD is not None

def list_audio_output_devices() -> List[dict]:
    """List available audio output devices"""
    devices = []
    
    # Try sounddevice first (most reliable for device listing)
    try:
        import sounddevice as sd_list
        for i, dev in enumerate(sd_list.query_devices()):
            if dev['max_output_channels'] > 0:  # Output device
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_output_channels'],
                    'default': i == sd_list.default.device[1]  # Check if default output
                })
        return devices
    except Exception:
        pass
    
    # Fallback: can't list devices
    return []

def set_audio_output_device(device_index: int) -> bool:
    """Set the audio output device"""
    devices = list_audio_output_devices()
    valid_indices = [d['index'] for d in devices]
    
    if device_index not in valid_indices:
        return False
    
    Config.AUDIO_OUTPUT_DEVICE = device_index
    
    # Reinitialize if using sounddevice
    if AUDIO_METHOD == "sounddevice":
        try:
            import sounddevice as sd_set
            sd_set.default.device = (sd_set.default.device[0], device_index)
            return True
        except:
            pass
    
    return True

def get_audio_duration(wav_path: str) -> float:
    try:
        with wave.open(wav_path, 'rb') as wf:
            return wf.getnframes() / float(wf.getframerate())
    except:
        return 0

def play_audio_file(wav_path: str, duration: float) -> bool:
    """
    Play an audio file through the configured output device.
    Returns True if playback was successful.
    """
    if AUDIO_METHOD == "sounddevice":
        try:
            import sounddevice as sd_play
            import soundfile as sf_play
            
            data, samplerate = sf_play.read(wav_path)
            
            # Use configured device or default
            device = Config.AUDIO_OUTPUT_DEVICE
            
            sd_play.play(data, samplerate, device=device)
            sd_play.wait()  # Wait for playback to finish
            return True
        except Exception as e:
            print(f"    ⚠ sounddevice playback error: {e}")
            time.sleep(duration)
            return False
    
    elif AUDIO_METHOD == "pygame":
        try:
            global _pygame_initialized
            if not _pygame_initialized:
                init_audio_output(Config.AUDIO_OUTPUT_DEVICE)
            
            import pygame
            sound = pygame.mixer.Sound(wav_path)
            sound.play()
            time.sleep(duration + 0.1)
            return True
        except Exception as e:
            print(f"    ⚠ pygame playback error: {e}")
            time.sleep(duration)
            return False
    
    elif AUDIO_METHOD == "playsound":
        try:
            playsound(wav_path)
            return True
        except Exception as e:
            print(f"    ⚠ playsound error: {e}")
            time.sleep(duration)
            return False
    
    else:
        # No audio method available, just wait
        time.sleep(duration)
        return False

def speak_text(text: str, emotion: str = "neutral"):
    """
    TTS with animation. Uses lock to prevent audio overlap.
    Pauses voice listening during playback to prevent feedback loop.
    Broadcasts state to peer link if enabled.
    """
    global _is_speaking
    
    clean_text = re.sub(r'\[EMOTION:\s*\w+\]', '', text).strip()
    if not clean_text:
        return
    
    # Acquire the TTS lock - this ensures only one speak_text runs at a time
    # Other callers will wait here until current TTS finishes
    with _tts_lock:
        _is_speaking = True
        
        # Broadcast SPEAKING state to peer
        peer_link.set_state(PeerLinkManager.STATE_SPEAKING)
        
        try:
            # Pause voice listening during TTS to prevent feedback loop
            was_listening = voice_manager.is_listening
            if was_listening:
                voice_manager.pause_listening()
            
            anim_path = anim_manager.get_for_emotion(emotion) if anim_manager.all_animations else None
            duration = 0
            wav_path = None
            
            if Config.PIPER_EXECUTABLE.exists():
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav_path = f.name
                
                cmd = f'"{Config.PIPER_EXECUTABLE}" --model "{Config.VOICE_MODEL}" --output_file "{wav_path}"'
                
                if platform.system() == "Windows":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
                        tf.write(clean_text)
                        temp_txt = tf.name
                    subprocess.run(f'type "{temp_txt}" | {cmd}', shell=True, capture_output=True)
                    try: os.remove(temp_txt)
                    except: pass
                else:
                    subprocess.run(f'echo "{clean_text}" | {cmd}', shell=True, capture_output=True)
                
                duration = get_audio_duration(wav_path)
            
            if duration == 0:
                duration = len(clean_text) * 0.05
            
            visemes = text_to_visemes(clean_text)
            anim_name = anim_path.stem if anim_path else "none"
            print(f"    🎭 {emotion} | 🎬 {anim_name} | ⏱ {duration:.1f}s")
            
            ws_manager.send_sync_playback(emotion, anim_path, visemes, duration)
            
            if wav_path and duration > 0:
                play_audio_file(wav_path, duration)
                try: os.remove(wav_path)
                except: pass
            else:
                time.sleep(duration)
            
            ws_manager.send_stop_visemes()
            time.sleep(Config.RETURN_TO_IDLE_DELAY)
            ws_manager.send_idle()
            
            # Resume voice listening after TTS completes
            if was_listening:
                # Small delay to ensure audio has fully stopped
                time.sleep(0.5)
                voice_manager.resume_listening()
        
        finally:
            _is_speaking = False
            # Broadcast IDLE state to peer
            peer_link.set_state(PeerLinkManager.STATE_IDLE)

# ============================================================================
# RESPONSE PROCESSING
# ============================================================================

def extract_emotion(response: str) -> str:
    match = re.search(r'\[EMOTION:\s*(\w+)\]', response, re.IGNORECASE)
    if match:
        emotion = match.group(1).lower()
        if emotion in VALID_EMOTIONS:
            return emotion
    return 'neutral'

def clean_response(response: str) -> str:
    return re.sub(r'\[EMOTION:\s*\w+\]\s*', '', response).strip()

def process_message(username: str, message: str, is_streaming: bool = False) -> Optional[str]:
    max_tokens, _ = chat_queue.get_token_settings()
    system_prompt = build_system_prompt(username, message, is_streaming)
    
    vector_context = memory.search(message)
    if vector_context:
        system_prompt += f"\n\nPast memories:\n{vector_context[:100]}"
    
    try:
        response = get_llm(max_tokens).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]).content
    except Exception as e:
        print(f"    ❌ LLM error: {e}")
        return None
    
    response = clean_llm_response(response, username, character.name)
    response = ensure_complete_sentence(response)
    
    display_text = clean_response(response)
    memory.log_interaction(username, message, display_text)
    
    return response

# ============================================================================
# ADMIN COMMANDS
# ============================================================================

def read_script(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return None
    try:
        return file_path.read_text(encoding='utf-8').strip()
    except:
        return None

def cmd_intro():
    script = read_script(Config.INTRO_FILE)
    if script:
        print(f"\n{character.name}: {script}")
        speak_text(f"[EMOTION: excited] {script}", "excited")
    else:
        print(f"    ⚠ Create {Config.INTRO_FILE} with your intro text")

def cmd_outro():
    script = read_script(Config.OUTRO_FILE)
    if script:
        print(f"\n{character.name}: {script}")
        speak_text(f"[EMOTION: happy] {script}", "happy")
    else:
        print(f"    ⚠ Create {Config.OUTRO_FILE} with your outro text")

def cmd_idle():
    ws_manager.send_emotion('neutral')
    idle = anim_manager.get_idle()
    if idle:
        ws_manager.send_animation(idle)
        print(f"    🧘 Idle: {idle.stem}")
    else:
        print("    ⚠ No idle animation set")

# ============================================================================
# MAIN LOOP
# ============================================================================

def is_admin(username: str) -> bool:
    """Check if user is in the admin list (case-insensitive)"""
    return username.lower() in [a.lower() for a in Config.ADMIN_USERS]

def print_help(is_admin_user: bool = False):
    print("""
Commands (anyone):
  user <name>        - Set username
  help               - This help
  exit               - Quit
""")
    if is_admin_user:
        print(f"""Admin Commands (prefix with '{Config.ADMIN_PREFIX}'):
  {Config.ADMIN_PREFIX}viewers <n>       - Set viewer count
  {Config.ADMIN_PREFIX}stream on/off     - Toggle streaming mode

  {Config.ADMIN_PREFIX}intro             - Play intro script
  {Config.ADMIN_PREFIX}outro             - Play outro script  
  {Config.ADMIN_PREFIX}idle              - Go to idle state
  {Config.ADMIN_PREFIX}setidle <name>    - Set idle animation

  {Config.ADMIN_PREFIX}emotion <e>       - Test emotion
  {Config.ADMIN_PREFIX}anim <name>       - Play animation
  {Config.ADMIN_PREFIX}anim list         - List animations

  {Config.ADMIN_PREFIX}reload            - Reload character/lorebook/animations
  {Config.ADMIN_PREFIX}addadmin <name>   - Add admin (session only)

Audio Output:
  {Config.ADMIN_PREFIX}audio             - Show audio output status
  {Config.ADMIN_PREFIX}audio devices     - List audio output devices
  {Config.ADMIN_PREFIX}audio device <n>  - Set output device by index

Voice Input (Whisper):
  {Config.ADMIN_PREFIX}voice on/off      - Enable/disable voice input
  {Config.ADMIN_PREFIX}voice test        - Test microphone + transcription
  {Config.ADMIN_PREFIX}voice devices     - List audio input devices
  {Config.ADMIN_PREFIX}voice device <n>  - Set input device by index
  {Config.ADMIN_PREFIX}voice speak <n>   - Set speaker name for logs
  {Config.ADMIN_PREFIX}voice ptt         - Push-to-talk (record once)
  {Config.ADMIN_PREFIX}voice listen      - Start VAD continuous listening
  {Config.ADMIN_PREFIX}voice stop        - Stop listening

Peer Link (AI-to-AI Sync):
  {Config.ADMIN_PREFIX}peer              - Show peer link status
  {Config.ADMIN_PREFIX}peer on host      - Start as host (wait for peer)
  {Config.ADMIN_PREFIX}peer on client [ip] - Connect to host
  {Config.ADMIN_PREFIX}peer off          - Disconnect from peer
  {Config.ADMIN_PREFIX}peer name <n>  - Set this instance's name

Kick.com Chat:
  {Config.ADMIN_PREFIX}kick              - Show Kick status
  {Config.ADMIN_PREFIX}kick connect      - Connect to your Kick channel
  {Config.ADMIN_PREFIX}kick connect <ch> - Connect to specific channel
  {Config.ADMIN_PREFIX}kick disconnect   - Disconnect from Kick
  {Config.ADMIN_PREFIX}kick poll         - Start polling chat messages
  {Config.ADMIN_PREFIX}kick stop         - Stop polling
  {Config.ADMIN_PREFIX}kick process      - Start auto-processing + polling
  {Config.ADMIN_PREFIX}kick next         - Process one message manually
  {Config.ADMIN_PREFIX}kick clear        - Clear message queue

Idle Chatter (Fill Dead Air):
  {Config.ADMIN_PREFIX}idle chatter      - Show idle chatter status
  {Config.ADMIN_PREFIX}idle chatter on   - Enable idle chatter
  {Config.ADMIN_PREFIX}idle chatter off  - Disable idle chatter
  {Config.ADMIN_PREFIX}idle topics       - List available topic files
  {Config.ADMIN_PREFIX}idle timeout <s>  - Set idle timeout (seconds)
  {Config.ADMIN_PREFIX}idle ramble       - Force a single ramble now
""")
    else:
        print("  (Admin commands hidden - you are not an admin)")
        print(f"  Current admins: {', '.join(Config.ADMIN_USERS)}")

def main():
    print("─" * 50)
    print(f"🎭 {character.name} - v5.8")
    print("─" * 50)
    print(f"WebSocket: ws://localhost:{Config.WEBSOCKET_PORT}")
    print(f"Admins: {', '.join(Config.ADMIN_USERS)}")
    print(f"Admin prefix: {Config.ADMIN_PREFIX}")
    
    # Voice input status
    if VOICE_INPUT_AVAILABLE:
        print(f"Voice: Available [{WHISPER_BACKEND}] (use !voice on to enable)")
    else:
        missing = []
        if not WHISPER_AVAILABLE:
            missing.append("whisper (pip install openai-whisper)")
        if not SOUNDDEVICE_AVAILABLE:
            missing.append("sounddevice (pip install sounddevice numpy)")
        print(f"Voice: Not available (missing: {', '.join(missing)})")
    
    # Kick status
    if KICK_AVAILABLE:
        print(f"Kick: Available (use !kick connect)")
    else:
        print(f"Kick: Not available (pip install KickApi)")
    
    # Idle chatter status
    topic_count = len(idle_chatter.get_topic_files())
    if topic_count > 0:
        print(f"Idle Chatter: {topic_count} topic files (use !idle chatter on)")
    else:
        print(f"Idle Chatter: No topics in {Config.IDLE_TOPICS_DIR}/ (create .txt files)")
    
    # Peer link status
    print(f"Peer Link: Available (use !peer on host or !peer on client)")
    if Config.VOICE_BUFFER_ENABLED:
        print(f"Voice Buffer: Enabled ({Config.VOICE_BUFFER_TIMEOUT}s timeout)")
    
    print("Type 'help' for commands")
    print("─" * 50)
    
    Config.INTRO_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    current_user = Config.DEFAULT_USERNAME
    is_streaming = False
    
    while True:
        try:
            max_tokens, _ = chat_queue.get_token_settings()
            icon = "🔴" if is_streaming else "⚫"
            admin_icon = "👑" if is_admin(current_user) else ""
            
            # Voice icons: 🎤 = actively listening, 🔊 = enabled but not listening
            if voice_manager.is_listening:
                voice_icon = "🎤"
            elif getattr(voice_manager, '_is_paused', False):
                voice_icon = "🎤"  # Still show mic since it will auto-resume
            elif voice_manager.is_enabled:
                voice_icon = "🔊"
            else:
                voice_icon = ""
            
            # Kick icons: 📡 = connected + polling, 🔌 = connected, nothing if not
            if kick_manager.is_connected and kick_manager.is_polling:
                kick_icon = "📡"
            elif kick_manager.is_connected:
                kick_icon = "🔌"
            else:
                kick_icon = ""
            
            # Show pending message count if Kick connected
            pending = chat_queue.get_pending_count() if kick_manager.is_connected else 0
            pending_str = f"|📨{pending}" if pending > 0 else ""
            
            prompt = f"\n{icon}{voice_icon}{kick_icon} [{chat_queue.viewer_count}v|{max_tokens}t{pending_str}] {admin_icon}{current_user}: "
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            prefix = Config.ADMIN_PREFIX
            
            # ─────────────────────────────────────────────────
            # PUBLIC COMMANDS (anyone can use, no prefix)
            # ─────────────────────────────────────────────────
            
            if cmd == "exit":
                # Cleanup
                if peer_link.is_enabled:
                    peer_link.stop()
                print("\n👋 Bye!")
                break
            
            if cmd == "help":
                print_help(is_admin(current_user))
                continue
            
            if cmd.startswith("user "):
                current_user = user_input[5:].strip() or Config.DEFAULT_USERNAME
                status = "known" if memory.is_known_user(current_user) else "new"
                admin_status = " [ADMIN]" if is_admin(current_user) else ""
                print(f"    ✓ {current_user} ({status}){admin_status}")
                continue
            
            # ─────────────────────────────────────────────────
            # ADMIN COMMANDS (require prefix + admin status)
            # ─────────────────────────────────────────────────
            
            # Check if it looks like an admin command (starts with prefix)
            if cmd.startswith(prefix):
                # Remove prefix for command matching
                admin_cmd = cmd[len(prefix):]
                admin_input = user_input[len(prefix):]
                
                # Verify admin status
                if not is_admin(current_user):
                    print(f"    🚫 Admin only. Current admins: {', '.join(Config.ADMIN_USERS)}")
                    continue
                
                # --- Admin: Viewer count ---
                if admin_cmd.startswith("viewers "):
                    try:
                        chat_queue.set_viewer_count(int(admin_input[8:]))
                        tokens, chance = chat_queue.get_token_settings()
                        print(f"    ✓ {chat_queue.viewer_count}v → {tokens}t @ {chance*100:.0f}%")
                    except:
                        print(f"    ⚠ {prefix}viewers <number>")
                    continue
                
                # --- Admin: Stream mode ---
                if admin_cmd == "stream on":
                    is_streaming = True
                    idle_chatter.is_streaming = True
                    chat_processor.is_streaming = True
                    print("    🔴 STREAMING ON")
                    continue
                
                if admin_cmd == "stream off":
                    is_streaming = False
                    idle_chatter.is_streaming = False
                    idle_chatter.stop()  # Stop idle chatter when not streaming
                    chat_processor.is_streaming = False
                    print("    ⚫ STREAMING OFF")
                    continue
                
                # --- Admin: Intro/Outro/Idle ---
                if admin_cmd == "intro":
                    cmd_intro()
                    continue
                
                if admin_cmd == "outro":
                    cmd_outro()
                    continue
                
                if admin_cmd == "idle":
                    cmd_idle()
                    continue
                
                if admin_cmd.startswith("setidle "):
                    name = admin_input[8:].strip()
                    if anim_manager.set_idle_animation(name):
                        print(f"    ✓ Idle: {anim_manager.idle_animation.stem}")
                    else:
                        print(f"    ⚠ Not found: {name}")
                    continue
                
                # --- Admin: Emotion/Animation testing ---
                if admin_cmd.startswith("emotion "):
                    e = admin_input[8:].strip().lower()
                    if e in VALID_EMOTIONS:
                        ws_manager.send_emotion(e)
                        anim = anim_manager.get_for_emotion(e)
                        if anim:
                            ws_manager.send_animation(anim)
                        print(f"    🎭 {e}")
                    else:
                        print(f"    ⚠ Valid: {', '.join(VALID_EMOTIONS)}")
                    continue
                
                if admin_cmd == "anim list":
                    print(f"    {', '.join(anim_manager.list_all()) or 'None'}")
                    continue
                
                if admin_cmd.startswith("anim "):
                    name = admin_input[5:].strip()
                    anim = anim_manager.get_by_name(name)
                    if anim:
                        ws_manager.send_animation(anim)
                        print(f"    🎬 {anim.stem}")
                    else:
                        print(f"    ⚠ Not found")
                    continue
                
                # --- Admin: Reload ---
                if admin_cmd == "reload":
                    character.reload()
                    lorebook.reload()
                    anim_manager.reload()
                    print("    ✓ Reloaded")
                    continue
                
                # --- Admin: Add admin (session only) ---
                if admin_cmd.startswith("addadmin "):
                    new_admin = admin_input[9:].strip()
                    if new_admin and new_admin.lower() not in [a.lower() for a in Config.ADMIN_USERS]:
                        Config.ADMIN_USERS.append(new_admin)
                        print(f"    ✓ Added admin: {new_admin} (session only)")
                    else:
                        print(f"    ⚠ Already admin or invalid")
                    continue
                
                # ─────────────────────────────────────────────────
                # AUDIO OUTPUT COMMANDS
                # ─────────────────────────────────────────────────
                
                # Bare !audio command - show status
                if admin_cmd == "audio":
                    print("    Audio Output Status:")
                    print(f"      Method: {AUDIO_METHOD or 'None'}")
                    print(f"      Speaking: {'Yes 🔊' if is_tts_speaking() else 'No'}")
                    if Config.AUDIO_OUTPUT_DEVICE is not None:
                        print(f"      Device: #{Config.AUDIO_OUTPUT_DEVICE}")
                    else:
                        print(f"      Device: System default")
                    print("    Commands: devices, device <index>")
                    continue
                
                if admin_cmd == "audio devices":
                    devices = list_audio_output_devices()
                    if devices:
                        print("    🔊 Audio Output Devices:")
                        for dev in devices:
                            default_marker = " (default)" if dev['default'] else ""
                            selected = " ◀" if dev['index'] == Config.AUDIO_OUTPUT_DEVICE else ""
                            print(f"      [{dev['index']}] {dev['name']}{default_marker}{selected}")
                        print(f"\n    Use '!audio device <index>' to select")
                    else:
                        print("    ⚠ Could not list audio devices")
                        print("    Install: pip install sounddevice soundfile")
                    continue
                
                if admin_cmd.startswith("audio device "):
                    try:
                        device_idx = int(admin_input[13:].strip())
                        if set_audio_output_device(device_idx):
                            devices = list_audio_output_devices()
                            dev_name = next((d['name'] for d in devices if d['index'] == device_idx), "Unknown")
                            print(f"    ✓ Audio output set to [{device_idx}] {dev_name}")
                        else:
                            print(f"    ⚠ Invalid device index. Use '!audio devices' to list.")
                    except ValueError:
                        print("    ⚠ Usage: !audio device <index>")
                    continue
                
                # ─────────────────────────────────────────────────
                # VOICE INPUT COMMANDS
                # ─────────────────────────────────────────────────
                
                # Bare !voice command - show status
                if admin_cmd == "voice":
                    print("    Voice Input Status:")
                    if VOICE_INPUT_AVAILABLE:
                        print(f"      Backend: {WHISPER_BACKEND}")
                        print(f"      Enabled: {'Yes 🎤' if voice_manager.is_enabled else 'No'}")
                        print(f"      Listening: {'Yes' if voice_manager.is_listening else 'No'}")
                        print(f"      Model: {Config.WHISPER_MODEL_SIZE}")
                        print(f"      Speaker: {Config.VOICE_SPEAKER_NAME}")
                    else:
                        print("      ⚠ Not available (missing dependencies)")
                        if not WHISPER_AVAILABLE:
                            print("        pip install openai-whisper")
                        if not SOUNDDEVICE_AVAILABLE:
                            print("        pip install sounddevice numpy")
                    print("    Commands: on, off, test, ptt, listen, stop, devices, device <n>, speak <name>")
                    continue
                
                if admin_cmd == "voice on":
                    if not VOICE_INPUT_AVAILABLE:
                        print("    ⚠ Voice input dependencies not installed:")
                        if not WHISPER_AVAILABLE:
                            print("      pip install openai-whisper")
                            print("      (or pip install faster-whisper for GPU acceleration)")
                        if not SOUNDDEVICE_AVAILABLE:
                            print("      pip install sounddevice numpy")
                    elif voice_manager.enable():
                        print(f"    🎤 Voice input ENABLED")
                        print(f"       Backend: {WHISPER_BACKEND}")
                        print(f"       Model: {Config.WHISPER_MODEL_SIZE}")
                        print(f"       Speaker: {Config.VOICE_SPEAKER_NAME}")
                    continue
                
                if admin_cmd == "voice off":
                    voice_manager.disable()
                    print("    🔇 Voice input DISABLED")
                    continue
                
                if admin_cmd == "voice test":
                    voice_manager.test()
                    continue
                
                if admin_cmd == "voice devices":
                    devices = voice_manager.list_devices()
                    if devices:
                        print("    Audio input devices:")
                        for d in devices:
                            marker = " *" if d['index'] == Config.VOICE_INPUT_DEVICE else ""
                            print(f"      [{d['index']}] {d['name']}{marker}")
                    else:
                        print("    ⚠ No input devices found (is sounddevice installed?)")
                    continue
                
                if admin_cmd.startswith("voice device "):
                    try:
                        idx = int(admin_input[13:].strip())
                        if voice_manager.set_device(idx):
                            print(f"    ✓ Input device set to [{idx}]")
                        else:
                            print(f"    ⚠ Invalid device index. Use !voice devices to list.")
                    except ValueError:
                        print(f"    ⚠ Usage: !voice device <number>")
                    continue
                
                if admin_cmd.startswith("voice speak "):
                    name = admin_input[12:].strip()
                    if name:
                        Config.VOICE_SPEAKER_NAME = name
                        print(f"    ✓ Speaker name: {name}")
                    else:
                        print(f"    ⚠ Usage: !voice speak <name>")
                    continue
                
                if admin_cmd == "voice ptt":
                    # Push-to-talk: record and transcribe once
                    if not voice_manager.is_enabled:
                        print("    ⚠ Voice not enabled. Use !voice on first.")
                        continue
                    
                    text = voice_manager.record_and_transcribe()
                    if text:
                        # Process as if it came from chat
                        print(f"    💬 {Config.VOICE_SPEAKER_NAME}: {text}")
                        print("    🤔...")
                        response = process_message(Config.VOICE_SPEAKER_NAME, text, is_streaming)
                        if response:
                            print(f"\n{character.name}: {clean_response(response)}")
                            speak_text(response, extract_emotion(response))
                    continue
                
                if admin_cmd == "voice listen":
                    # Start VAD continuous listening
                    if not voice_manager.is_enabled:
                        print("    ⚠ Voice not enabled. Use !voice on first.")
                        continue
                    
                    def on_voice_input(text):
                        """Callback when VAD detects speech (with peer link coordination)"""
                        print(f"\n    💬 {Config.VOICE_SPEAKER_NAME}: {text}")
                        
                        # If peer link is active, wait for peer to finish before responding
                        if peer_link.is_enabled and peer_link.is_connected:
                            if peer_link.is_peer_busy():
                                print(f"    ⏳ Waiting for {peer_link.peer_name} to finish...")
                                if not peer_link.wait_for_peer_idle():
                                    print(f"    ⚠ Peer still busy, responding anyway")
                        
                        # Broadcast that we're processing
                        peer_link.set_state(PeerLinkManager.STATE_PROCESSING)
                        
                        print("    🤔...")
                        response = process_message(Config.VOICE_SPEAKER_NAME, text, is_streaming)
                        if response:
                            print(f"\n{character.name}: {clean_response(response)}")
                            speak_text(response, extract_emotion(response))
                        else:
                            # No response generated, go back to idle
                            peer_link.set_state(PeerLinkManager.STATE_IDLE)
                        
                        # Re-show prompt
                        print(f"\n{icon} [{chat_queue.viewer_count}v|{max_tokens}t] {admin_icon}{current_user}: ", end="", flush=True)
                    
                    voice_manager.start_listening(on_voice_input)
                    continue
                
                if admin_cmd == "voice stop":
                    voice_manager.stop_listening()
                    continue
                
                # ─────────────────────────────────────────────────
                # PEER LINK COMMANDS (AI-to-AI Synchronization)
                # ─────────────────────────────────────────────────
                
                if admin_cmd == "peer" or admin_cmd == "peer status":
                    status = peer_link.get_status()
                    print("    Peer Link Status:")
                    print(f"      Enabled: {'Yes' if status['enabled'] else 'No'}")
                    if status['enabled']:
                        print(f"      Mode: {status['mode']}")
                        print(f"      Connected: {'Yes 🤝' if status['connected'] else 'No'}")
                        print(f"      My name: {status['my_name']} ({status['my_state']})")
                        if status['connected']:
                            print(f"      Peer: {status['peer_name']} ({status['peer_state']})")
                    print("    Commands:")
                    print(f"      {Config.ADMIN_PREFIX}peer on host         - Start as host (listen for peer)")
                    print(f"      {Config.ADMIN_PREFIX}peer on client [ip]  - Connect to host")
                    print(f"      {Config.ADMIN_PREFIX}peer off             - Disconnect")
                    print(f"      {Config.ADMIN_PREFIX}peer name <name>     - Set instance name")
                    continue
                
                if admin_cmd == "peer on host":
                    if peer_link.is_enabled:
                        print("    ⚠ Peer Link already active. Use !peer off first.")
                        continue
                    peer_link.name = Config.PEER_LINK_NAME
                    peer_link.start_host(Config.PEER_LINK_PORT)
                    continue
                
                if admin_cmd.startswith("peer on client"):
                    if peer_link.is_enabled:
                        print("    ⚠ Peer Link already active. Use !peer off first.")
                        continue
                    # Parse optional IP: !peer on client 192.168.1.100
                    parts = admin_cmd.split()
                    host_ip = parts[3] if len(parts) > 3 else Config.PEER_LINK_HOST
                    peer_link.name = Config.PEER_LINK_NAME
                    peer_link.start_client(host_ip, Config.PEER_LINK_PORT)
                    continue
                
                if admin_cmd == "peer off":
                    peer_link.stop()
                    continue
                
                if admin_cmd.startswith("peer name "):
                    new_name = admin_cmd[len("peer name "):].strip()
                    if new_name:
                        peer_link.name = new_name
                        Config.PEER_LINK_NAME = new_name
                        print(f"    ✓ Peer Link name: {new_name}")
                    else:
                        print(f"    ⚠ Usage: !peer name <name>")
                    continue
                
                # ─────────────────────────────────────────────────
                # KICK.COM CHAT COMMANDS
                # ─────────────────────────────────────────────────
                
                # Bare !kick command - show status
                if admin_cmd == "kick":
                    status = kick_manager.get_status()
                    print("    Kick.com Chat Status:")
                    if status['available']:
                        print(f"      Connected: {'Yes 📡' if status['connected'] else 'No'}")
                        if status['connected']:
                            print(f"      Channel: {status['channel']}")
                            print(f"      Followers: {status['followers']}")
                            print(f"      Polling: {'Yes' if status['polling'] else 'No'}")
                            print(f"      Auto-process: {'Yes ⚙️' if chat_processor.is_running else 'No'}")
                        print(f"      Pending msgs: {status['pending_messages']}")
                        print(f"      Messages/min: {status['mpm']:.1f}")
                    else:
                        print("      ⚠ Not available (missing dependencies)")
                        print("        pip install KickApi")
                    print("    Commands: connect, disconnect, poll, stop, process, next, clear")
                    continue
                
                if admin_cmd == "kick connect":
                    if not KICK_AVAILABLE:
                        print("    ⚠ KickApi not available. Install: pip install KickApi")
                        continue
                    kick_manager.connect()
                    continue
                
                if admin_cmd.startswith("kick connect "):
                    if not KICK_AVAILABLE:
                        print("    ⚠ KickApi not available. Install: pip install KickApi")
                        continue
                    channel = admin_input[13:].strip()
                    kick_manager.connect(channel)
                    continue
                
                if admin_cmd == "kick disconnect":
                    kick_manager.disconnect()
                    if chat_processor.is_running:
                        chat_processor.stop()
                    continue
                
                if admin_cmd == "kick poll":
                    # Start polling only
                    kick_manager.start_polling()
                    continue
                
                if admin_cmd == "kick stop":
                    kick_manager.stop_polling()
                    if chat_processor.is_running:
                        chat_processor.stop()
                    continue
                
                if admin_cmd == "kick process":
                    # Start both polling and processing
                    if not kick_manager.is_connected:
                        print("    ⚠ Not connected. Use '!kick connect' first")
                        continue
                    if not kick_manager.is_polling:
                        kick_manager.start_polling()
                    chat_processor.is_streaming = is_streaming
                    chat_processor.start()
                    continue
                
                if admin_cmd == "kick next":
                    # Process one message manually
                    msg = chat_queue.get_next_message()
                    if msg:
                        print(f"    📨 {msg.username}: {msg.content}")
                        print("    🤔...")
                        response = process_message(msg.username, msg.content, is_streaming)
                        if response:
                            print(f"\n{character.name}: {clean_response(response)}")
                            speak_text(response, extract_emotion(response))
                    else:
                        print("    📭 No messages in queue")
                    continue
                
                if admin_cmd == "kick clear":
                    chat_queue.clear()
                    print("    🗑️ Queue cleared")
                    continue
                
                # ─────────────────────────────────────────────────
                # IDLE CHATTER COMMANDS
                # ─────────────────────────────────────────────────
                
                # Bare !idle chatter command - show status
                if admin_cmd == "idle chatter":
                    status = idle_chatter.get_status()
                    print("    Idle Chatter Status:")
                    print(f"      Enabled: {'Yes 💭' if status['enabled'] else 'No'}")
                    print(f"      Rambling: {'Yes' if status['rambling'] else 'No'}")
                    print(f"      Idle for: {status['idle_seconds']}s (timeout: {status['timeout']}s)")
                    print(f"      Topic files: {status['topic_files']}")
                    if status['current_topic']:
                        print(f"      Current topic: {status['current_topic']} ({status['rambles_on_topic']} rambles)")
                    print("    Commands: on, off, topics, timeout <s>, ramble")
                    continue
                
                if admin_cmd == "idle chatter on":
                    if not is_streaming:
                        print("    ⚠ Enable streaming first with !stream on")
                        continue
                    
                    # Check if topic files exist
                    topic_files = idle_chatter.get_topic_files()
                    if not topic_files:
                        print(f"    ⚠ No topic files found in {Config.IDLE_TOPICS_DIR}/")
                        print(f"    Create .txt files with prompts (one per line)")
                        print(f"    Example: echo 'Talk about your favorite video game' > topics/games.txt")
                        continue
                    
                    idle_chatter.is_streaming = True
                    idle_chatter.enable()
                    idle_chatter.start()
                    print(f"    Found {len(topic_files)} topic file(s): {', '.join(idle_chatter.list_topics())}")
                    continue
                
                if admin_cmd == "idle chatter off":
                    idle_chatter.disable()
                    continue
                
                if admin_cmd == "idle topics":
                    topics = idle_chatter.list_topics()
                    if topics:
                        print(f"    📚 Topic files ({len(topics)}):")
                        for name in topics:
                            file_path = Config.IDLE_TOPICS_DIR / f"{name}.txt"
                            if file_path.exists():
                                lines = len([l for l in file_path.read_text(encoding='utf-8').splitlines() if l.strip() and not l.strip().startswith('#')])
                                print(f"      {name}.txt ({lines} prompts)")
                    else:
                        print(f"    📂 No topic files in {Config.IDLE_TOPICS_DIR}/")
                        print(f"    Create .txt files with prompts, one per line")
                    continue
                
                if admin_cmd.startswith("idle timeout "):
                    try:
                        seconds = float(admin_input[13:].strip())
                        if seconds < 5:
                            print("    ⚠ Minimum timeout is 5 seconds")
                        else:
                            Config.IDLE_TIMEOUT_SECONDS = seconds
                            print(f"    ✓ Idle timeout set to {seconds}s")
                    except ValueError:
                        print("    ⚠ Usage: !idle timeout <seconds>")
                    continue
                
                if admin_cmd == "idle ramble":
                    # Force a single ramble now (for testing)
                    prompt = idle_chatter.get_ramble_prompt()
                    if prompt:
                        print(f"    💭 Forcing ramble: {prompt[:50]}...")
                        idle_chatter._generate_ramble(prompt)
                    else:
                        print("    ⚠ No topics available. Add .txt files to topics/")
                    continue
                
                # Unknown admin command
                print(f"    ⚠ Unknown admin command. Type 'help' for list.")
                continue
            
            # ─────────────────────────────────────────────────
            # NORMAL CHAT (anything else goes to LLM)
            # ─────────────────────────────────────────────────
            # Reset idle chatter when manual chat happens
            idle_chatter.reset_activity()
            
            print("    🤔...")
            response = process_message(current_user, user_input, is_streaming)
            
            if response:
                print(f"\n{character.name}: {clean_response(response)}")
                speak_text(response, extract_emotion(response))
            
            # Reset activity again after responding
            idle_chatter.reset_activity()
            
        except KeyboardInterrupt:
            if peer_link.is_enabled:
                peer_link.stop()
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"    ❌ {e}")

if __name__ == "__main__":
    main()
