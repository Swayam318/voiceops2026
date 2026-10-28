"""Step 4: Voice Layer for ZentroMart Customer Support Agent.

Integrates Gnani STT and TTS SDK (gnani-vachana) with the Step 3 conversational agent.
- Transcribes customer audio using Gnani STT
- Handles empty/unintelligible audio gracefully with a polite fallback
- Formats text responses for clear spoken delivery (spoken numbers, rupee expansion, short sentences)
- Synthesizes spoken replies using Gnani TTS (voice="Kaveri", model="timbre-v2.5")
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, "project.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from src.agent import ZentroMartAgent

logger = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VoiceLayer:
    """Voice layer wrapping ZentroMartAgent with Gnani STT and TTS."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_voice: str = "Kaveri",
        default_model: str = "timbre-v2.5",
        default_language: str = "en-IN"
    ):
        self.gnani_api_key = api_key or os.getenv("GNANI_API_KEY")
        if not self.gnani_api_key:
            raise ValueError("GNANI_API_KEY not found in environment or project.env")

        # Set environment variable if needed by SDK internally
        os.environ["GNANI_API_KEY"] = self.gnani_api_key

        self.stt = GnaniSTTClient()
        self.tts = GnaniTTSClient()
        self.agent = ZentroMartAgent()

        self.default_voice = default_voice
        self.default_model = default_model
        self.default_language = default_language

    def format_for_tts(self, text: str) -> str:
        """Format reply text for clear, natural speech synthesis."""
        if not text:
            return ""

        formatted = str(text)

        # 1. Remove markdown symbols (headers, bold, italics, code blocks)
        formatted = re.sub(r'#+\s*', '', formatted)
        formatted = re.sub(r'\*\*([^*]+)\*\*', r'\1', formatted)
        formatted = re.sub(r'\*([^*]+)\*', r'\1', formatted)
        formatted = re.sub(r'`([^`]+)`', r'\1', formatted)

        # 2. Expand currency symbols: ₹500 -> 500 rupees, ₹ 49 -> 49 rupees
        formatted = re.sub(r'₹\s*(\d+)', r'\1 rupees', formatted)
        formatted = re.sub(r'Rs\.?\s*(\d+)', r'\1 rupees', formatted)

        # 3. Replace ampersands with 'and'
        formatted = formatted.replace('&', ' and ')

        # 4. Break 8-digit order IDs into spaced groups of 4: "89782837" -> "8978 2837"
        formatted = re.sub(r'\b(\d{4})(\d{4})\b', r'\1 \2', formatted)

        # 5. Clean parenthetical duplicate dates or technical annotations
        formatted = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', formatted)

        # 6. Ensure clean spacing and remove extra whitespace
        formatted = re.sub(r'\s+', ' ', formatted).strip()

        # 7. Keep sentences short: ensure no sentence exceeds 25 words
        sentences = []
        for raw_s in re.split(r'(?<=[.!?])\s+', formatted):
            raw_s = raw_s.strip()
            if not raw_s:
                continue
            words = raw_s.split()
            if len(words) > 25:
                # Split on comma or conjunction if too long
                sub_parts = re.split(r'[,;]\s+', raw_s)
                for part in sub_parts:
                    p_clean = part.strip()
                    if p_clean:
                        sentences.append(p_clean if p_clean.endswith(('.', '!', '?')) else p_clean + '.')
            else:
                sentences.append(raw_s)

        result = " ".join(sentences)
        return result.strip()

    def transcribe(self, audio_path: str, language_code: str = "en-IN") -> Dict[str, Any]:
        """Transcribe customer audio using Gnani STT REST endpoint."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            res = self.stt.transcribe(audio_path, language_code=language_code)
            transcript = res.get("transcript", "").strip()
            if not transcript:
                return {"transcript": "", "fallback": True}
            return {"transcript": transcript, "fallback": False}
        except Exception as e:
            logger.warning("STT transcription error on %s: %s", audio_path, e)
            return {"transcript": "", "fallback": True}

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None
    ) -> str:
        """Synthesize spoken audio using Gnani TTS and save to output_path."""
        voice = voice or self.default_voice
        model = model or self.default_model
        language = language or self.default_language

        formatted_text = self.format_for_tts(text)
        if not formatted_text:
            formatted_text = "I am listening."

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        models_to_try = [model, "timbre-v3.0"] if model != "timbre-v3.0" else ["timbre-v3.0", "timbre-v2.5"]
        last_error = None

        for m in models_to_try:
            try:
                audio_bytes = self.tts.synthesize(
                    formatted_text,
                    voice=voice,
                    model=m,
                    language=language,
                    output_file=output_path
                )
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path
            except Exception as e:
                logger.warning("TTS synthesize with model %s failed: %s. Retrying...", m, e)
                last_error = e

        if last_error:
            raise last_error
        return output_path

    def voice_turn(
        self,
        audio_path: str,
        output_audio_path: str,
        language_code: str = "en-IN",
        voice: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute one complete voice turn: Audio In -> STT -> Agent -> TTS -> Audio Out."""
        # 1. Transcribe customer speech
        stt_result = self.transcribe(audio_path, language_code=language_code)
        transcript = stt_result["transcript"]
        fallback_triggered = stt_result["fallback"]

        # 2. Get Agent reply
        if fallback_triggered or not transcript:
            reply = "Sorry, I didn't catch that. Could you say it again?"
        else:
            reply = self.agent.respond(transcript)

        # 3. Format text and synthesize spoken audio
        formatted_reply = self.format_for_tts(reply)
        self.synthesize(
            text=formatted_reply,
            output_path=output_audio_path,
            voice=voice,
            model=model,
            language=language_code
        )

        return {
            "transcript": transcript,
            "reply": reply,
            "reply_formatted": formatted_reply,
            "output_audio": output_audio_path,
            "fallback_triggered": fallback_triggered,
            "retrieved_chunks": getattr(self.agent, "last_retrieved_chunks", []),
            "last_decision": getattr(self.agent, "last_decision", "answer"),
            "tool_calls": getattr(self.agent, "last_tool_calls", [])
        }

    def reset(self):
        """Reset conversation state for a new caller."""
        self.agent.reset()
