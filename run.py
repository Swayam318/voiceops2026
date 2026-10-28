#!/usr/bin/env python3
"""Batch Runner for VoiceOps 2026 Customer Support Agent.

Interface:
    python run.py --input_dir ./audio_inputs/ --output_dir ./outputs/

For each .wav file in input_dir, it:
1. Transcribes customer audio with Gnani STT
2. Processes through ZentroMart Conversational Agent (RAG + Tools + Memory)
3. Synthesizes spoken reply with Gnani TTS
4. Saves to output_dir:
   - <name>_transcript.txt
   - <name>_reply.txt
   - <name>_reply.wav
   - <name>_chunks.json
"""

import os
import sys
import glob
import json
import argparse
import logging
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add workspace directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.voice import VoiceLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run")


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceOps 2026 Batch Audio Runner")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--language", type=str, default="en-IN", help="Language code (default: en-IN)")
    parser.add_argument("--voice", type=str, default="Kaveri", help="Gnani TTS voice (default: Kaveri)")
    parser.add_argument("--model", type=str, default="timbre-v2.5", help="Gnani TTS model (default: timbre-v2.5)")
    parser.add_argument("--reset_per_file", action="store_true", default=True,
                        help="Reset agent memory between distinct audio files (default: True)")
    return parser.parse_args()


def process_audio_batch(
    input_dir: str,
    output_dir: str,
    language: str = "en-IN",
    voice: str = "Kaveri",
    model: str = "timbre-v2.5",
    reset_per_file: bool = True
):
    """Process all .wav files in input_dir and write 4 deliverables per file."""
    os.makedirs(output_dir, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))

    if not wav_files:
        print(f"\n[ERROR] No .wav files found in directory: {input_dir}")
        print("Please place test audio files (.wav) in the input folder.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"VOICEOPS 2026 — BATCH AUDIO RUNNER")
    print(f"Input Directory:  {input_dir} ({len(wav_files)} files)")
    print(f"Output Directory: {output_dir}")
    print(f"STT/TTS Language: {language} | Voice: {voice} | Model: {model}")
    print("=" * 65 + "\n")

    voice_layer = VoiceLayer(default_voice=voice, default_model=model, default_language=language)

    processed_count = 0
    fallback_count = 0

    for idx, audio_path in enumerate(wav_files, 1):
        filename = os.path.basename(audio_path)
        stem = Path(audio_path).stem

        # Destination paths
        out_transcript_path = os.path.join(output_dir, f"{stem}_transcript.txt")
        out_reply_path = os.path.join(output_dir, f"{stem}_reply.txt")
        out_audio_path = os.path.join(output_dir, f"{stem}_reply.wav")
        out_chunks_path = os.path.join(output_dir, f"{stem}_chunks.json")

        print(f"[{idx:02d}/{len(wav_files):02d}] Processing {filename}...")

        try:
            # Execute one voice turn end-to-end
            res = voice_layer.voice_turn(
                audio_path=audio_path,
                output_audio_path=out_audio_path,
                language_code=language,
                voice=voice,
                model=model
            )

            # 1. Save transcript
            with open(out_transcript_path, "w", encoding="utf-8") as f:
                f.write(res["transcript"] + "\n")

            # 2. Save text reply
            with open(out_reply_path, "w", encoding="utf-8") as f:
                f.write(res["reply"] + "\n")

            # 3. Save chunks used
            chunks_data = {
                "audio_file": filename,
                "transcript": res["transcript"],
                "decision": res["last_decision"],
                "tool_calls": res["tool_calls"],
                "retrieved_chunks": res["retrieved_chunks"]
            }
            with open(out_chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            if res["fallback_triggered"]:
                fallback_count += 1

            processed_count += 1
            reply_preview = res["reply"][:60].replace("\n", " ")
            print(f"[{stem}] Transcript: \"{res['transcript']}\" | Reply: \"{reply_preview}\"")
            print(f"  -> Saved: {stem}_transcript.txt, {stem}_reply.txt, {stem}_reply.wav, {stem}_chunks.json in {output_dir}\n")

        except Exception as e:
            logger.error("Error processing %s: %s", filename, e)

        if reset_per_file:
            voice_layer.reset()

    print("=" * 65)
    print("BATCH PROCESSING COMPLETED")
    print(f"Processed: {processed_count} files")
    print(f"Fallbacks triggered: {fallback_count}")
    print(f"Output saved to: {output_dir}")
    print("=" * 65 + "\n")


def main():
    args = parse_args()
    process_audio_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        language=args.language,
        voice=args.voice,
        model=args.model,
        reset_per_file=args.reset_per_file
    )


if __name__ == "__main__":
    main()
