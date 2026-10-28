"""Step 1: Baseline LLM Agent for ZentroMart Customer Support.

This module implements Layer 1 of VoiceOps 2026:
- Calls Gemini LLM API with a ZentroMart AI system prompt
- Evaluates against practice/questions.csv using the `asr_text` column
- Evaluates correctness based on facts and policy boundaries
- Prints and saves the baseline accuracy score to results/step1_score.txt
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to sys.path for relative imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluate import evaluate_question_response

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, "project.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger("llm_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ZentroMartLLMBaseline:
    """Baseline LLM agent for ZentroMart without document grounding (Step 1)."""

    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or .env/project.env")
        self.models = models or ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
        self.system_prompt = (
            "You are ZentroMart's AI customer support assistant for an Indian online store "
            "selling groceries, fashion, and electronics. Always introduce yourself as ZentroMart's AI assistant. "
            "Provide helpful, concise customer support answers in the requested language (English, Hindi, or Hinglish). "
            "Note: You do not yet have access to internal company policy documents, order databases, or warranty records, "
            "so you must answer based solely on standard general ecommerce knowledge."
        )

    def generate_reply(self, prompt: str, max_retries: int = 3) -> str:
        """Call the Gemini API with the customer question and system prompt."""
        payload = {
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 200,
            }
        }

        for model_name in self.models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            for attempt in range(max_retries):
                try:
                    resp = requests.post(url, json=payload, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip()
                        return "Hello, I am ZentroMart's AI assistant. I'm sorry, I could not process your request at this time."
                    elif resp.status_code in (429, 503):
                        wait_time = (attempt + 1) * 2
                        logger.warning("Model %s returned %d. Retrying in %ds...", model_name, resp.status_code, wait_time)
                        time.sleep(wait_time)
                    else:
                        logger.warning("Model %s error %d: %s", model_name, resp.status_code, resp.text[:150])
                        break  # Try next model
                except Exception as e:
                    logger.warning("Exception calling %s: %s", model_name, e)
                    time.sleep(1)

        return "Hello, I am ZentroMart's AI assistant. I'm having trouble connecting to support servers right now."


def find_questions_csv() -> str:
    """Locate questions.csv in the workspace."""
    candidates = [
        os.path.join(BASE_DIR, "Capstone_Project_Dataset_FTTT_2026", "practice", "questions.csv"),
        os.path.join(BASE_DIR, "practice", "questions.csv"),
        os.path.join(BASE_DIR, "data", "practice", "questions.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find questions.csv in any expected path: {candidates}")


def run_baseline_evaluation(
    questions_path: Optional[str] = None,
    results_dir: Optional[str] = None
) -> Tuple[float, int, int]:
    """Run baseline evaluation on practice/questions.csv using asr_text column.
    
    Returns:
        (accuracy_pct, correct_count, total_count)
    """
    if questions_path is None:
        questions_path = find_questions_csv()

    if results_dir is None:
        results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading questions from %s", questions_path)
    df = pd.read_csv(questions_path, dtype=str)

    agent = ZentroMartLLMBaseline()

    total = len(df)
    correct_count = 0
    predictions = []

    print(f"\n--- Running Step 1: LLM Baseline Evaluation ({total} questions) ---", flush=True)

    for idx, row in df.iterrows():
        qid = row.get("id", f"Q{idx+1}")
        # Use asr_text column as required by Step 1 specification
        asr_input = row.get("asr_text")
        if not asr_input or pd.isna(asr_input) or not str(asr_input).strip():
            asr_input = row.get("question", "")

        reply = agent.generate_reply(str(asr_input))

        must_inc = row.get("must_include", "")
        must_not_inc = row.get("must_not_include", "")
        exp_dec = row.get("expected_decision", None)
        acc_dec = row.get("acceptable_decisions", None)

        is_correct, reason = evaluate_question_response(
            reply=reply,
            must_include=must_inc,
            must_not_include=must_not_inc,
            expected_decision=exp_dec,
            acceptable_decisions=acc_dec
        )

        if is_correct:
            correct_count += 1

        predictions.append({
            "id": qid,
            "asr_text": asr_input,
            "agent_reply": reply,
            "is_correct": is_correct,
            "reason": reason,
            "must_include": must_inc,
            "must_not_include": must_not_inc,
            "gold_evidence": row.get("gold_evidence", "")
        })

        status_sym = "[PASS]" if is_correct else "[FAIL]"
        print(f"[{idx+1:02d}/{total:02d}] {qid} {status_sym} -> {reason}", flush=True)
        time.sleep(0.2)

    accuracy_pct = round((correct_count / total) * 100.0, 2)

    # Save detailed predictions for reporting analysis
    pred_path = os.path.join(results_dir, "step1_predictions.csv")
    pd.DataFrame(predictions).to_csv(pred_path, index=False, encoding="utf-8")

    # Format score text
    score_summary = (
        f"Step 1 — LLM Baseline Evaluation Score\n"
        f"----------------------------------------\n"
        f"Total Questions Evaluated: {total}\n"
        f"Correct Answers:          {correct_count}\n"
        f"Failed Answers:           {total - correct_count}\n"
        f"Accuracy:                 {accuracy_pct}%\n"
        f"Input Column:             asr_text\n"
        f"Model:                    gemini-3.5-flash-lite (Prompt-only, ungrounded)\n"
    )

    score_path = os.path.join(results_dir, "step1_score.txt")
    with open(score_path, "w", encoding="utf-8") as f:
        f.write(score_summary)

    print("\n" + "=" * 50, flush=True)
    print(score_summary, flush=True)
    print("=" * 50, flush=True)
    print(f"Score saved to: {score_path}", flush=True)
    print(f"Predictions saved to: {pred_path}\n", flush=True)

    return accuracy_pct, correct_count, total


if __name__ == "__main__":
    run_baseline_evaluation()
