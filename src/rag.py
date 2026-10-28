"""Step 2: RAG Pipeline and Document Grounding for ZentroMart Customer Support.

This module implements Layer 2 of VoiceOps 2026:
- Chunks all 20 active documents in kb/*.md, preserving [SECTION-ID], status, effective_from
- Explicitly ignores pol_returns_v3_superseded.md
- Generates embeddings using models/gemini-embedding-001
- Builds and persists a FAISS index in kb_index/ along with chunks_metadata.json
- Implements prioritized retrieval: pol_* > agent_* > help_* > faq_* > mkt_*
- Answers strictly from retrieved chunks or declares "I don't have information on that"
- Logs retrieved chunks to results/step2_chunks.jsonl
- Evaluates against practice/questions.csv and saves scores to results/step2_score.txt
- Performs detailed failure analysis saved to results/step2_failures.txt
"""

import os
import sys
import re
import glob
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
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

import faiss
from src.evaluate import evaluate_question_response

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, "project.env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Priority weights for document types
# Precedence: pol_* > agent_* > help_* > faq_* > mkt_*
DOC_TYPE_PRIORITY = {
    "policy": 0.12,
    "agent_guide": 0.08,
    "meta": 0.07,
    "help": 0.04,
    "faq": 0.01,
    "marketing": 0.00
}


def get_doc_priority(source_file: str, doc_type: str) -> float:
    """Calculate priority bonus for retrieval ranking."""
    fname = os.path.basename(source_file).lower()
    if fname.startswith("pol_"):
        return 0.12
    elif fname.startswith("agent_"):
        return 0.08
    elif fname.startswith("00_about_"):
        return 0.07
    elif fname.startswith("help_"):
        return 0.04
    elif fname.startswith("faq_"):
        return 0.01
    elif fname.startswith("mkt_"):
        return 0.00
    return DOC_TYPE_PRIORITY.get(doc_type, 0.0)


def parse_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    """Extract frontmatter and remaining body from markdown content."""
    meta = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
    return meta, body


def chunk_kb_documents(kb_dir: str) -> List[Dict[str, Any]]:
    """Parse and chunk all documents in kb/*.md.
    
    Excludes pol_returns_v3_superseded.md per project rules.
    Splits by sections with ## [SECTION-ID].
    """
    chunks = []
    pattern = os.path.join(kb_dir, "*.md")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No markdown files found in {kb_dir}")

    for file_path in files:
        fname = os.path.basename(file_path)
        # CRITICAL RULE 6: Ignore pol_returns_v3_superseded.md
        if fname == "pol_returns_v3_superseded.md":
            logger.info("Ignoring superseded document: %s", fname)
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        meta, body = parse_frontmatter(raw_text)
        doc_id = meta.get("doc_id", "")
        doc_title = meta.get("title", fname.replace(".md", ""))
        doc_type = meta.get("doc_type", "policy")
        effective_from = meta.get("effective_from", "2026-08-01")
        status = meta.get("status", "current")

        # Split on section headings: ## [SECTION-ID] Section Title
        # Regex finds all sections
        section_splits = re.split(r'(^##\s*\[[^\]]+\][^\n]*)', body, flags=re.MULTILINE)

        if len(section_splits) <= 1:
            # Document has no ## [SECTION-ID], create a single chunk
            clean_text = body.strip()
            if clean_text:
                chunks.append({
                    "chunk_id": f"{fname}_00",
                    "section_id": doc_id or fname,
                    "title": doc_title,
                    "source_file": fname,
                    "doc_title": doc_title,
                    "doc_type": doc_type,
                    "status": status,
                    "effective_from": effective_from,
                    "chunk_text": f"Document: {doc_title} ({fname})\nStatus: {status} | Effective: {effective_from}\n\n{clean_text}"
                })
            continue

        # First part before the first ## heading (e.g. # Title)
        intro_text = section_splits[0].strip()

        # Pair each heading with its section body
        for i in range(1, len(section_splits), 2):
            heading = section_splits[i].strip()
            section_content = section_splits[i + 1].strip() if (i + 1) < len(section_splits) else ""

            # Extract section ID: ## [SECTION-ID] Title
            sec_match = re.search(r'\[([^\]]+)\]\s*(.*)$', heading)
            if sec_match:
                section_id = sec_match.group(1).strip()
                section_title = sec_match.group(2).strip()
            else:
                section_id = f"SEC-{i}"
                section_title = heading

            full_chunk_text = (
                f"Document: {doc_title} ({fname})\n"
                f"Section: [{section_id}] {section_title}\n"
                f"Effective: {effective_from} | Status: {status} | Type: {doc_type}\n\n"
                f"{heading}\n{section_content}"
            )

            chunks.append({
                "chunk_id": f"{fname}_{section_id}",
                "section_id": section_id,
                "title": section_title,
                "source_file": fname,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "status": status,
                "effective_from": effective_from,
                "chunk_text": full_chunk_text
            })

    logger.info("Successfully extracted %d chunks from %d KB documents", len(chunks), len(files) - 1)
    return chunks


def get_gemini_embeddings(texts: List[str], api_key: str, batch_size: int = 15) -> np.ndarray:
    """Batch embed texts using Google models/gemini-embedding-001 via REST."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={api_key}"
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        requests_payload = [
            {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}}
            for t in batch
        ]
        payload = {"requests": requests_payload}

        for attempt in range(4):
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    embs = [item["values"] for item in data.get("embeddings", [])]
                    all_embeddings.extend(embs)
                    break
                elif resp.status_code in (429, 503):
                    wait = (attempt + 1) * 3
                    logger.warning("Embeddings API %d. Retrying in %ds...", resp.status_code, wait)
                    time.sleep(wait)
                else:
                    logger.error("Embeddings error %d: %s", resp.status_code, resp.text[:150])
                    time.sleep(2)
            except Exception as e:
                logger.error("Exception during batch embedding: %s", e)
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")

    matrix = np.array(all_embeddings, dtype=np.float32)
    # Normalize to unit length for exact cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return matrix


class ZentroMartRAG:
    """RAG-grounded customer support agent with FAISS vector search and priority re-ranking."""

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        force_reindex: bool = False
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        if kb_dir is None:
            kb_dir = os.path.join(BASE_DIR, "Capstone_Project_Dataset_FTTT_2026", "kb")
            if not os.path.exists(kb_dir):
                kb_dir = os.path.join(BASE_DIR, "kb")
        self.kb_dir = kb_dir

        if index_dir is None:
            index_dir = os.path.join(BASE_DIR, "kb_index")
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_file = os.path.join(self.index_dir, "faiss_index.bin")
        self.meta_file = os.path.join(self.index_dir, "chunks_metadata.json")

        self.chunks: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.init_index(force_reindex=force_reindex)

    def init_index(self, force_reindex: bool = False):
        """Build or load the FAISS index and chunk metadata."""
        if not force_reindex and os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            logger.info("Loading existing FAISS index from %s", self.index_file)
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info("Loaded index with %d vectors and metadata chunks", len(self.chunks))
        else:
            logger.info("Building new FAISS index from %s", self.kb_dir)
            self.chunks = chunk_kb_documents(self.kb_dir)
            texts = [c["chunk_text"] for c in self.chunks]
            logger.info("Generating embeddings for %d chunks using gemini-embedding-001...", len(texts))
            embeddings = get_gemini_embeddings(texts, self.api_key)

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            logger.info("Saving index to %s and metadata to %s", self.index_file, self.meta_file)
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def retrieve(self, query: str, top_k: int = 5, relevance_threshold: float = 0.42) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks with priority re-ranking."""
        if not query or not query.strip():
            return []

        # Generate normalized embedding for query
        q_emb = get_gemini_embeddings([query], self.api_key)[0]
        q_vec = np.array([q_emb], dtype=np.float32)

        # Retrieve top candidates (15) for priority re-ranking
        candidate_count = min(15, len(self.chunks))
        scores, indices = self.index.search(q_vec, candidate_count)

        raw_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            base_score = float(score)
            priority_bonus = get_doc_priority(chunk["source_file"], chunk.get("doc_type", ""))
            re_ranked_score = base_score + priority_bonus

            raw_results.append({
                **chunk,
                "base_score": round(base_score, 4),
                "re_ranked_score": round(re_ranked_score, 4)
            })

        # Check maximum relevance against threshold to detect unsupported questions
        if not raw_results or max(r["base_score"] for r in raw_results) < relevance_threshold:
            # Query is out of domain (e.g., student discount, midnight delivery)
            return []

        # Sort by re-ranked score descending (priority: policy > agent_guide > help > faq > mkt)
        raw_results.sort(key=lambda x: x["re_ranked_score"], reverse=True)
        return raw_results[:top_k]

    def answer_question(self, question: str, max_retries: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Answer question strictly from retrieved KB chunks."""
        retrieved = self.retrieve(question, top_k=5)

        # If no relevant chunks retrieved: policy states declare no info
        if not retrieved:
            return "I don't have information on that.", []

        # Format context from retrieved chunks
        context_blocks = []
        for i, c in enumerate(retrieved, 1):
            context_blocks.append(
                f"[Source {i}: {c['source_file']} | Section: {c['section_id']} | Status: {c['status']}]\n"
                f"{c['chunk_text']}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

        system_instruction = (
            "You are ZentroMart's AI customer support assistant. Answer the customer's question ONLY "
            "from the provided context. If the context does not contain enough information to answer, "
            "respond exactly with:\n"
            "I don't have information on that.\n\n"
            "CRITICAL RULES:\n"
            "1. Answer ONLY from the provided context. Never invent policies, dates, amounts, or timelines.\n"
            "2. When two documents disagree, current policies always win over FAQs or marketing.\n"
            "3. Reference date: today is 20 October 2026. Business days are Monday-Saturday, excluding warehouse holidays.\n"
            "4. Return windows are calendar days.\n"
            "5. Answer in the same language or tone (English, Hindi, or Hinglish) as the customer question."
        )

        user_content = (
            f"CONTEXT:\n{context_str}\n\n"
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"ANSWER:"
        )

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 250,
            }
        }

        models = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            for attempt in range(max_retries):
                try:
                    resp = requests.post(url, json=payload, timeout=20)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip(), retrieved
                    elif resp.status_code in (429, 503):
                        time.sleep((attempt + 1) * 2)
                    else:
                        break
                except Exception as e:
                    logger.warning("Error generating reply from %s: %s", model, e)
                    time.sleep(1)

        return "I don't have information on that.", retrieved


def run_rag_evaluation(
    questions_path: Optional[str] = None,
    results_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run RAG evaluation on practice/questions.csv and produce Step 2 deliverables."""
    if questions_path is None:
        candidates = [
            os.path.join(BASE_DIR, "Capstone_Project_Dataset_FTTT_2026", "practice", "questions.csv"),
            os.path.join(BASE_DIR, "practice", "questions.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                questions_path = p
                break

    if results_dir is None:
        results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(questions_path, dtype=str)
    total = len(df)

    rag = ZentroMartRAG()

    chunks_log_file = os.path.join(results_dir, "step2_chunks.jsonl")
    f_chunks = open(chunks_log_file, "w", encoding="utf-8")

    correct_count = 0
    predictions = []
    failures = []

    print(f"\n--- Running Step 2: RAG Agent Evaluation ({total} questions) ---", flush=True)

    for idx, row in df.iterrows():
        qid = row.get("id", f"Q{idx+1}")
        asr_input = row.get("asr_text")
        if not asr_input or pd.isna(asr_input) or not str(asr_input).strip():
            asr_input = row.get("question", "")

        reply, retrieved = rag.answer_question(str(asr_input))

        # Log retrieved chunks to step2_chunks.jsonl
        chunk_entry = {
            "question_id": qid,
            "asr_text": asr_input,
            "retrieved_chunk_ids": [c["section_id"] for c in retrieved],
            "retrieved_sources": [c["source_file"] for c in retrieved],
            "retrieved_scores": [c["re_ranked_score"] for c in retrieved],
            "agent_reply": reply
        }
        f_chunks.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")
        f_chunks.flush()

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
        else:
            failures.append({
                "id": qid,
                "question": row.get("question", ""),
                "asr_text": asr_input,
                "retrieved_chunks": [c["section_id"] for c in retrieved],
                "retrieved_sources": [c["source_file"] for c in retrieved],
                "agent_reply": reply,
                "must_include": must_inc,
                "must_not_include": must_not_inc,
                "failure_reason": reason,
                "gold_evidence": row.get("gold_evidence", "")
            })

        predictions.append({
            "id": qid,
            "asr_text": asr_input,
            "agent_reply": reply,
            "is_correct": is_correct,
            "reason": reason,
            "must_include": must_inc,
            "must_not_include": must_not_inc,
            "retrieved_chunks": "; ".join([c["section_id"] for c in retrieved]),
            "gold_evidence": row.get("gold_evidence", "")
        })

        status_sym = "[PASS]" if is_correct else "[FAIL]"
        print(f"[{idx+1:02d}/{total:02d}] {qid} {status_sym} -> {reason}", flush=True)
        time.sleep(0.15)

    f_chunks.close()

    step2_acc = round((correct_count / total) * 100.0, 2)

    # Read Step 1 score if available
    step1_acc = 48.08
    step1_score_file = os.path.join(results_dir, "step1_score.txt")
    if os.path.exists(step1_score_file):
        try:
            content = open(step1_score_file, encoding="utf-8").read()
            m = re.search(r'Accuracy:\s*([\d\.]+)%', content)
            if m:
                step1_acc = float(m.group(1))
        except Exception:
            pass

    # Save Step 2 score
    step2_score_text = (
        f"Step 2 — RAG Evaluation Score\n"
        f"----------------------------------------\n"
        f"Total Questions Evaluated: {total}\n"
        f"Correct Answers:          {correct_count}\n"
        f"Failed Answers:           {total - correct_count}\n"
        f"Step 2 (RAG) Accuracy:    {step2_acc}%\n"
        f"Step 1 (Baseline) Acc:    {step1_acc}%\n"
        f"Accuracy Gain:            +{round(step2_acc - step1_acc, 2)}%\n"
        f"Embedding Model:          models/gemini-embedding-001\n"
        f"Vector Store:             FAISS (IndexFlatIP, normalized cosine)\n"
        f"Top Chunks Retrieved:     5\n"
    )

    step2_score_file = os.path.join(results_dir, "step2_score.txt")
    with open(step2_score_file, "w", encoding="utf-8") as f:
        f.write(step2_score_text)

    # Save failures analysis
    failures_file = os.path.join(results_dir, "step2_failures.txt")
    with open(failures_file, "w", encoding="utf-8") as f:
        f.write(f"Step 2 — Failure Analysis ({len(failures)} failures out of {total} questions)\n")
        f.write("=" * 70 + "\n\n")
        for fail in failures:
            f.write(f"Question ID: {fail['id']}\n")
            f.write(f"Original Question: {fail['question']}\n")
            f.write(f"ASR Input: {fail['asr_text']}\n")
            f.write(f"Gold Evidence: {fail['gold_evidence']}\n")
            f.write(f"Retrieved Chunks: {fail['retrieved_chunks']}\n")
            f.write(f"Retrieved Sources: {fail['retrieved_sources']}\n")
            f.write(f"Agent Reply: {fail['agent_reply']}\n")
            f.write(f"Must Include: {fail['must_include']}\n")
            f.write(f"Must Not Include: {fail['must_not_include']}\n")
            f.write(f"Why It Failed: {fail['failure_reason']}\n")
            f.write("-" * 70 + "\n\n")

    # Print Side-by-Side comparison
    print("\n" + "=" * 60, flush=True)
    print("STEP 1 vs STEP 2 ACCURACY COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"  Step 1 (LLM Baseline, Prompt-only): {step1_acc:>6.2f}% ({int(step1_acc * total / 100)}/{total})", flush=True)
    print(f"  Step 2 (RAG + FAISS + KB Grounding): {step2_acc:>6.2f}% ({correct_count}/{total})", flush=True)
    print(f"  Performance Improvement:            +{step2_acc - step1_acc:>6.2f}%", flush=True)
    print("=" * 60, flush=True)
    print(f"Score saved to: {step2_score_file}", flush=True)
    print(f"Retrieved chunks logged to: {chunks_log_file}", flush=True)
    print(f"Failure analysis saved to: {failures_file}\n", flush=True)

    return {
        "step1_acc": step1_acc,
        "step2_acc": step2_acc,
        "correct": correct_count,
        "total": total,
        "failures": failures
    }


if __name__ == "__main__":
    run_rag_evaluation()
