# VoiceOps 2026: ZentroMart AI Customer Support Voice Agent

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Swayam318/voiceops2026)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Course](https://img.shields.io/badge/IIT%20Guwahati-Gnani.ai%20Capstone-orange.svg)](https://www.iitg.ac.in/)

- **GitHub Repository:** [https://github.com/Swayam318/voiceops2026](https://github.com/Swayam318/voiceops2026)
- **Course:** LLM & Generative AI Course · IIT Guwahati × Gnani.ai
- **Demo Video:** `demo.mp4` (2-minute video walkthrough included in root)
- **Executive Report:** `report.pdf` (4-page comprehensive engineering report)

An enterprise-grade, speech-enabled AI voice support agent built for **ZentroMart**, an Indian e-commerce platform. VoiceOps 2026 automates complex customer support inquiries across order tracking, refund status, return eligibility, product catalog queries, and human supervisor escalation using **Gnani Speech-to-Text (STT)**, **Gnani Text-to-Speech (TTS)**, a **Grounded RAG Pipeline (FAISS + Google Embeddings)**, and **Stateful Tool Calling** with caller identity verification.

---

## Architecture Overview

```
                      +-----------------------------+
                      |     Customer Speech Input   |
                      |          (.wav audio)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Gnani STT (Speech-to-Text)|
                      |    Model: en-IN (English)   |
                      +--------------+--------------+
                                     | Transcript
                                     v
                      +-----------------------------+
                      |    ZentroMart Voice Agent   |
                      |   (Gemini 3.5 Flash-Lite)   |
                      +-------+--------------+------+
                              |              |
           Query Context      |              | Verified Actions
                              v              v
          +-----------------------+      +---------------------------+
          |      RAG Engine       |      |   Verified Tools Wrapper  |
          |  (FAISS + Embeddings) |      |   - Caller Verification   |
          |  20 Knowledge Base    |      |   - Order Lookup          |
          |  Markdown Documents   |      |   - Safe Ticket Creation  |
          +-----------------------+      |   - Delivery Pincode Join |
                                         |   - Return Window Logic   |
                                         +---------------------------+
                              |              |
                              +-------+------+
                                      |
                                      v Spoken Text Formatting
                      +-----------------------------+
                      |  Spoken Text Sanitizer      |
                      |  - Spaced order numbers     |
                      |  - Rupee symbol expansion   |
                      |  - Sentence length bounds   |
                      +--------------+--------------+
                                     | Formatted String
                                     v
                      +-----------------------------+
                      |   Gnani TTS (Text-to-Speech)|
                      |    Voice: Kaveri            |
                      |    Model: timbre-v2.5       |
                      +--------------+--------------+
                                     |
                                     v Spoken Audio Reply
                      +-----------------------------+
                      |     Customer Audio Output   |
                      |          (.wav reply)       |
                      +-----------------------------+
```

---

## Milestone Performance Comparison

| Milestone | Architecture / Pipeline | Correct | Accuracy (%) | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Step 1: Baseline** | LLM Only (Ungrounded Prompt-Only) | 25 / 52 | **48.08%** | Baseline |
| **Step 2: RAG** | FAISS Vector Index + Google Embeddings | 35 / 52 | **67.31%** | **+19.23%** |
| **Step 3: Full Agent** | RAG + Verified Tools + State + Coreference | 45 / 52 | **86.54%** | **+38.46%** |

### Multi-Turn Conversation Performance (Step 3 Full Agent)
- **Benchmark:** `practice/conversations.csv` (23 multi-turn customer dialogues, 96 total turns)
- **Turn-Level Pass Rate:** **92.71%** (89 PASS, 7 PARTIAL, 0 FAIL)
- **Perfect Conversations:** **16 / 23 (69.57%)** complete dialogs passed with 100% precision
- **Safety & Compliance:** 0 unauthorized data disclosures, 0 fictitious ticket creations, 100% identity verification adherence

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10 to 3.13
- A valid **Gnani AI** API key
- A valid **Google Gemini** API key

### 2. Environment Setup & Dependency Installation
Clone the repository and install dependencies:
```bash
# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies with exact pinned versions
pip install -r requirements.txt
```

### 3. API Key Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your API keys in `.env` (or `project.env`):
```ini
GNANI_API_KEY=your_gnani_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

> **Security Note:** `.env`, `project.env`, and `kb_index/` are included in `.gitignore` to prevent leaking private credentials or binary index files to public repositories.

---

## Running the Batch Audio Runner

The batch runner processes incoming `.wav` audio files through the complete end-to-end voice pipeline:
```bash
python run.py --input_dir ./audio_inputs/ --output_dir ./outputs/
```

### Optional Command-Line Arguments
- `--language`: STT and TTS language code (default: `en-IN`)
- `--voice`: Gnani TTS voice name (default: `Kaveri`)
- `--model`: Gnani TTS model (default: `timbre-v2.5`, fallback: `timbre-v3.0`)
- `--reset_per_file`: Resets agent state and caller memory between different audio files (default: `True`)

### Output Files Generated Per Audio Input
For each `<filename>.wav` in the input directory, the runner outputs 4 files in `--output_dir`:
1. `<filename>_transcript.txt` — Raw transcript recovered by Gnani STT.
2. `<filename>_reply.txt` — Natural language text response generated by the agent.
3. `<filename>_reply.wav` — High-fidelity synthesized speech reply generated by Gnani TTS.
4. `<filename>_chunks.json` — Knowledge base chunk IDs retrieved and tools invoked during the turn.

---

## Adding Your Own Audio Files for Testing

To run the voice pipeline on custom customer queries:
1. Place standard 16kHz mono or stereo `.wav` audio recordings into the `audio_inputs/` folder.
2. Example filenames:
   - `audio_inputs/order_status_check.wav`
   - `audio_inputs/laptop_return_query.wav`
3. Execute the batch runner:
   ```bash
   python run.py --input_dir ./audio_inputs/ --output_dir ./outputs/
   ```
4. Check the results in `outputs/` for the audio replies and JSON logs.

---

## Running Reusable Evaluations

Run the evaluation suite across questions, multi-turn conversations, or both:

```bash
# Evaluate standalone policy & order questions:
python src/evaluate.py --mode questions

# Evaluate multi-turn conversations:
python src/evaluate.py --mode conversations

# Run complete evaluation and print/save the 3-way milestone comparison table:
python src/evaluate.py --mode both
```

---

## Project Structure

```
├── Capstone_Project_Dataset_FTTT_2026/   # Ground-truth dataset (CSVs, KB, tools)
│   ├── data/                             # customers.csv, orders.csv, products.csv, etc.
│   ├── kb/                               # 21 Policy, FAQ, Agent markdown files
│   ├── practice/                         # questions.csv, conversations.csv
│   └── tools/                            # Mock tools implementation
├── src/
│   ├── agent.py                          # Full ZentroMart agent (state, memory, prompt)
│   ├── evaluate.py                       # Reusable benchmark evaluation script
│   ├── llm_baseline.py                   # Step 1 baseline ungrounded LLM
│   ├── rag.py                            # Step 2 FAISS chunking, indexing & retrieval
│   ├── tools_wrapper.py                  # Step 3 verified caller tools wrapper
│   └── voice.py                          # Step 4 Gnani STT and TTS voice layer
├── audio_inputs/                         # Input audio recordings (.wav)
├── sample_outputs/                       # Verification outputs (.txt, .wav, .json)
├── results/                              # Score reports and milestone comparisons
│   ├── step1_score.txt
│   ├── step2_score.txt
│   ├── step3_results.json
│   ├── step3_questions_score.txt
│   └── three_way_comparison.txt
├── .env.example                          # Environment template
├── .gitignore                            # Repository ignore rules
├── requirements.txt                      # Pinned production dependencies
├── run.py                                # CLI batch audio runner
└── README.md                             # Project documentation
```
