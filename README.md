# energy-agent

A tool-calling AI agent for German electricity market processes (GPKE: supplier switch,
start/end of supply, default supply). It answers questions with retrieval over public
BNetzA documents, calculates process deadlines and looks up switching requests in a test
database. The same agent runs against a hosted model (Mistral) and a local model (Ollama).

**Results in short** (29 hand-checked golden questions, details in [Evaluation](#evaluation)):
Mistral `ministral-8b-latest` via API answers 86–90 % correctly in ~2 s; the same model
running locally in Ollama on a 16 GB MacBook reaches 79 % in ~14 s. A loop-level guardrail
against answers from memory lifts a weak local model (qwen3:8b) from 41 % to 59 %.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (installs Python 3.12 automatically), Docker
(e.g. OrbStack) and [Ollama](https://ollama.com).

```bash
uv sync
cp .env.example .env                        # then add your Mistral API key
ollama pull bge-m3                          # local embedding model
docker compose up -d                        # Postgres 17 + pgvector
uv run python scripts/download_sources.py   # BNetzA PDFs -> data/raw/
uv run energy-agent ingest                  # parse, chunk, embed, store (~1 min)
uv run energy-agent seed                    # test data for supplier switch requests
uv run energy-agent search "Wer beliefert einen Kunden in der Ersatzversorgung?"
uv run energy-agent tool calculate_deadline '{"process": "lieferbeginn", "step": 6, "step1_sent_at": "2026-12-23T10:00"}'
uv run energy-agent ask "Wie ist der Status von LW-2026-0003 und bis wann muss der NB antworten?"
uv run energy-agent ask --provider ollama "..."   # same agent, local model (ollama pull ministral-3:8b)
uv run pytest                               # unit + integration tests
```

Example run (Mistral `ministral-8b-latest`, free API tier):

```
$ uv run energy-agent ask "Wie ist der Status der Wechselanfrage LW-2026-0003 und bis wann muss der Netzbetreiber spätestens antworten?"
  · LLM call 1: 593 ms
  → get_switch_status({"request_id": "LW-2026-0003"})  [ok]
  · LLM call 2: 1040 ms
  → calculate_deadline({"process": "lieferbeginn", "step": 6, "step1_sent_at": "2026-10-26T17:20:00"})  [ok]
  · LLM call 3: 1205 ms

Die Wechselanfrage LW-2026-0003 (MaLo-ID: 51002003005) ist aktuell angemeldet und wartet auf die
Antwort des Netzbetreibers (Testnetz Rhein GmbH). Der Netzbetreiber muss die Anmeldung spätestens
bis zum 27.10.2026, 11:00 Uhr ablehnen oder bestätigen.
Quelle: GPKE Teil 2 (Anlage 1b zu BK6-22-024), 2.1.2 SD: Lieferbeginn, Nr. 6, S. 21.

[3 LLM calls, 2 tool calls, 4656+233 tokens, 2.8 s, trace 6aa7b1ca7d82 in traces.jsonl]
```

## Web demo

`energy-agent serve` starts a small FastAPI app on `127.0.0.1:8000`: one page with a
question box, a switch between the Mistral API and the local model, and a live view of
every agent step (model calls, tool calls with arguments and results, guardrail) streamed
as server-sent events, followed by the answer.

It is published as `https://demo.example.com` through a Cloudflare Tunnel from the
laptop, so it works while the laptop is on and costs nothing:

```bash
scripts/demo.sh    # database, Ollama, tunnel (if configured), server; keeps the Mac awake
```

The page protects itself because it is reachable from the internet: HTTP Basic auth on
every route (the server refuses to start without `DEMO_PASSWORD`), a global limit of
6 questions per minute and 150 per day, one question at a time, 500 characters per
question, `noindex` headers and no API docs. The tunnel exposes only this port, and the
agent's tools only read data. Tunnel setup: [`deploy/cloudflared.example.yml`](deploy/cloudflared.example.yml).

## Agent and LLM layer

```
question ─▶ Agent ──messages + tool schemas──▶ ChatModel ─┬─ MistralChat (hosted, EU)
              ▲                                            └─ OllamaChat  (local)
              │  tool calls
              ▼
     search_documents · calculate_deadline · get_switch_status
              │
              └─▶ Tracer ──▶ logs/traces.jsonl (one JSON object per step)
```

- **Plain tool-calling loop** ([`agent.py`](src/energy_agent/agent.py)): send messages and
  tool schemas, run the tool calls the model returns, append the results, repeat until the
  model answers in text. The step budget is 6; the last step offers no tools, which forces
  an answer. No framework: the loop is ~60 lines and every step is visible.
- **Guardrail: no answers from memory.** A prompt can ask for tool use but not enforce it,
  and local models often skip the tools. The loop rejects an answer given before any tool
  returned a result, once, with a reminder; if the model still answers without tool results,
  the result is marked `grounded=False` (shown as a warning in the CLI, counted in the eval).
  It guarantees a tool call, not that the answer uses it: qwen3:8b searched after the
  reminder and then repeated its earlier answer. Checking the answer against the tool
  results is part of the evaluation. `Agent(require_tool_use=False)` switches it off.
- **Provider-agnostic** ([`llm/`](src/energy_agent/llm)): the agent only knows `Message`,
  `ToolCall` and the `ChatModel` protocol. The adapters handle the differences: Mistral
  returns tool arguments as a JSON string and assigns call ids; Ollama returns a JSON object
  and no ids, and its tool results are matched by tool name. The Mistral adapter retries
  `429` responses with `Retry-After`; the Ollama adapter raises the context window to 16k
  tokens (the default would silently truncate retrieved passages) and turns off qwen3's
  thinking mode.
- **Tracing** ([`tracing.py`](src/energy_agent/tracing.py)): every run writes `run_start`,
  `llm_call` (latency, tokens, requested tools), `tool_call` (arguments, result, error) and
  `run_end` events with a shared `run_id` to `logs/traces.jsonl`.
- **Tests** use a scripted fake model for the loop and mocked HTTP for both adapters, so they
  run without network or API key.

## Evaluation

**Golden set:** 29 hand-checked questions in [`eval/golden.yaml`](eval/golden.yaml):
11 deadlines, 5 switch request lookups, 3 combined (lookup, then deadline), 9 questions
answered from the documents, 1 out of scope. Expected dates were derived from the GPKE
wording and the working-day rules; `tests/test_golden.py` checks that they agree with the
deadline tool. "Today" is fixed to 2026-10-27.

**Checks** ([`checks.py`](src/energy_agent/evaluation/checks.py)) are deterministic, with
no LLM judge, so runs are cheap and reproducible:

| Check | Passes when |
|---|---|
| Tool use | the required tools were called |
| Deadline args | `calculate_deadline` got the right process and step |
| Answer | expected dates, times and keywords appear; forbidden words do not |
| Faithful | no date or time in the answer is missing from the question and tool results, and every weekday written next to a date is correct |
| Cites source | rule and deadline answers name the document and section or page |

A question passes when tool use, arguments, answer and faithfulness pass.

### Results

Guardrail on unless noted; 16 GB MacBook (Apple M5) for the local models.

| Model | Runs on | Pass | Deadlines | Lookups | Combined | Documents | Faithful | s / question |
|---|---|---|---|---|---|---|---|---|
| `ministral-8b-latest` | Mistral API | **86–90 %** (4 runs) | 10/11 | 5/5 | 3/3 | 7/9 | 97 % | 1.8 |
| `ministral-8b-latest`, no guardrail | Mistral API | 79–86 % (4 runs) | 10/11 | 5/5 | 3/3 | 6/9 | 93 % | 1.9 |
| `ministral-3:8b` (same weights, 4-bit) | Ollama | **79 %** | 10/11 | 4/5 | 1/3 | 7/9 | 100 % | 13.8 |
| `qwen3:14b` | Ollama | 72 % | 9/11 | 5/5 | 0/3 | 6/9 | 93 % | 24.3 |
| `qwen3:8b` | Ollama | 59 % | 8/11 | 3/5 | 0/3 | 5/9 | 90 % | 12.6 |
| `qwen3:8b`, no guardrail | Ollama | 41 % | 8/11 | 3/5 | 0/3 | 0/9 | 90 % | 10.8 |

For Mistral the category columns show the latest run. `ministral-3:14b` is not listed: Ollama did not parse its tool calls, the model printed them
as text (`calculate_deadline[ARGS]{...}`). Raw results per question: [`eval/results/`](eval/results).

Retrieval alone (9 document questions, no LLM, `energy-agent eval-retrieval`):

| Search | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| vector (bge-m3) | 0.33 | 0.56 | 0.78 | 0.49 |
| full-text (German) | 0.33 | 0.33 | 0.56 | 0.38 |
| hybrid (RRF) | 0.33 | 0.56 | 0.67 | 0.47 |

### What the numbers show

- **Same weights, different result.** Ministral 8B scores 86–90 % through the API and 79 %
  as a 4-bit model in Ollama, at 7× the latency. The likely causes are the quantisation and
  Ollama's chat template for tool calls; this eval does not separate the two.
- **Bigger is not better here.** qwen3:14b is slower and less accurate than ministral-3:8b.
- **The guardrail matters for weak tool users.** It fired in 9 of 29 questions for
  qwen3:8b, which then called the required tools in 79 % of questions instead of 52 %;
  document questions went from 0/9 to 5/9 and the pass rate from 41 % to 59 %. For Mistral
  it fired once in 29 questions, and the difference is within the run-to-run spread.
- **Multi-step questions separate the models.** After looking up a request, local models
  often skip the deadline tool or pick the wrong step (Nr. 2 "Information an den LFA",
  07:00, instead of Nr. 5/6, 11:00).
- **The faithfulness check catches restated errors.** qwen3 wrote "Montag, 11.06.2025"
  (a Wednesday) on top of a correct tool result; Mistral once gave 06.06.2025 from the
  system prompt instead of the documents.
- **Two document questions fail for every model, and the cause is retrieval.** For the
  definitions of Werktag and Zuordnungsbeginn the relevant chunk is at rank 5–7 in vector
  search and outside the top 10 in hybrid search, so no model saw it. On this small set
  hybrid search does not beat pure vector search.

### Limitations and next steps

- 29 questions: one question is 3.4 points, and Mistral varies by up to 10 points between
  runs at temperature 0. Differences below ~10 points are noise.
- Keyword checks cannot tell whether a document answer is supported by the retrieved
  passages (`rag-ohne-vertrag` passes although the expected section was not retrieved).
  Next: passage-level citation checks or an LLM judge on a sample.
- Corpus gap: BNetzA "Mitteilung Nr. 4" moved the go-live from 01.04.2025 (the date in the
  decision) to 06.06.2025; it is not part of the corpus.
- Improvements to measure with this set: one chunk per glossary definition and a reranker
  for retrieval; a "timeline" mode of `calculate_deadline` that returns all steps of a
  process so the model does not have to pick step numbers; a fallback parser for tool calls
  printed as text.

```bash
uv run energy-agent eval --provider mistral                    # writes eval/results/*.json
uv run energy-agent eval --provider ollama --model qwen3:8b --no-guardrail
uv run energy-agent eval-retrieval
uv run pytest -m eval                                          # fails below 80 % pass rate
```

## Retrieval (RAG)

```
PDF ──PyMuPDF──▶ Markdown pages ──▶ section-aware chunks ──bge-m3 (Ollama)──▶ Postgres
                                                                   ├─ vector(1024), HNSW index
                                                                   └─ tsvector('german'), GIN index
query ──▶ vector top-20 ─┐
      └─▶ full-text top-20 ─┴─▶ Reciprocal Rank Fusion ──▶ top-k chunks
```

- **Parsing:** PyMuPDF renders each page as Markdown, so the GPKE process tables
  (`Nr. | Aktion | Frist | Hinweis`) keep their columns. Page footers are removed.
- **Chunking** ([`chunking.py`](src/energy_agent/rag/chunking.py)): split at headings, then
  packed to ~1,500 characters. Each chunk stores its section path
  (`2 Zuordnungsprozesse > 2.1 Use-Case: Lieferbeginn > 2.1.2 SD: Lieferbeginn`) and page range;
  split tables repeat their header row. 4 documents, 231 pages → 464 chunks.
- **Embeddings:** `bge-m3` (multilingual, German-capable) runs locally via Ollama. The document
  title and section path are embedded together with the chunk text.
- **Hybrid search** ([`search.py`](src/energy_agent/rag/search.py)): vector similarity finds
  paraphrases, German full-text search finds exact terms such as `MaLo-ID`; both rankings are
  merged with Reciprocal Rank Fusion, which needs ranks only, not comparable scores.

Note: PyMuPDF is AGPL-licensed, which is fine for this open-source project; a closed-source
product would need its commercial licence or another PDF parser.

## Agent tools

| Tool | What it does | Backed by |
|---|---|---|
| `search_documents` | Hybrid search, returns passages with document, section and page | pgvector + German full-text |
| `calculate_deadline` | Deadline of a GPKE process step, with explanation and citation | Rule table from GPKE Teil 2 + working-day calendar |
| `get_switch_status` | Supplier switch request by request ID or MaLo-ID | `switch_requests` test table (fictional data) |

- **Deadlines are code, not model reasoning.** Language models are unreliable at date
  arithmetic, so each rule restates one "Frist" cell of the GPKE process tables
  ([`deadlines.py`](src/energy_agent/tools/deadlines.py)) and returns its source, e.g.
  *GPKE Teil 2, 2.1.2 SD: Lieferbeginn, Nr. 6, S. 21*. Covered: MaLo-ID request,
  termination, supply start, supply end (by supplier and by grid operator), default supply.
  Non-EEG market locations only.
- **Werktage** ([`workdays.py`](src/energy_agent/tools/workdays.py)) follow GPKE Teil 1,
  Kap. 7: a holiday in any federal state counts nationwide, and 24.12./31.12. count as
  holidays. The calendar is the union of all state holidays from 2023 onwards, including
  one-off ones such as 8 May 2025 in Berlin.
- **Interpretation of the LFW24 registration rule** ("spätester ÜT ist der Tag vor dem
  letzten WT vor dem Zuordnungsbeginn"): a registration sent on Monday allows supply from
  Wednesday, one sent on Friday from Tuesday. Derived from the wording; the BNetzA decision
  leaves worked examples to BDEW guidance, which is not part of this corpus.
- **Tool calls are validated.** Arguments are pydantic models, which also generate the
  JSON schemas sent to the model. Invalid arguments and domain errors (unknown step, wrong
  MaLo-ID check digit) come back to the model as `{"error": ...}` so it can retry.

## Data

The corpus is the current GPKE as published by the Bundesnetzagentur with decision
[BK6-22-024](https://www.bundesnetzagentur.de/DE/Beschlusskammern/1_GZ/BK6-GZ/2022/BK6-22-024/BK6-22-024_Verfahren.html)
(supplier switch within 24 hours, in force since 2025-06-06): the decision itself and
GPKE parts 1, 2 and 4. The PDFs are not stored in the repo; URLs and SHA-256 checksums are
pinned in [`data/sources.toml`](data/sources.toml).

```bash
uv run python scripts/download_sources.py   # downloads to data/raw/ and verifies checksums
```
