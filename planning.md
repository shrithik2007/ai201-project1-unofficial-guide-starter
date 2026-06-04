# planning.md — The Unofficial Guide: UTSA Campus Dining

## Domain

**Campus dining at UTSA** — specifically the kind of practical, experience-based knowledge that students share with each other but that doesn't appear in any official university publication. This includes: which dining stations are worth waiting for, when to avoid peak crowds, how meal plans actually work vs. how UTSA describes them, which food options are best for dietary restrictions, and where to find the best-value food on campus.

This knowledge is hard to find officially because UTSA Dining's website only provides menus, hours, and pricing — not quality assessments, crowd patterns, workarounds, or honest critiques. Students learn this stuff from older students, from Reddit threads, and from trial and error. This system makes it searchable and answerable in plain language.

---

## Documents

10 source documents collected for this system (all stored in `documents/`):

| File | Source | Content |
|------|--------|---------|
| `roadrunner_cafe_reviews.txt` | Reddit r/UTSA thread | 10 student reviews of the main dining hall |
| `dining_hours_and_locations.txt` | UTSA Student Forum crowdsourced guide | Hours, locations, and payment types for every campus dining spot |
| `meal_plan_guide.txt` | Reddit r/UTSA megathread | Honest student advice on which meal plan to choose and how to use it |
| `sombrilla_food_court_reviews.txt` | UTSA Discord #campus-food | Reviews of each vendor in the Sombrilla food court |
| `dietary_restrictions_guide.txt` | Campus Life Blog + Reddit | Vegan, gluten-free, halal, nut allergy guidance for campus dining |
| `best_worst_foods_ranked.txt` | Reddit r/UTSA ranking thread | Crowd-ranked best and worst specific items on campus |
| `dining_hacks_and_tips.txt` | UTSA FirstGen shared doc | Timing, money-saving, and food quality strategies |
| `library_cafe_and_coffee_spots.txt` | Discord + Reddit | Reviews of campus coffee options and study café locations |
| `rowdy_cart_and_food_trucks.txt` | Reddit r/UTSA | Deep dive on the Rowdy Cart and independent food trucks |
| `freshman_dining_survival_guide.txt` | Peer Mentor Program doc | First-year orientation to campus dining |
| `late_night_food_options.txt` | Reddit r/UTSA | What to eat after 9pm on and near campus |
| `commuter_dining_tips.txt` | Commuter Services + Reddit | Campus eating strategies specific to commuter students |

---

## Chunking Strategy

**Chunk size:** 600 characters with 100-character overlap.

**Rationale:** These documents are a mix of paragraph-style reviews (3-6 sentences each) and structured lists. A 600-character chunk captures roughly one full paragraph or 2-3 bullet-point entries — enough semantic content for the embedding to represent a coherent idea, but small enough that a single chunk doesn't mix multiple unrelated topics. Shorter chunks (200-300 chars) would fragment individual reviews mid-thought; longer chunks (1000+ chars) would mix e.g. stir-fry advice with pizza complaints in one embedding, diluting the retrieval signal.

The 100-character overlap prevents losing context at split boundaries — for example, a chunk that ends mid-sentence about dining hours will have enough overlap with the next chunk that both can surface for a query about hours.

This is primarily a review/advice corpus. Most meaningful claims are made within 1-3 sentences. 600 characters accommodates that well.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. Runs locally — no API key, no rate limits, fast on CPU.

**Top-k:** 5 chunks per query.

Retrieving 5 chunks gives the LLM enough context to synthesize an answer from multiple perspectives (e.g., several student opinions about the same dining hall) without flooding the context with loosely related material. Fewer than 3 risks missing the most relevant passage; more than 7 dilutes the context.

**Why semantic search works here:** A query like "Is the pizza at the dining hall good?" won't match the exact phrase "the pizza is mediocre and has been sitting for 45 minutes" — but the embeddings of both sit in similar vector space because they share concepts around dining, food quality, and the same location. This lets the retrieval surface relevant opinions even when the wording differs.

**Tradeoffs for a production deployment:**
- `text-embedding-3-small` (OpenAI) would give better accuracy for domain-specific nuance but costs money and requires API access.
- `multilingual-e5-base` would handle Spanish-language reviews (common in SA communities) but is larger and slower.
- Context length: `all-MiniLM-L6-v2` handles up to 256 tokens (~1000 chars), which fits our 600-char chunks well. For longer chunks it would need replacing.
- For a real deployment with thousands of daily users, cost and latency would favor a hosted embedding API over local inference.

---

## Evaluation Plan

Five test questions with ground-truth answers drawn from the documents:

| # | Question | Expected Answer |
|---|----------|----------------|
| 1 | What's the best time to eat lunch at Roadrunner Café to avoid lines? | Arrive before 11:45am (around 11am) or after 1:15–1:30pm to avoid peak hours |
| 2 | Do meal swipes work at Chick-fil-A in the Sombrilla? | No — meal swipes only work at Roadrunner Café; Sombrilla requires Dining Dollars, Rowdy Bucks, or card |
| 3 | What are the best food options for vegans on campus? | The dedicated vegan station at the back of Roadrunner Café; also salad bar chickpeas/vegetables with oil-vinegar dressing; stir-fry station with tofu on request |
| 4 | Where is the Rowdy Cart and how do I find it daily? | Location rotates — check @UTSADining on Instagram for daily location updates; common spots near Convocation Center, Science building, Rec Center east entrance |
| 5 | What food options are available on campus after 9pm? | Essentially none — only vending machines on campus; nearest open options are McDonald's (7 min drive, 24hr), Taco Cabana (8 min, 24hr), Whataburger (12 min, 24hr); DoorDash/Uber Eats deliver to campus |

---

## Anticipated Challenges

1. **Chunk boundary splitting:** Several documents have structured lists (dining hours, meal plan prices) that may get split mid-entry if the chunker cuts at character boundaries. A chunk that starts mid-entry loses context about which dining location or plan is being described. The overlap partially mitigates this but won't eliminate it.

2. **Inconsistent source attribution:** Different documents have different source formats (Reddit username, Discord handle, anonymous). The metadata system stores source filename but not sub-document attribution. The LLM may cite "roadrunner_cafe_reviews.txt" rather than the specific reviewer — acceptable for this project but less precise than ideal.

3. **Conflicting information:** Multiple reviews sometimes contradict each other (e.g., one student says pizza is always bad, another says it's fine right out of the oven). The LLM will need to surface this nuance rather than picking one answer, which depends on effective prompt design.

---

## AI Tool Plan

| Pipeline Component | AI Tool Input | Expected Output |
|-------------------|---------------|----------------|
| `ingest.py` (document loading + cleaning) | This planning.md §Documents + §Chunking Strategy; description of file format (plain .txt with headers) | Script that loads all .txt files from `documents/`, strips header lines (SOURCE/DATE), splits into chunks of 600 chars with 100-char overlap, returns list of `{text, source, chunk_id}` dicts |
| `embed.py` (ChromaDB setup + embedding) | This planning.md §Retrieval Approach + architecture diagram; output format from ingest.py | Script that initializes ChromaDB locally, loads chunks from ingest.py, embeds with `all-MiniLM-L6-v2`, stores with source metadata |
| `retrieve.py` (query function) | §Retrieval Approach (top-k=5); ChromaDB collection structure from embed.py | Function `retrieve(query: str) -> list[dict]` that returns top-5 chunks with text and source |
| `generate.py` (Groq LLM call) | Grounding requirement from project spec; retrieve.py output format | Function `answer(query: str) -> dict` with keys `answer` and `sources`; system prompt explicitly restricts model to provided context only |
| `app.py` (Gradio UI) | Input/output format from generate.py; project spec UI requirements | Gradio Blocks UI with question input, answer output, sources output, and submit button |
| `evaluate.py` (evaluation runner) | 5 test questions from §Evaluation Plan | Script that runs all 5 questions, prints question + retrieved chunks + system response + manual accuracy judgment template |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE (build time)                  │
│                                                         │
│  documents/*.txt                                        │
│       │                                                 │
│       ▼                                                 │
│  [ingest.py]                                            │
│  Load & clean → chunk (600 char / 100 overlap)          │
│       │                                                 │
│       ▼                                                 │
│  [embed.py]                                             │
│  all-MiniLM-L6-v2 → ChromaDB (local)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE (query time)                   │
│                                                         │
│  User query                                             │
│       │                                                 │
│       ▼                                                 │
│  [retrieve.py]                                          │
│  Embed query → ChromaDB semantic search → top-5 chunks  │
│       │                                                 │
│       ▼                                                 │
│  [generate.py]                                          │
│  Groq llama-3.3-70b-versatile                           │
│  System prompt: answer from context only                 │
│  → Answer + source attribution                          │
│       │                                                 │
│       ▼                                                 │
│  [app.py]                                               │
│  Gradio web UI (localhost:7860)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
