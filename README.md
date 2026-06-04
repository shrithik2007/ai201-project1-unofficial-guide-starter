# The Unofficial Guide — UTSA Campus Dining RAG System

A RAG (Retrieval-Augmented Generation) system that makes student-generated campus dining knowledge searchable. Ask plain-language questions like "Is the pizza at Roadrunner good?" or "What can I eat after 9pm?" and get grounded, cited answers drawn from real student reviews, Reddit threads, and peer guides.

---

## Setup

```bash
# 1. Clone your fork and enter the directory
cd unofficial-guide

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
source .venv/Scripts/activate      # Windows (Git Bash)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# Edit .env and replace your_key_here with your key from console.groq.com

# 5. Build the vector store (one-time)
python embed.py

# 6. Launch the app
python app.py
# Open http://localhost:7860
```

---

## Domain and Document Sources

**Domain:** Campus dining at UTSA — the practical, experience-based knowledge that students share with each other but that doesn't appear on the official dining website. This includes which stations are worth waiting for, when to avoid crowds, how meal plans actually work, what to eat with dietary restrictions, and where to find the best value on campus.

**Why it's hard to find officially:** UTSA Dining's website publishes menus, hours, and pricing. It does not publish crowd patterns, honest quality assessments, meal plan gotchas, or student workarounds. Students learn this from Reddit, Discord, older students, and trial and error.

### Source Documents (12 files in `documents/`)

| File | Source Description |
|------|--------------------|
| `roadrunner_cafe_reviews.txt` | Reddit r/UTSA thread — 10 student reviews of the main dining hall |
| `dining_hours_and_locations.txt` | UTSA Student Forum crowdsourced guide — hours, locations, payment types |
| `meal_plan_guide.txt` | Reddit r/UTSA megathread — which plan to choose and how to avoid common mistakes |
| `sombrilla_food_court_reviews.txt` | UTSA Discord #campus-food — reviews of each Sombrilla vendor |
| `dietary_restrictions_guide.txt` | Campus Life Blog + Reddit — vegan, gluten-free, halal, nut allergy guidance |
| `best_worst_foods_ranked.txt` | Reddit r/UTSA ranking thread — crowd-voted best and worst items |
| `dining_hacks_and_tips.txt` | UTSA FirstGen shared Google Doc — timing, money-saving, quality strategies |
| `library_cafe_and_coffee_spots.txt` | Discord + Reddit — campus coffee options and study café reviews |
| `rowdy_cart_and_food_trucks.txt` | Reddit r/UTSA — deep dive on the Rowdy Cart and independent food trucks |
| `freshman_dining_survival_guide.txt` | Peer Mentor Program doc — first-year orientation to campus dining |
| `late_night_food_options.txt` | Reddit r/UTSA — what to eat after 9pm on and near campus |
| `commuter_dining_tips.txt` | Commuter Student Services + Reddit — eating strategies for commuters |

---

## Chunking Strategy

**Chunk size:** 600 characters  
**Overlap:** 100 characters  
**Method:** Sliding window with paragraph-boundary preference

A 600-character chunk captures roughly one full paragraph or 2–3 review sentences — enough semantic content for the embedding to represent a coherent idea, but small enough that unrelated topics don't share a chunk. This corpus is review-and-advice text; most meaningful claims live within 1–3 sentences. Shorter chunks (200–300 chars) would fragment individual reviews mid-thought and produce embeddings with too little signal. Longer chunks (1000+ chars) would mix e.g. stir-fry advice with pizza complaints in one vector, diluting retrieval precision.

The 100-character overlap prevents stranding context at split boundaries. When the splitter detects a newline within the last 20% of a chunk, it breaks there instead of at the character limit — this respects paragraph boundaries and keeps most review entries intact.

### Sample Chunks

**Chunk 1** — `roadrunner_cafe_reviews.txt`
> The breakfast at Roadrunner Café is genuinely the best meal of the day. The made-to-order omelets are fresh and the staff are super friendly in the morning. Avoid going between 11:45am and 1:15pm — the lunch rush is brutal and the lines for the grill station can be 20+ minutes. I usually hit it at 11am or after 1:30pm and never wait more than 5 minutes.

**Chunk 2** — `meal_plan_guide.txt`
> Dining Dollars vs. Rowdy Bucks: Dining Dollars are loaded onto your 1501 card as part of a meal plan, expire at end of semester, and can only be used at campus dining locations. Rowdy Bucks are a general spending account loaded onto the same card, don't expire, and can be used anywhere that accepts 1501 card.

**Chunk 3** — `dietary_restrictions_guide.txt`
> Roadrunner Café has a dedicated vegan station near the back of the dining hall. This station uses separate utensils and is generally considered reliable for avoiding cross-contamination. The menu at the vegan station rotates but typically includes a grain bowl, roasted vegetables, and a plant-based protein option.

**Chunk 4** — `rowdy_cart_and_food_trucks.txt`
> The Rowdy Cart is UTSA Dining's mobile food cart and, according to most students, the best food value on the entire campus. It's operated by campus dining staff and accepts Dining Dollars, Rowdy Bucks, and credit/debit. Location: The cart rotates between several campus spots. The most common locations are near the Convocation Center, outside the Science building (SCI), and near the east entrance of the Rec Center.

**Chunk 5** — `late_night_food_options.txt`
> The hard reality of UTSA's main campus: after 9pm on weekdays (8pm on Fridays, 8pm on weekends), there is no hot food available on campus. The dining options close and don't reopen until morning. What remains: Vending machines in every academic building.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`

This model runs locally with no API key and no rate limits, produces 384-dimensional embeddings, and handles up to 256 tokens (~1000 characters) per chunk — a comfortable fit for our 600-character chunks. Inference on CPU is fast enough for a development system with ~400 chunks.

**Production tradeoffs I'd consider:**
- **OpenAI `text-embedding-3-small`**: Higher accuracy on domain-specific nuance, but costs money and requires API access. For a system serving thousands of daily users, the quality improvement might justify the cost.
- **`multilingual-e5-base`**: Would handle Spanish-language reviews (common in San Antonio student communities) but is ~3x larger and slower on CPU. Worth it if the corpus expands to bilingual content.
- **Context length**: `all-MiniLM-L6-v2` truncates at 256 tokens. If chunking strategy changed to larger chunks, this model would need replacing with one supporting longer context (e.g., `bge-large-en`).
- **Local vs. API**: Local inference (this system) has zero marginal cost and no latency from network calls. API embeddings are simpler to scale but introduce per-query cost and a dependency on external availability.

---

## Retrieval Test Results

### Query 1: "What's the best time to eat lunch at Roadrunner Café to avoid long lines?"

**Top chunks returned:**

1. `roadrunner_cafe_reviews.txt` (distance: 0.18) — "Avoid going between 11:45am and 1:15pm — the lunch rush is brutal and the lines for the grill station can be 20+ minutes. I usually hit it at 11am or after 1:30pm and never wait more than 5 minutes."
2. `dining_hacks_and_tips.txt` (distance: 0.22) — "The single most impactful change you can make to your dining experience: stop eating at peak hours. Peak hours are 11:45am–1:15pm and 5:30pm–7:00pm..."
3. `freshman_dining_survival_guide.txt` (distance: 0.31) — "The dining hall is most crowded 11:45am–1:15pm and 5:30pm–7pm. If your schedule allows even a 30-minute shift, it makes a real difference."

**Why these chunks are relevant:** All three directly address lunch timing at Roadrunner Café. The semantic similarity is high because the query and the chunks share concepts of crowds, timing, and the dining hall. The system correctly surfaces the same advice from three different documents, which gives the LLM enough material to synthesize a confident, well-supported answer.

---

### Query 2: "Do meal swipes work at Chick-fil-A in the Sombrilla?"

**Top chunks returned:**

1. `meal_plan_guide.txt` (distance: 0.21) — "I thought the Chick-fil-A and Panda Express in the Sombrilla accepted meal swipes. They don't. Only Roadrunner Café takes swipes."
2. `dining_hours_and_locations.txt` (distance: 0.24) — "Payment: Dining Dollars, Rowdy Bucks, credit/debit (meal swipes NOT accepted)" [Sombrilla entry]
3. `freshman_dining_survival_guide.txt` (distance: 0.29) — "Not knowing that meal swipes only work at Roadrunner Café: The Sombrilla, Library Café, and food trucks do not accept meal swipes."

**Why these chunks are relevant:** The query's key terms ("meal swipes," "Chick-fil-A," "Sombrilla") appear in close proximity in these chunks. The retrieval correctly identifies three documents that each explicitly answer this exact question — a case where the system works well because the answer is stated clearly in the documents.

---

### Query 3: "What food is there after 9pm on campus?"

**Top chunks returned:**

1. `late_night_food_options.txt` (distance: 0.14) — "after 9pm on weekdays... there is no hot food available on campus. What remains: Vending machines in every academic building."
2. `dining_hours_and_locations.txt` (distance: 0.27) — "LATE NIGHT OPTIONS (after 9pm): Vending machines: All academic buildings, 24/7. The Sombrilla closes at 8pm..."
3. `late_night_food_options.txt` (distance: 0.31) — "McDonald's on Fredericksburg Road: About 7 minutes by car, open 24 hours..."

**Why these chunks are relevant:** "After 9pm" and "on campus" are strong retrieval signals. The top result is the most relevant document in the corpus for this query and surfaces immediately. Distance score of 0.14 is very low — high confidence match.

---

## Grounded Generation

Grounding is enforced at the system prompt level in `generate.py`. The system prompt reads:

> *"Answer the question using ONLY the document excerpts provided below. Do not use any outside knowledge or make up information. If the provided excerpts do not contain enough information to answer the question, say exactly: 'I don't have enough information about that in my documents.'"*

The LLM receives the retrieved chunks as the only context in the user message. Temperature is set to 0.2 to reduce hallucination risk. Source filenames are appended to every response programmatically — they are not left to the model to include or omit.

**Test of grounding (out-of-scope query):** When asked "What is the UTSA dining hall's score on the latest health inspection?", the system responds: *"I don't have enough information about that in my documents."* This is correct — no health inspection data appears in any source document. The model does not attempt to fabricate an answer.

---

## Example Responses

### Example 1 — In-scope query with source attribution

**Query:** "What are the best vegan options at Roadrunner Café?"

**Response:**
> The best vegan option at Roadrunner Café is the dedicated vegan station near the back of the dining hall, which uses separate utensils and is considered reliable for avoiding cross-contamination. It typically includes a grain bowl, roasted vegetables, and a plant-based protein. The salad bar is also vegan-friendly — look for chickpeas, fresh vegetables, and oil-and-vinegar dressings (avoid the creamy dressings, which contain dairy). The stir-fry station can also accommodate vegans if you ask for tofu and specify no meat.

**Sources:** `dietary_restrictions_guide.txt`, `roadrunner_cafe_reviews.txt`

---

### Example 2 — In-scope query with nuanced/conflicting information

**Query:** "Is the pizza at Roadrunner Café good?"

**Response:**
> Student opinion on the pizza is mixed but leans negative. Most reviews describe it as thin, greasy, and sitting too long under heat lamps. However, one reviewer notes it's acceptable if you catch it right when it comes out of the oven, which happens roughly every 40 minutes. The general advice is to skip it unless you time your visit to get a fresh tray.

**Sources:** `best_worst_foods_ranked.txt`, `roadrunner_cafe_reviews.txt`

---

### Example 3 — Out-of-scope query (refusal)

**Query:** "What is the UTSA dining hall's score on the latest health inspection?"

**Response:**
> I don't have enough information about that in my documents.

---

## Query Interface

The interface is a Gradio web app (`app.py`) running at `http://localhost:7860`.

**Input:** A text box labeled "Your question" — type any dining-related question and press Enter or click "Ask."

**Outputs:**
- **Answer** — the LLM's grounded response (2–5 sentences)
- **Retrieved from** — the source documents the answer was drawn from
- **See retrieved chunks** (expandable accordion) — the raw text of the top-5 retrieved chunks with distance scores, for full transparency

**Sample interaction:**

```
Question: Which meal plan should I get as a freshman living on campus?

Answer: For freshmen living on campus, the Unlimited Plan is most commonly 
recommended. While it costs more ($2,450/semester), it gives you unlimited 
swipes at Roadrunner Café, which prevents the problem of running out of 
swipes mid-semester — something students on block plans report happening 
regularly. If you're confident you'll eat 7–8 meals per week at the dining 
hall, the Block 160 plan works but requires weekly tracking.

Retrieved from:
• meal_plan_guide.txt
• freshman_dining_survival_guide.txt
```

---

## Evaluation Report

### Test Question 1
**Question:** What's the best time to eat lunch at Roadrunner Café to avoid long lines?  
**Expected:** Arrive before 11:45am (around 11am) or after 1:15–1:30pm; peak hours are 11:45am–1:15pm.  
**System response:** "The best time to eat lunch at Roadrunner Café is before 11:45am (around 11am) or after 1:15–1:30pm. Peak hours run from 11:45am to 1:15pm, when lines at the grill station can exceed 20 minutes. Multiple student reviews consistently recommend arriving at 11am or waiting until after 1:30pm."  
**Accuracy:** ✅ Accurate  
**Chunks retrieved:** `roadrunner_cafe_reviews.txt` (0.18), `dining_hacks_and_tips.txt` (0.22), `freshman_dining_survival_guide.txt` (0.31)

---

### Test Question 2
**Question:** Do meal swipes work at Chick-fil-A in the Sombrilla?  
**Expected:** No — meal swipes only work at Roadrunner Café; Sombrilla requires Dining Dollars, Rowdy Bucks, or card.  
**System response:** "No, meal swipes do not work at Chick-fil-A in the Sombrilla. Meal swipes are only accepted at Roadrunner Café. At the Sombrilla, you'll need to use Dining Dollars, Rowdy Bucks, or a credit/debit card. This surprises a lot of freshmen the first time they try."  
**Accuracy:** ✅ Accurate  
**Chunks retrieved:** `meal_plan_guide.txt` (0.21), `dining_hours_and_locations.txt` (0.24), `freshman_dining_survival_guide.txt` (0.29)

---

### Test Question 3
**Question:** What are the best vegan food options on campus?  
**Expected:** Dedicated vegan station at the back of Roadrunner Café; salad bar (chickpeas, vegetables, oil-vinegar dressing); stir-fry station with tofu on request.  
**System response:** "The best vegan options are the dedicated vegan station at the back of Roadrunner Café (separate utensils, reliable for cross-contamination avoidance), the salad bar (chickpeas, fresh vegetables, oil-and-vinegar dressing — avoid creamy dressings), and the stir-fry station where you can request tofu. One student notes that labeling is sometimes inconsistent outside the dedicated station, so it's worth asking staff directly."  
**Accuracy:** ✅ Accurate  
**Chunks retrieved:** `dietary_restrictions_guide.txt` (0.17), `roadrunner_cafe_reviews.txt` (0.38), `dining_hacks_and_tips.txt` (0.41)

---

### Test Question 4
**Question:** Where is the Rowdy Cart located and how do I find out where it is each day?  
**Expected:** Location rotates; check @UTSADining on Instagram daily; common spots near Convocation Center, Science building, Rec Center east entrance.  
**System response:** "The Rowdy Cart rotates between several campus locations. The most common spots are near the Convocation Center, outside the Science building, and near the east entrance of the Rec Center. To find the daily location, check @UTSADining on Instagram — they post an update each day. Note it doesn't operate on rainy days or during campus events that take over its usual spots."  
**Accuracy:** ✅ Accurate  
**Chunks retrieved:** `rowdy_cart_and_food_trucks.txt` (0.12), `dining_hours_and_locations.txt` (0.29), `dining_hacks_and_tips.txt` (0.33)

---

### Test Question 5
**Question:** What food options are available on campus after 9pm on a weekday?  
**Expected:** Only vending machines on campus; nearest open options are McDonald's (7 min, 24hr), Taco Cabana (8 min, 24hr), Whataburger (12 min, 24hr); DoorDash/Uber Eats deliver to campus.  
**System response:** "After 9pm on weekdays, there is essentially no hot food available on campus — only vending machines remain. The nearest 24-hour options by car are McDonald's on Fredericksburg Road (about 7 minutes), Taco Cabana on Wurzbach Parkway (about 8 minutes), and Whataburger on Loop 1604 (about 12 minutes). DoorDash and Uber Eats deliver to campus until midnight or later; use 'Meet at entrance' to help drivers navigate."  
**Accuracy:** ✅ Accurate  
**Chunks retrieved:** `late_night_food_options.txt` (0.14), `dining_hours_and_locations.txt` (0.27), `late_night_food_options.txt` (0.31)

---

### Failure Case Analysis

**Question 3 showed a partial retrieval weakness.** While the final answer was accurate, the second and third retrieved chunks (`roadrunner_cafe_reviews.txt` at 0.38 and `dining_hacks_and_tips.txt` at 0.41) had noticeably higher distance scores than the top result. This means the system was partly relying on one highly relevant chunk rather than three equally strong ones.

**Root cause:** The query "best vegan food options on campus" is semantically broad. The dedicated vegan document (`dietary_restrictions_guide.txt`) is a strong match, but content about vegan options is scattered across multiple documents in small mentions. The 600-character chunking meant that in `roadrunner_cafe_reviews.txt`, the one sentence mentioning the vegan station was embedded alongside unrelated review content — diluting the vector signal. A smaller chunk size (300–400 chars) might improve retrieval precision for scattered single-sentence facts, at the cost of losing context in longer reviews.

This is a genuine tradeoff: the same chunk size that preserves review context hurts retrieval for sparse, spread-out facts. A hybrid approach (smaller chunks for list-format documents, larger for narrative reviews) would address this but adds pipeline complexity.

---

## Spec Reflection

**One way the spec helped:** Writing the evaluation plan before any code forced me to think about what "correct" means for each question before the system existed. Questions 2 and 4 (meal swipes, Rowdy Cart location) had clear, factual expected answers — those turned out to be the cleanest retrieval cases because the answers were stated explicitly in the documents. The evaluation plan revealed upfront that vague questions would be harder.

**One way implementation diverged from spec:** The spec called for source attribution to be handled by instructing the LLM to cite sources in its response. In practice, I moved attribution to a programmatic step — source filenames are appended from the retrieved metadata after generation, not left to the model. This change was necessary because in early testing the model occasionally omitted citations or hallucinated source names. Programmatic attribution is more reliable than prompt-based attribution.

---

## AI Usage

**Instance 1 — Chunking implementation:** I described my chunking strategy to Claude (600-char chunks, 100-char overlap, paragraph-boundary preference) and asked it to implement the `chunk_text()` function. The generated code used `text.rfind("\n")` to snap to paragraph boundaries, which matched my intent. I modified it to only search for boundaries in the last 20% of the chunk (not the full chunk) — the original searched too aggressively and was producing chunks much shorter than 600 chars on paragraph-heavy documents.

**Instance 2 — System prompt for grounding:** I asked Claude to write a system prompt that would enforce grounding (answer only from context) and produce a clear refusal for out-of-scope questions. The first version included the phrase "as an AI assistant, I can only..." which I cut — it added a disclaimer that made responses feel evasive rather than direct. I replaced it with a plain instruction: "If the provided excerpts do not contain enough information, say exactly: 'I don't have enough information about that in my documents.'" The explicit fallback phrase made the refusal behavior much more consistent across queries.
