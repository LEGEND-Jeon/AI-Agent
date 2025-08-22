# AI Agent & RAG Study

---

## 1. Introduction

### Overview
- RAG (Retrieval-Augmented Generation) = retrieval of external knowledge + LLM-based generation  
- AI Agent = LLM with agentic behaviors (query rewriting, decision-making, validation)  
- Agentic RAG = extends RAG with autonomous loops for self-critique, retrieval, and revision  

---

## 2. Key Concepts

### RAG
- Retriever: fetches context from vector DB, APIs, internet  
- Generator (LLM): produces responses based on query + retrieved context  
- Advantage: higher factuality, reduced hallucination, up-to-date knowledge  

### AI Agent
- Beyond simple generation:  
  - Query rewriting  
  - Decide if more details are needed  
  - Choose proper knowledge source  
  - Validate and critique output  

### Agentic RAG
- Adds agentic loop on top of RAG  
- Steps include:  
  1. Rewrite query  
  2. Decide if more details are required  
  3. Retrieve context from sources  
  4. Generate draft answer  
  5. Judge validates correctness  
  6. If failed → loop back and retry  
- Benefits: Robust, adaptive, human-like reasoning flow  

---

## 3. gentic RAG Diagram

<p align="center">
  <img src="https://embed.filekitcdn.com/e/k7YHPN24SoxyM8nGKZnDxa/81cg2zGi6cLRVmnNVQNV9r/email" width="600" alt="Agentic RAG Diagram"/>
</p>

---

## 4. Case Study: Travel Itinerary Example

### User Query
> “Please generate a 10-day itinerary (2025-07-14 to 2025-07-23) covering Paris, Madrid, and Lisbon.”

---
<img width="1442" alt="image" src="https://github.com/LEGEND-Jeon/AI-Agent/blob/main/examplerag.png?raw=true">
<br>

### Step 1 – Heavy Generator (Draft Itinerary)
Day 1 (7/14): Versailles in the afternoon  
Day 2 (7/15): Louvre, Eiffel Tower  
Day 7 (7/20): Travel Madrid → Lisbon (2h)  

---

### Step 2 – Light Judge #1 (Feedback)
Issues Found
- Day 1: Versailles closed on Monday (7/14)  
- Day 2: Louvre closed on Tuesday (7/15)  
- Day 7: Travel time underestimated (~4h needed)  

---

### Step 3 – Heavy Generator (Revised Itinerary)
Day 1 (7/14): Montmartre walk  
Day 2 (7/15): Eiffel Tower, Seine River cruise  
Day 3 (7/16): Louvre Museum  
Day 7 (7/20): Travel Madrid → Lisbon (~4h)  

---

### Step 4 – Light Judge #2 (Feedback)
Feedback
- Grounded in context  
- No further issues → ✅ Approved  

---

### Final Output
July 14–23, 2025  
10-day Paris–Madrid–Lisbon itinerary (considering opening hours and travel time)  

---

## 5. Learnings

- Agents improve robustness: Iterative feedback reduces hallucinations and factual errors  
- Heavy Generator (e.g., GPT-4) + Light Judge (e.g., GPT-3.5) balance cost and quality  
- Verification loops are crucial for real-world applications (travel planning, Q&A, enterprise search)  

---

## 6. References
- Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020)  
- Yunfan Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023)  
- Aditi Singh et al., *Agentic RAG: A Survey on Agentic Retrieval-Augmented Generation* (2025)  
- [Daily Dose of Data Science: RAG vs Agentic RAG](https://www.dailydoseofds.com/p/rag-vs-agentic-rag)  

---
