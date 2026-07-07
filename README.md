# Mini Search Engine with RAG Extension
### AI356 — Information Retrieval | Jordan University of Science and Technology

---

##  Team Members

| Name |
|------|
| Samira Aldamen |
| Wajd Almadi |
| Batool Mahashi |

---

##  Project Overview

A classical Information Retrieval system built on the **Cranfield Dataset** (1400 aerodynamics documents), extended with a **RAG (Retrieval-Augmented Generation)** pipeline that uses an LLM to generate natural language answers from retrieved documents.

---

##  Project Structure

```
├── IR_RAG_SearchEngine.ipynb   # Main project notebook (all parts)
├── docs/
│   ├── Project_Report.docx
│   └── Presentation.pptx
├── requirements.txt
├── .gitignore
└── README.md

# Generated at runtime (not tracked in this repo):
├── cran.all.1400           # Cranfield document collection
├── cran.qry                # Test queries (225 queries)
├── cranqrel                # Relevance judgments (ground truth)
├── docs.pkl                # Parsed documents (auto-generated)
├── processed_docs.pkl      # Preprocessed tokens (auto-generated)
├── inverted_index.pkl      # Inverted index (auto-generated)
├── idf.pkl                 # IDF values (auto-generated)
└── tfidf_matrix.pkl        # TF-IDF matrix (auto-generated)
```

> **Note:** The Cranfield dataset files (`cran.all.1400`, `cran.qry`, `cranqrel`) are not included in this repository. Upload them when running the notebook (the first cell handles this via `google.colab.files.upload()`), or download them from the standard Cranfield collection source.

---

##  Requirements

### Python Version
Python 3.8+

### Install Dependencies

```bash
pip install nltk openai
```

### Download NLTK Data (run once)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

---

##  How to Run

> **Note:** The project was developed and tested on **Google Colab**. To run locally, remove all `google.colab` imports and replace `files.upload()` with direct file paths.

### Step 1 — Place Dataset Files
Make sure `cran.all.1400`, `cran.qry`, and `cranqrel` are in the same directory as the script.

### Step 2 — Run the Notebook / Script
Execute all cells in order:

1. **Parse** the Cranfield dataset → builds `docs` dictionary
2. **Preprocess** text (lowercase, tokenize, remove stopwords, stem)
3. **Build Inverted Index** → saves `inverted_index.pkl`
4. **Compute TF-IDF** → saves `idf.pkl` and `tfidf_matrix.pkl`
5. **Run Sample Queries** → prints Top-K retrieval results
6. **Evaluate** → computes MAP and NDCG at multiple k values
7. **RAG Pipeline** → generates LLM answers using OpenAI API

### Step 3 — Set OpenAI API Key (for RAG)

In Google Colab, add your key via **Secrets** (🔑 icon in sidebar):
- Key name: `OPENAI_API_KEY`

To run locally, set the environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
```

And replace this line in the code:
```python
# FROM (Colab)
from google.colab import userdata
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# TO (Local)
import os
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
```

---

##  Sample Queries

```python
sample_queries = [
    "aircraft engine efficiency",
    "wing aerodynamics",
    "boundary layer heat transfer",
    "compressible flow pressure"
]
```

---

##  Evaluation Results

### MAP (Mean Average Precision)

| k | MAP Score |
|---|-----------|
| 10 | 0.0024 |
| 50 | 0.0065 |
| 100 | 0.0083 |

### NDCG (Normalized Discounted Cumulative Gain)

| k | NDCG Score |
|---|------------|
| 10 | 0.0090 |
| 50 | 0.0228 |
| 100 | 0.0343 |
| 500 | 0.0916 |
| 1000 | 0.1564 |

> Scores are consistent with published baselines for TF-IDF on the Cranfield dataset. Low scores reflect vocabulary mismatch — a known limitation of exact-term retrieval.

---

##  RAG Pipeline

The RAG module:
1. Takes a user query
2. Retrieves Top-K documents using TF-IDF + Cosine Similarity
3. Passes document titles and abstracts as context to **GPT-4o-mini**
4. Returns a grounded natural language answer

The LLM is instructed to answer **only from retrieved documents** and acknowledge gaps rather than hallucinate.

---

##  Dataset

**Cranfield Collection** — a standard IR benchmark from the 1960s:

| File | Content |
|------|---------|
| `cran.all.1400` | 1400 aerodynamics abstracts |
| `cran.qry` | 225 test queries |
| `cranqrel` | Graded relevance judgments (1–4, -1) |

Relevance mapping for evaluation:
- `1, 2, 3, 4` → **Relevant** (MAP) / Gains `4, 3, 2, 1` (NDCG)
- `-1` → **Not relevant**

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3 |
| NLP | NLTK (tokenization, stopwords, Porter Stemmer) |
| Retrieval | Custom TF-IDF + Cosine Similarity |
| LLM | OpenAI GPT-4o-mini |
| Storage | Python `pickle` |
| Environment | Google Colab |

---

##  References

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

