# AJARNNAIDEE — Chula Engineering Expert Finder

A Streamlit app that helps students find Chulalongkorn University Faculty of Engineering professors whose research may relate to a topic. It supports Thai and English queries, semantic search, multilingual cross-encoder reranking, and an optional AI Research Advisor.

> This is an independent academic project and is not an officially endorsed university product. Faculty information is based on publicly available sources. Please verify details on the original profile pages.

## Features

- View dataset coverage, including faculty profiles, departments, research information, and email availability.
- Search with a research topic in Thai or English.
- Filter by department and choose between 1 and 20 results.
- Retrieve candidates with `Qwen/Qwen3-Embedding-0.6B` and rerank them with `Qwen/Qwen3-Reranker-0.6B`.
- See matching evidence from each profile's listed research interests and description.
- Optionally use the Groq-powered AI Research Advisor to explain and compare matches and suggest questions to ask. The advisor does not search for or add professors.

Search scores are ranking signals, not calibrated measures of expertise. They do not indicate a professor's availability or willingness to collaborate.

## Requirements

- Python 3.10 or later
- Internet access to download Hugging Face models on first use
- PyTorch automatically selects CUDA, Apple MPS, or CPU when available. Search may take longer on CPU.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the virtual environment with PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Open the URL printed by Streamlit in your terminal (usually `http://localhost:8501`).

The first search may take a while because the embedding and reranker models need to be downloaded and loaded. Streamlit caches the models and generated embeddings for subsequent use.

## Optional: enable the AI Research Advisor

Search and reranking work without Groq. The advisor requires a Groq API key.

**Option 1: environment variable**

```bash
export GROQ_API_KEY="your-groq-api-key"
streamlit run app.py
```

In PowerShell:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
streamlit run app.py
```

**Option 2: Streamlit secrets**

Create `.streamlit/secrets.toml` locally and add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit or publish your API key. To select a different Groq model, set the `GROQ_MODEL` environment variable before starting the app. If unset, the app uses `qwen/qwen3.8-27b`.

When the advisor is enabled, your topic and the profiles of the top five matched professors are sent to Groq to generate its response. Consider this before using the feature.

## Dataset

The app reads `data/professors.json`, which should contain a JSON array of professor records. The array may also be wrapped in an object under the `professors` or `faculty` key. Records can include fields such as:

- `professor_id`, `name_th`, `name_en`, `academic_title`
- `department_id`, `email`
- `research_interests` (a list or text delimited by newlines, semicolons, or pipes)
- `research_description`, `profile_url`, `source_url`

Non-object records are skipped. Duplicate records are removed using `professor_id`, or a name when no ID is available. CSV files are also supported if `DATA_PATH` in `app.py` is changed to point to a `.csv` file with matching column names.

## Project structure

```text
.
├── app.py                  # Streamlit app and search/advisor pipeline
├── data/
│   └── professors.json     # Faculty dataset
└── requirements.txt        # Python dependencies
```

## Limitations and responsible use

- Results depend on the completeness and freshness of the research information in the dataset.
- If the reranker cannot be loaded, the app displays semantic-search results instead.
- The AI Advisor only uses profiles retrieved by the search system. It does not discover additional professors, and its advice may be inaccurate.
- Verify information and contact requirements on the original profile pages before making decisions.
- The reranker model card specifies a research/evaluation-only license. Review the model card and license terms before using it for other purposes.
