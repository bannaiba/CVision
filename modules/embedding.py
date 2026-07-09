"""
modules/embedding.py
====================
Semantic resume ranking using ONNX-based embeddings via fastembed.

Why fastembed instead of sentence-transformers?
------------------------------------------------
sentence-transformers requires PyTorch, which does not yet have official wheels
for Python 3.14+ (as of mid-2025). fastembed by Qdrant achieves the same result
by running the same models (e.g., all-MiniLM-L6-v2) via ONNX Runtime:

  - No PyTorch required — only onnxruntime (pure C++, pre-built for all Python)
  - Downloads ~85 MB ONNX model on first use; cached in ~/.cache/fastembed
  - Identical semantic quality to the torch-based version
  - 30–100 ms inference per document on CPU

Fallback chain
--------------
If fastembed is not installed or fails, this module gracefully falls back to
TF-IDF + Cosine Similarity (scikit-learn). TF-IDF is less semantically aware
but always works with zero extra dependencies and is a solid baseline.

The app.py UI indicates to the user which backend is active.

Future upgrade path
-------------------
When PyTorch ships Python 3.14 wheels, simply:
    pip install sentence-transformers
Then swap the backend in load_embedding_model() — the rest of the module
remains unchanged because fastembed and sentence-transformers have identical
.encode() semantics and output shapes.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
"""Model identifier used for both fastembed and sentence-transformers backends."""

# ── Curated Technical Skills Vocabulary ───────────────────────────────────────
# Exact-phrase matching against resume Markdown text (word-boundary regex).
TECH_SKILLS_VOCAB: list[str] = [
    # ── Programming Languages ─────────────────────────────────────────────────
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "kotlin", "r", "matlab", "sql", "bash",
    # ── Machine Learning & AI ─────────────────────────────────────────────────
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "llm", "transformer",
    "bert", "gpt", "neural network",
    # ── ML Frameworks & Libraries ─────────────────────────────────────────────
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "hugging face", "langchain", "pandas", "numpy", "scipy", "matplotlib",
    # ── Data Engineering ──────────────────────────────────────────────────────
    "apache spark", "kafka", "airflow", "dbt", "etl", "hadoop",
    "databricks", "snowflake", "bigquery", "redshift", "data pipeline",
    # ── Cloud & DevOps ────────────────────────────────────────────────────────
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ci/cd", "github actions", "mlops", "linux", "git",
    # ── Backend & APIs ────────────────────────────────────────────────────────
    "fastapi", "flask", "django", "rest api", "graphql", "microservices",
    "postgresql", "mongodb", "redis", "elasticsearch",
    # ── Analytics & BI ────────────────────────────────────────────────────────
    "tableau", "power bi", "data visualization", "excel",
    # ── Soft Skills ───────────────────────────────────────────────────────────
    "agile", "scrum", "project management", "leadership", "communication",
]


# ── Backend Detection ──────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """
    Detect the best available embedding backend on this system.

    Priority order:
    1. fastembed (ONNX, no PyTorch, highly stable on Windows)    ← preferred
    2. sentence-transformers (PyTorch, if it happens to work)    ← fallback
    3. tfidf (scikit-learn, always works)                        ← last resort

    Returns:
        One of: ``"fastembed"``, ``"sentence-transformers"``, ``"tfidf"``
    """
    try:
        import fastembed  # noqa: F401
        return "fastembed"
    except ImportError as e:
        print(f"\n\n🚨 FATAL FASTEMBED IMPORT ERROR: {e}\n\n")
        logger.warning(f"fastembed import failed: {e}")
        pass

    return "tfidf"


ACTIVE_BACKEND: str = _detect_backend()
"""The embedding backend that will be used by this module."""


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Load and return an embedding model using the best available backend.

    The function is designed to be called once per process and wrapped with
    ``@st.cache_resource`` in app.py to avoid repeated loading.

    Backend priority:
    1. **fastembed** — loads an ONNX model from HuggingFace/Qdrant CDN.
       ``model_name`` must be a HuggingFace-style identifier.
    2. **sentence-transformers** — loads a PyTorch model.
    3. **tfidf** — returns ``None``; ``embed_texts()`` handles the None case.

    Args:
        model_name: HuggingFace model identifier.
                    Default: ``"sentence-transformers/all-MiniLM-L6-v2"``

    Returns:
        A model object (fastembed.TextEmbedding or SentenceTransformer),
        or ``None`` if using TF-IDF fallback.

    Raises:
        ImportError: Only if a specific backend is explicitly requested but
                     not installed (normal usage auto-selects).
    """
    backend = ACTIVE_BACKEND

    # Fix UI short-names for fastembed which requires full HuggingFace identifiers
    if model_name == "all-MiniLM-L6-v2":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if backend == "fastembed":
        try:
            from fastembed import TextEmbedding
            logger.info(
                "Loading fastembed model: %s (ONNX, no PyTorch required)", model_name
            )
            # fastembed downloads ONNX weights on first use (~85MB)
            model = TextEmbedding(model_name=model_name)
            logger.info("fastembed model ready.")
            return model
        except Exception as exc:
            print(f"\n\n🚨 FASTEMBED CRASHED DURING LOAD: {exc}\n\n")
            logger.warning("fastembed load failed: %s — falling back to TF-IDF.", exc)
            return None

    elif backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: %s", model_name)
            model = SentenceTransformer(model_name)
            logger.info("sentence-transformers model ready.")
            return model
        except Exception as exc:
            logger.warning("sentence-transformers load failed: %s — falling back to TF-IDF.", exc)
            return None

    else:
        logger.warning(
            "No BERT/ONNX embedding backend available. "
            "Using TF-IDF fallback. "
            "Install fastembed for better results: pip install fastembed"
        )
        return None


# ── Text Embedding ─────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model) -> np.ndarray:
    """
    Encode a list of strings into L2-normalized embedding vectors.

    Dispatches to the appropriate backend based on the type of ``model``.
    If ``model`` is None (TF-IDF mode), returns an empty array — callers
    must use the TF-IDF ranking path in that case.

    L2 normalization means cosine similarity = dot product, which is faster
    and produces scores cleanly in the range [−1, 1].

    Args:
        texts: Strings to encode (plain text or Markdown).
        model: Model object from ``load_embedding_model()``, or ``None``.

    Returns:
        2D numpy array of shape ``(len(texts), embedding_dim)`` with L2-normalized
        rows. Returns empty array if model is None.
    """
    if model is None or not texts:
        return np.array([])

    logger.info("Encoding %d text(s)...", len(texts))

    # ── fastembed backend ──────────────────────────────────────────────────────
    # fastembed.embed() returns a generator of numpy arrays (one per text)
    try:
        from fastembed import TextEmbedding
        if isinstance(model, TextEmbedding):
            embeddings = np.array(list(model.embed(texts)))
            # fastembed already L2-normalizes by default, but be explicit:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)   # Avoid divide-by-zero
            embeddings = embeddings / norms
            logger.info("fastembed encoding complete — shape: %s", embeddings.shape)
            return embeddings
    except ImportError:
        pass

    # ── sentence-transformers backend ──────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        if isinstance(model, SentenceTransformer):
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
                convert_to_numpy=True,
            )
            logger.info("sentence-transformers encoding complete — shape: %s", embeddings.shape)
            return embeddings
    except ImportError:
        pass

    logger.warning("Could not identify model backend — returning empty array.")
    return np.array([])


def compute_similarity_scores(
    jd_embedding: np.ndarray,
    resume_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between the JD vector and all resume vectors.

    With L2-normalized vectors: cosine_sim(a, b) = a · b (dot product).

    Args:
        jd_embedding: 1D array ``(embedding_dim,)`` or 2D ``(1, dim)``.
        resume_embeddings: 2D array ``(n_resumes, embedding_dim)``.

    Returns:
        1D float array ``(n_resumes,)`` with scores in ``[0, 100]``.
    """
    jd_vec = jd_embedding.flatten()
    raw_scores = resume_embeddings @ jd_vec
    clipped = np.clip(raw_scores, 0.0, 1.0)
    return clipped * 100.0


# ── TF-IDF Fallback Ranking ────────────────────────────────────────────────────

def _rank_with_tfidf(
    job_description: str,
    filenames: list[str],
    resume_markdowns: list[str],
    candidate_metadata: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """
    Rank resumes using TF-IDF + Cosine Similarity (scikit-learn).

    This is the fallback ranking method used when no BERT/ONNX backend is
    available. It is less semantically aware but works without any ML frameworks
    and produces reasonable keyword-overlap-based scores.

    Args:
        job_description: Full JD text.
        filenames: PDF filenames (parallel to resume_markdowns).
        resume_markdowns: Extracted Markdown text per resume.
        candidate_metadata: Optional list of metadata dicts.

    Returns:
        Ranked DataFrame (same schema as ``rank_resumes_semantic()`` output).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("Using TF-IDF fallback ranking for %d resumes.", len(filenames))

    all_docs = [job_description] + resume_markdowns
    
    # Loosen constraints for very small document batches to avoid ValueError
    min_df = 1 if len(all_docs) <= 3 else 2
    max_df = 1.0 if len(all_docs) <= 3 else 0.95

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vec, resume_vecs).flatten() * 100.0

    skills_per_resume = [
        extract_skills_from_markdown(md)[:10] for md in resume_markdowns
    ]

    data: dict = {
        "Filename":      filenames,
        "Fit Score (%)": scores,
        "Top Skills":    [", ".join(s) if s else "—" for s in skills_per_resume],
        "Ranking Mode":  ["TF-IDF (fallback)"] * len(filenames),
    }

    if candidate_metadata and len(candidate_metadata) == len(filenames):
        data["Candidate Name"]      = [m.get("name", "")        for m in candidate_metadata]
        data["Email"]               = [m.get("email", "")       for m in candidate_metadata]
        data["CGPA"]                = [
            f"{m['cgpa']:.2f}" if m.get("cgpa", -1) != -1 else "N/A"
            for m in candidate_metadata
        ]
        data["Years of Experience"] = [
            f"{m['years_exp']:.1f}" if m.get("years_exp", -1) != -1 else "N/A"
            for m in candidate_metadata
        ]
        data["Degree"]              = [
            m.get("degree", "").title() or "N/A" for m in candidate_metadata
        ]

    df = (
        pd.DataFrame(data)
        .sort_values("Fit Score (%)", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# ── Skills Extraction ──────────────────────────────────────────────────────────

def extract_skills_from_markdown(
    markdown_text: str,
    vocab: list[str] = TECH_SKILLS_VOCAB,
) -> list[str]:
    """
    Identify technical skills present in a resume's Markdown text.

    Uses whole-word exact-phrase matching (word boundaries via regex) to avoid
    false positives (e.g., ``r`` matching inside ``researcher``).

    Args:
        markdown_text: Markdown string from the parser module.
        vocab: Ordered list of skill phrases to search for.

    Returns:
        Deduplicated list of matched skill strings in vocabulary order.
    """
    text_lower = markdown_text.lower()
    matched: list[str] = []

    for skill in vocab:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(skill)

    return matched


# ── Core Ranking Function ──────────────────────────────────────────────────────

def rank_resumes_semantic(
    job_description: str,
    filenames: list[str],
    resume_markdowns: list[str],
    model,
    candidate_metadata: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """
    Rank resumes against a job description. Uses the best available backend.

    Tries ONNX/BERT semantic ranking first (fastembed or sentence-transformers).
    Falls back to TF-IDF if no vector model is available.

    Args:
        job_description: Full job description text.
        filenames: PDF filenames (parallel to resume_markdowns).
        resume_markdowns: Extracted Markdown text per resume.
        model: Model from ``load_embedding_model()`` or None (triggers TF-IDF).
        candidate_metadata: Optional list of dicts with candidate metadata fields.

    Returns:
        DataFrame sorted by ``Fit Score (%)`` descending with columns:
        ``Rank``, ``Filename``, ``Fit Score (%)``, ``Top Skills``
        and optionally: ``Candidate Name``, ``Email``, ``CGPA``,
        ``Years of Experience``, ``Degree``, ``Ranking Mode``.
    """
    if not filenames or not resume_markdowns:
        logger.error("rank_resumes_semantic: empty inputs.")
        return pd.DataFrame()

    # ── BERT/ONNX path ─────────────────────────────────────────────────────────
    if model is not None:
        all_texts = [job_description] + resume_markdowns
        all_embeddings = embed_texts(all_texts, model)

        if all_embeddings.size > 0:
            jd_embedding      = all_embeddings[0]
            resume_embeddings = all_embeddings[1:]
            scores = compute_similarity_scores(jd_embedding, resume_embeddings)

            skills_per_resume = [
                extract_skills_from_markdown(md)[:10] for md in resume_markdowns
            ]

            data: dict = {
                "Filename":      filenames,
                "Fit Score (%)": scores,
                "Top Skills":    [", ".join(s) if s else "—" for s in skills_per_resume],
                "Ranking Mode":  [f"BERT ({ACTIVE_BACKEND})"] * len(filenames),
            }

            if candidate_metadata and len(candidate_metadata) == len(filenames):
                data["Candidate Name"] = [m.get("name", "")  for m in candidate_metadata]
                data["Email"]          = [m.get("email", "") for m in candidate_metadata]
                data["CGPA"]           = [
                    f"{m['cgpa']:.2f}" if m.get("cgpa", -1) != -1 else "N/A"
                    for m in candidate_metadata
                ]
                data["Degree"]         = [
                    m.get("degree", "").title() or "N/A" for m in candidate_metadata
                ]

            df = (
                pd.DataFrame(data)
                .sort_values("Fit Score (%)", ascending=False)
                .reset_index(drop=True)
            )
            df.insert(0, "Rank", range(1, len(df) + 1))
            logger.info(
                "BERT ranking complete — %d candidates, top: %.1f%%",
                len(df), df["Fit Score (%)"].iloc[0],
            )
            return df

    # ── TF-IDF fallback path ───────────────────────────────────────────────────
    return _rank_with_tfidf(job_description, filenames, resume_markdowns, candidate_metadata)


# ── Summary Statistics ─────────────────────────────────────────────────────────

def compute_summary_stats(results_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from the ranked results DataFrame.

    Args:
        results_df: Ranked output from ``rank_resumes_semantic()``.

    Returns:
        Dict with keys: ``top_score``, ``avg_score``, ``lowest_score``,
        ``n_candidates``, ``score_std``, ``above_50``, ``above_70``.
    """
    if results_df.empty:
        return {k: 0 for k in [
            "top_score", "avg_score", "lowest_score", "n_candidates",
            "score_std", "above_50", "above_70",
        ]}

    scores = results_df["Fit Score (%)"]
    return {
        "top_score":    float(scores.iloc[0]),
        "avg_score":    float(scores.mean()),
        "lowest_score": float(scores.iloc[-1]),
        "n_candidates": int(len(scores)),
        "score_std":    float(scores.std()),
        "above_50":     int((scores >= 50).sum()),
        "above_70":     int((scores >= 70).sum()),
    }
