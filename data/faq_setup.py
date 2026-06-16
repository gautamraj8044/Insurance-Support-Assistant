"""
data/faq_setup.py - Loads the FAQ dataset from HuggingFace into ChromaDB.
"""

from __future__ import annotations

import chromadb
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from core.logging_setup import get_logger
from core.settings import faq, paths

logger = get_logger(__name__)


def setup_vector_store() -> None:
    """Load the FAQ dataset and upsert it into a persistent ChromaDB collection."""
    logger.info("Loading FAQ dataset '%s' from HuggingFace...", faq.dataset_name)
    ds = load_dataset(faq.dataset_name)

    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    df["combined"] = "Question: " + df["input"] + " \n Answer: " + df["output"]
    logger.info("Dataset shape: %s", df.shape)

    sample_size = min(faq.sample_size, len(df))
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    client = chromadb.PersistentClient(path=paths.chroma_path)
    # Start clean so re-running setup doesn't duplicate or stale-out entries.
    try:
        client.delete_collection(name=paths.chroma_collection)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=paths.chroma_collection)

    logger.info("Upserting %d FAQ records into ChromaDB...", len(df))
    for i in tqdm(range(0, len(df), faq.batch_size), desc="Loading FAQs"):
        batch = df.iloc[i : i + faq.batch_size]
        collection.add(
            documents=batch["combined"].tolist(),
            metadatas=[
                {"question": q, "answer": a}
                for q, a in zip(batch["input"], batch["output"])
            ],
            ids=batch.index.astype(str).tolist(),
        )

    logger.info("ChromaDB vector store ready with %d documents.", collection.count())
