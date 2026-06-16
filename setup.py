"""
setup.py - One-time initialisation: creates the SQLite DB and loads
           FAQ embeddings into ChromaDB.

Run once before launching the Streamlit app:
    python setup.py
"""

from __future__ import annotations

from core.logging_setup import configure_logging, get_logger
from core.settings import validate_settings
from data.db_setup import setup_database
from data.faq_setup import setup_vector_store
from data.generator import generate_sample_data

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    problems = validate_settings()
    if problems:
        for p in problems:
            logger.warning("Configuration warning: %s", p)
        print("Note: some configuration is missing (see warnings above).")
        print("Setup will still run, but the app will not work without GROQ_API_KEY.\n")

    print("=" * 60)
    print("Insurance Support Assistant - One-time Setup")
    print("=" * 60)

    print("\n[1/2] Generating synthetic data and creating SQLite database...")
    data = generate_sample_data()
    setup_database(data)
    print("Database created successfully.")

    print("\n[2/2] Setting up ChromaDB vector store (this may take a minute)...")
    setup_vector_store()
    print("Vector store ready.")

    print("\nSetup complete. Launch the app with:")
    print("    streamlit run ui/app.py")


if __name__ == "__main__":
    main()
