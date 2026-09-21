import pytest
import chromadb
from unittest.mock import MagicMock

from app import ingestion, main

# nomic-embed-text produit des vecteurs de dimension 768
FAKE_EMBEDDING = [0.1] * 768


@pytest.fixture(autouse=True)
def mock_ollama_and_chroma(monkeypatch):
    """Remplace Ollama et ChromaDB par des doubles de test pour tous les tests."""
    fake_client = MagicMock()
    fake_client.embeddings.return_value = {"embedding": FAKE_EMBEDDING}
    fake_client.generate.return_value = {"response": "Réponse simulée par le mock."}
    fake_client.list.return_value = {"models": [{"model": "llama3.2:3b"}]}

    # main.py a importé ollama_client par son nom : il faut patcher les deux modules
    monkeypatch.setattr(ingestion, "ollama_client", fake_client)
    monkeypatch.setattr(main, "ollama_client", fake_client)

    chroma = chromadb.EphemeralClient()
    test_collection = chroma.get_or_create_collection(name="test_documents")
    monkeypatch.setattr(ingestion, "collection", test_collection)

    yield fake_client

    chroma.delete_collection("test_documents")