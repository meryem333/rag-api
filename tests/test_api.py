from fastapi.testclient import TestClient
from app.main import app
from app import ingestion

client = TestClient(app)


def test_root():
    """Vérifie que la route racine répond correctement."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check_ollama_ok():
    """/health confirme la connexion à Ollama et liste les modèles."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama_connected"] is True
    assert "llama3.2:3b" in data["available_models"]


def test_health_check_ollama_down(mock_ollama_and_chroma):
    """/health passe en 'degraded' si Ollama est injoignable."""
    mock_ollama_and_chroma.list.side_effect = ConnectionError("Ollama injoignable")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama_connected"] is False
    assert "Ollama injoignable" in data["error"]


def test_ask_without_documents_indexed():
    """Sans document indexé, /ask répond avec une structure correcte et aucune source."""
    response = client.post("/ask", json={"question": "Quelles langues parle Meryem ?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["sources_used"] == []


def test_ask_with_indexed_document(mock_ollama_and_chroma):
    """Un chunk indexé est retrouvé et transmis au LLM (simulé)."""
    chunk = "Meryem parle français, arabe et anglais."
    ingestion.store_chunks([chunk], "cv_test")

    response = client.post("/ask", json={"question": "Quelles langues parle Meryem ?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Réponse simulée par le mock."
    assert chunk in data["sources_used"]
    mock_ollama_and_chroma.generate.assert_called_once()


def test_ask_missing_question_field():
    """Vérifie que l'API rejette une requête mal formée (validation Pydantic)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # erreur de validation attendue
    