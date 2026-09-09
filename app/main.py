from fastapi import FastAPI
import ollama

app = FastAPI(title="RAG Document Q&A API")

@app.get("/health")
def health_check():
    """Vérifie que l'API tourne et qu'Ollama est joignable."""
    try:
        models = ollama.list()
        return {
            "status": "ok",
            "ollama_connected": True,
            "available_models": [m["model"] for m in models.get("models", [])]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "ollama_connected": False,
            "error": str(e)
        }

@app.get("/")
def root():
    return {"message": "Bienvenue sur le RAG Document Q&A API"}