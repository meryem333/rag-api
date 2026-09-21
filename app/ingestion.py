import os
import io
from pypdf import PdfReader
import ollama
import chromadb

# Adresse d'Ollama : configurable via variable d'environnement.
# En local (hors Docker), ça reste http://localhost:11434 par défaut.
# Dans le conteneur, on passera OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Client ChromaDB persistant
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrait tout le texte brut d'un fichier PDF (en mémoire)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Découpe le texte en morceaux avec un léger chevauchement."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    """Transforme un morceau de texte en vecteur numérique via Ollama."""
    response = ollama_client.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def store_chunks(chunks: list[str], source_name: str):
    """Génère un embedding pour chaque chunk et le stocke dans ChromaDB."""
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        chunk_id = f"{source_name}_chunk_{i}"

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": source_name, "chunk_index": i}]
        )
    print(f"{len(chunks)} chunks stockés pour '{source_name}'")


def answer_question(question: str, n_results: int = 3) -> dict:
    """Cherche les chunks pertinents et demande au LLM de répondre."""
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    relevant_chunks = results["documents"][0]

    context = "\n\n".join(relevant_chunks)
    prompt = f"""Voici des extraits de document :

{context}

En te basant uniquement sur ces extraits, réponds à la question suivante.
Si la réponse ne se trouve pas dans les extraits, dis-le clairement.

Question : {question}
Réponse :"""

    response = ollama_client.generate(model="llama3.2:3b", prompt=prompt)

    return {
        "question": question,
        "answer": response["response"],
        "sources_used": relevant_chunks
    }