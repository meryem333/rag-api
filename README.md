# RAG Document Q&A API

[![CI](https://github.com/meryem333/rag-api/actions/workflows/ci.yml/badge.svg)](https://github.com/meryem333/rag-api/actions/workflows/ci.yml)

Assistant IA de questions-réponses sur documents (RAG), 100% local — API FastAPI + Ollama + ChromaDB, conteneurisée avec Docker et déployée sur Kubernetes (Minikube).
## Tests de charge

Tests réalisés avec Locust (5 utilisateurs simultanés, endpoint /ask) :

| Métrique | Valeur |
|---|---|
| Requêtes testées | 25 |
| Taux d'échec | 0% |
| Temps de réponse médian | ~30s |
| 95e percentile | ~82s |

**Limitation identifiée** : le LLM (llama3.2:3b) tourne en inférence CPU locale via Ollama, 
sans GPU. Sous charge concurrente, les requêtes s'empilent car Ollama traite les appels 
de façon quasi séquentielle sur une machine avec ~8 Go de RAM. En production, cette 
limite serait levée par :
- Un GPU dédié pour l'inférence
- Plusieurs instances Ollama derrière un load balancer
- Une file d'attente asynchrone (Celery/RQ) avec traitement en arrière-plan