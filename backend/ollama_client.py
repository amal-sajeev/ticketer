import ollama
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def call_ollama(prompt: str, model: str = "gemma3n:e2b", json_mode: bool = False) -> Any:
    try:
        options = {}
        if json_mode:
            options["format"] = "json"
            
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options
        )
        return response['response']
    except Exception as e:
        print(f"Ollama error: {e}")
        return "" if not json_mode else {}

def get_embedding(text: str) -> bytes:
    """Generate and serialize embedding vector"""
    embedding = embedding_model.encode(text)
    return embedding.tobytes()

def tag_query(query: str) -> Dict[str, str]:
    prompt = f"""
    Analyze this telecom query and output JSON with:
    - intent: [complaint/question/inquiry/feedback]
    - topic: [billing/network/service/account/technical]
    - urgency: [high/medium/low]
    - requires_human: [true/false] (true for complaints, technical issues, or high urgency)
    
    Query: {query}
    """
    response = call_ollama(prompt, json_mode=True)
    
    # Default fallback values
    defaults = {
        "intent": "question",
        "topic": "general",
        "urgency": "medium",
        "requires_human": False
    }
    
    if isinstance(response, dict):
        return {**defaults, **response}
    try:
        return json.loads(response)
    except:
        return defaults

def cluster_tickets(tickets: List[Dict[str, Any]]) -> List[List[int]]:
    """Cluster tickets using embeddings and DBSCAN"""
    if len(tickets) < 2:
        return []
    
    # Get embeddings from DB or generate new
    embeddings = []
    for ticket in tickets:
        if 'embedding' in ticket and ticket['embedding']:
            emb = np.frombuffer(ticket['embedding'], dtype=np.float32)
        else:
            emb = embedding_model.encode(ticket['query'])
        embeddings.append(emb)
    
    # Cluster with DBSCAN
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(np.array(embeddings))
    
    # Group tickets by cluster
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:  # Ignore outliers
            clusters.setdefault(label, []).append(tickets[idx]["id"])
    
    return list(clusters.values())

def generate_auto_reply(query: str, tags: Dict[str, str]) -> Dict[str, Any]:
    prompt = f"""
    As telecom support agent, respond to this {tags['urgency']} urgency {tags['intent']} about {tags['topic']}:
    Query: {query}

    Return JSON:
    {{
        "reply": "...",
        "escalate": true or false
    }}

    Guidelines:
    - Resolve if possible.
    - If unsure, set escalate = true.
    """
    response = call_ollama(prompt, json_mode=True)
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except:
        return {"reply": "I'll escalate this to our support team", "escalate": True}

def summarize_cluster(queries: List[str]) -> str:
    prompt = f"""
    Summarize these {len(queries)} telecom customer queries into 1-2 sentences:
    Identify the common theme and root cause.
    
    {"".join([f"- {q}\n" for q in queries])}
    """
    return call_ollama(prompt)

def find_similar_solutions(query: str, embeddings: List[bytes]) -> List[int]:
    """Find similar resolved tickets using cosine similarity"""
    if not embeddings:
        return []
    
    # Convert stored embeddings to vectors
    stored_embeddings = [np.frombuffer(emb, dtype=np.float32) for emb in embeddings]
    
    # Calculate similarity
    query_embedding = embedding_model.encode(query)
    similarities = []
    for idx, emb in enumerate(stored_embeddings):
        cos_sim = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        similarities.append((idx, cos_sim))
    
    # Return top 3 most similar
    return [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:3]]