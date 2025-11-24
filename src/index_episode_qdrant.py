# src/index_episode_qdrant.py

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "web_agent_experiences"


def load_trajectory(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_text(step: dict) -> str:
    """
    Build the text string we will embed for this step.
    This combines URL, reward, observation text, and action.
    """
    obs_text = step.get("obs_text", "")
    action_str = step.get("action_str", "")
    reward = step.get("reward", 0.0)
    url = step.get("info", {}).get("url", "")

    return (
        f"URL: {url}\n"
        f"Reward: {reward}\n"
        f"Observation:\n{obs_text}\n\n"
        f"Action:\n{action_str}"
    )


def main():
    # 1. Load your logged episode
    episode_path = Path("logged_episodes") / "episode_0.json"
    steps = load_trajectory(episode_path)
    print(f"Loaded {len(steps)} steps from {episode_path}")

    # 2. Load sentence-transformers model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    # 3. Connect to local Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # 4. Create collection if needed
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")

    # 5. Build vectors and payloads
    texts = [build_text(step) for step in steps]
    vectors = model.encode(texts, convert_to_numpy=True).tolist()

    ids = []
    payloads = []
    for i, (step, vec) in enumerate(zip(steps, vectors)):
        ids.append(i)  # simple numeric IDs
        payloads.append(
            {
                "episode": 0,
                "step": step.get("step"),
                "obs_text": step.get("obs_text", ""),
                "action_str": step.get("action_str", ""),
                "reward": step.get("reward", 0.0),
                "url": step.get("info", {}).get("url", ""),
            }
        )

    # 6. Upsert into Qdrant
    points = [
        {"id": pid, "vector": vec, "payload": pl}
        for pid, vec, pl in zip(ids, vectors, payloads)
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Indexed {len(ids)} points into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
