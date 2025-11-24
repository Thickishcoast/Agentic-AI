from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "web_agent_experiences"  # same name you used when indexing
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # 1. Connect to Qdrant and load the same embedding model
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(MODEL_NAME)

    # 2. Build a query embedding
    query_text = "Example Domain homepage with a Learn more link"
    query_vec = model.encode(query_text).tolist()

    # 3. Query Qdrant (new API)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=3,
        with_payload=True,
        with_vectors=False,  # set True if you also want the stored vectors
    )

    # 4. Inspect the results
    print(f"Top results for query: {query_text!r}")
    for pt in res.points:
        print("-" * 60)
        print("ID:", pt.id)
        print("Score:", pt.score)

        payload = pt.payload or {}
        print("Episode:", payload.get("episode"))
        print("Step:", payload.get("step"))
        print("URL:", payload.get("url"))
        print("Reward:", payload.get("reward"))

        obs_snippet = (payload.get("obs_text") or "")[:200]
        print("Obs snippet:", obs_snippet.replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
