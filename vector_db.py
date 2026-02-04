from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
from dotenv import load_dotenv



class QdrantStorage:
    def __init__(self, collection="docs", dim=768):
        # 1. Load URL and Key from Environment Variables
        # If running on Render, it uses the "Secret" variables.
        # If running locally, it defaults back to your localhost.
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY", None)

        # 2. Initialize the client (api_key is required for Cloud, ignored for local)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
        
        self.collection = collection
        
        # 3. Create collection if missing
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k
        )
        
        # Access the list of points from the result
        results = result.points

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
