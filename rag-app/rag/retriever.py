# rag/retriever.py

import time

from rag.embedder import embed_query
from rag.vectorstore import get_collection
from utils import config
from utils.logger import get_logger


logger = get_logger("retriever")


def retrieve(query, k=None):
    k = k or config.get_top_k() * 2
    query_embedding = embed_query(query)

    start = time.time()
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    elapsed = (time.time() - start) * 1000

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    payload = [
        {"text": doc, "metadata": metadata, "score": score}
        for doc, metadata, score in zip(docs, metadatas, scores)
    ]

    preview = [f"score={s:.4f} src={m.get('source','?')}" for s, m in zip(scores, metadatas)]
    logger.info("retrieved %d docs in %.1f ms | %s", len(payload), elapsed, preview[:5])

    return payload