from scipy.special import expit
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank_retrieval(chunks: list[list[str]], k :int) -> list[str]:
    rerank_scores = reranker.predict(chunks)

    reranked = sorted(zip(chunks,rerank_scores), key = lambda x:x[1],reverse=True)

    top_k = reranked[: k]
    final_chunks = [x[0][1] for x in top_k]  

    return final_chunks
