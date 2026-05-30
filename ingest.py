import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model = "BAAI/bge-small-en-v1.5")


def semantic_chunking(
    sentences: list[str],
    model,
    max_words: int = 120,
    percentile: int = 20,
    overlap_sentences: int = 1
) -> list[list[str]]:
    """
    Semantic chunking using:
    - Precomputed embeddings
    - Adaptive percentile threshold
    - Running word count
    - Running centroid
    - Optional sentence overlap
    """

    if not sentences:
        return []

    # Generate embeddings once
    sentence_embeddings = np.array(
        model.embed_documents(sentences)
    )

    # Compute adjacent similarities
    adjacent_similarities = []

    for i in range(len(sentence_embeddings) - 1):
        score = cosine_similarity(
            [sentence_embeddings[i]],
            [sentence_embeddings[i + 1]]
        )[0][0]

        adjacent_similarities.append(score)

    # Adaptive threshold
    threshold = np.percentile(
        adjacent_similarities,
        percentile
    )

    chunks = []

    current_chunk = [sentences[0]]
    current_embeddings = [sentence_embeddings[0]]

    current_word_count = len(
        sentences[0].split()
    )

    running_sum = np.array(
        sentence_embeddings[0]
    )

    for i in range(1, len(sentences)):

        next_sentence = sentences[i]
        next_embedding = sentence_embeddings[i]

        centroid = (
            running_sum /
            len(current_embeddings)
        )

        similarity = cosine_similarity(
            [centroid],
            [next_embedding]
        )[0][0]

        next_words = len(
            next_sentence.split()
        )

        should_split = (
            similarity < threshold
            or current_word_count + next_words > max_words
        )

        if should_split:

            chunks.append(current_chunk)

            overlap_chunk = (
                current_chunk[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

            overlap_embeddings = (
                current_embeddings[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

            current_chunk = overlap_chunk.copy()
            current_embeddings = overlap_embeddings.copy()

            current_word_count = sum(
                len(s.split())
                for s in current_chunk
            )

            if current_embeddings:
                running_sum = np.sum(
                    current_embeddings,
                    axis=0
                )
            else:
                running_sum = np.zeros_like(
                    next_embedding
                )

        current_chunk.append(next_sentence)
        current_embeddings.append(next_embedding)

        current_word_count += next_words
        running_sum += next_embedding

    if current_chunk:
        chunks.append(current_chunk)

    return chunks