import numpy as np
import fitz
import re
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import spacy
from spacy.cli import download


download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

def semantic_chunking(
    data: list[dict],
    model,
    max_words: int = 120,
    percentile: int = 20,
    overlap_sentences: int = 1
)-> list[dict] :

    if not data:
        return []

    sentences = [
        item["content"]
        for item in data
    ]

    # Embedding all sentences at once
    sentence_embeddings = np.array(
        model.embed_documents(sentences)
    )

    # similarity with the previous sentence for calculating the threshold 
    adjacent_similarities = []

    for i in range(len(sentence_embeddings) - 1):

        score = cosine_similarity(
            [sentence_embeddings[i]],
            [sentence_embeddings[i + 1]]
        )[0][0]

        adjacent_similarities.append(score)

    # Adaptive threshold for the similarity scores
    threshold = np.percentile(
        adjacent_similarities,
        percentile
    )

    chunks = []

    current_chunk = {
        "content": [sentences[0]],
        "metadata": {
            "page_start": data[0]["metadata"]["page"],
            "page_end": data[0]["metadata"]["page"],
            "source": data[0]["metadata"]["source"]
        }
    }

    current_embeddings = [
        sentence_embeddings[0]
    ]

    current_word_count = len(
        sentences[0].split()
    )

    running_sum = np.array(
        sentence_embeddings[0]
    )

    for i in range(1, len(sentences)):

        next_sentence = sentences[i]
        next_embedding = sentence_embeddings[i]

        # running centroid for topic similarity
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

            chunks.append({
                "content": " ".join(
                    current_chunk["content"]
                ),
                "metadata": current_chunk["metadata"]
            })

            # overlap content
            overlap_content = (
                current_chunk["content"][-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

            overlap_embeddings = (
                current_embeddings[-overlap_sentences:]
                if overlap_sentences > 0
                else []
            )

            if overlap_content:

                overlap_start_idx = max(
                    0,
                    i - overlap_sentences
                )

                current_chunk = {
                    "content": overlap_content.copy(),
                    "metadata": {
                        "page_start": data[
                            overlap_start_idx
                        ]["metadata"]["page"],

                        "page_end": data[
                            i - 1
                        ]["metadata"]["page"],

                        "source": data[
                            i - 1
                        ]["metadata"]["source"]
                    }
                }

            else:

                current_chunk = {
                    "content": [],
                    "metadata": {
                        "page_start": data[i]["metadata"]["page"],
                        "page_end": data[i]["metadata"]["page"],
                        "source": data[i]["metadata"]["source"]
                    }
                }

            current_embeddings = overlap_embeddings.copy()

            current_word_count = sum(
                len(sentence.split())
                for sentence in current_chunk["content"]
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

        # add current sentence

        current_chunk["content"].append(
            next_sentence
        )

        current_chunk["metadata"]["page_end"] = (
            data[i]["metadata"]["page"]
        )

        current_embeddings.append(
            next_embedding
        )

        current_word_count += next_words

        running_sum += next_embedding


    if current_chunk["content"]:

        chunks.append({
            "content": " ".join(
                current_chunk["content"]
            ),
            "metadata": current_chunk["metadata"]
        })

    return chunks

def process_pdf(pdf_path: str):

    doc = fitz.open(pdf_path)

    corpus = []

    for page_num, page in enumerate(doc):

        # Extract text
        text = page.get_text()

        # Clean text
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Sentence segmentation
        sentences = [
            sent.text.strip()
            for sent in nlp(text).sents
        ]

        # Store sentences with metadata
        for sentence in sentences:

            corpus.append({
                "content": sentence,
                "metadata": {
                    "page": page_num + 1,
                    "source": pdf_path
                }
            })

    return corpus

def add_chunks_to_chroma(
    collection,
    chunks,
    model
):

    documents = [
        chunk["content"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    embeddings = model.embed_documents(
        documents
    )

    ids = [
        str(uuid.uuid4())        
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

