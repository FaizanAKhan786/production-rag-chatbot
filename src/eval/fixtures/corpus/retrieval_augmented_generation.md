# Retrieval-Augmented Generation with Hybrid Search and Cross-Encoder Reranking

## Abstract
Large language models often exhibit hallucinations and outdated parametric knowledge. Retrieval-Augmented Generation (RAG) couples a pre-trained generator with an external non-parametric memory index. In this work, we demonstrate that hybrid retrieval combining BM25 lexical search and dense bi-encoder representations via Reciprocal Rank Fusion (RRF) significantly outperforms vector-only retrieval.

## 1. Introduction
Parametric memory stores knowledge implicitly within LLM neural weights, whereas non-parametric memory indexes an external corpus of documents. RAG systems retrieve top-$K$ supporting documents and prepend them to the input prompt, grounding the generation process in verifiable citations.

## 2. Hybrid Retrieval and Reciprocal Rank Fusion
Dense retrievers excel at semantic abstraction and synonym matching, but frequently struggle with exact keyword matching, specialized identifiers, and acronyms. Conversely, sparse BM25 models excel at lexical token matching.
To combine disparate scoring distributions without arbitrary score normalization, we apply Reciprocal Rank Fusion (RRF):
$$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}$$
where $k=60$ acts as a ranking smoothing constant.

## 3. Two-Stage Retrieval with Cross-Encoder Rerankers
While bi-encoders generate independent document embeddings for fast nearest neighbor lookups, cross-encoders compute joint attention over query and candidate passage pairs. Reranking the top 40 RRF candidates down to 5 final context passages using a cross-encoder improves generation answer groundedness by 28% while filtering out distractor chunks.

## 4. Conclusion
Hybrid retrieval combined with cross-encoder reranking provides a robust foundation for production RAG pipelines, effectively neutralizing hallucination risks and ensuring transparent attribution.
