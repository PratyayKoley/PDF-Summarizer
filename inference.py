from models.pdf_to_text import pdf_to_text_pymupdf as pdf_to_text
from models.keyphrase_extraction import extract_keyphrases
from models.chunking import smart_chunk_by_sentences
from models.embeddings import get_sentence_embedding, get_chunk_embeddings_batch
from models.sentence_scoring import compute_comprehensive_scores
from models.mmr_selection import mmr_select_sentences
import numpy as np

def process_pdf_and_summarize(pdf_path, summary_sentences=5):
    """
    Complete pipeline for PDF summarization with improved coherence
    """
    
    print(f"[1/6] Extracting text from PDF...")
    text = pdf_to_text(pdf_path)
    
    # Limit very long documents
    if len(text) > 100000:
        print(f"   Document too long ({len(text)} chars), using first 100k characters")
        text = text[:100000]
    
    print(f"   Extracted {len(text)} characters")
    
    print(f"[2/6] Extracting keyphrases...")
    keyphrases = extract_keyphrases(text, top_n=25)
    print(f"   Found {len(keyphrases)} keyphrases")
    
    print(f"[3/6] Chunking document by sentences...")
    chunks, sentences = smart_chunk_by_sentences(text, max_tokens=384, overlap_sentences=2)
    print(f"   Created {len(chunks)} chunks from {len(sentences)} sentences")
    
    print(f"[4/6] Computing embeddings...")
    chunk_embeddings = get_chunk_embeddings_batch(chunks, keyphrases)
    
    # Get sentence embeddings for MMR
    sentence_embeddings = []
    for sent in sentences:
        emb = get_sentence_embedding(sent)
        sentence_embeddings.append(emb)
    sentence_embeddings = np.array(sentence_embeddings)
    print(f"   Computed embeddings for {len(chunks)} chunks and {len(sentences)} sentences")
    
    print(f"[5/6] Scoring sentences...")
    scores = compute_comprehensive_scores(
        sentences, 
        chunk_embeddings, 
        chunks, 
        keyphrases, 
        text
    )
    print(f"   Top 5 scores: {sorted(scores, reverse=True)[:5]}")
    
    print(f"[6/6] Selecting diverse sentences using MMR...")
    summary_sentences_list = mmr_select_sentences(
        sentences,
        scores,
        sentence_embeddings,
        top_k=summary_sentences,
        lambda_param=0.7
    )
    
    summary = " ".join(summary_sentences_list)
    
    print(f"✅ Summary generation complete!")
    print(f"   Summary length: {len(summary)} characters, {len(summary_sentences_list)} sentences")
    
    return {
        "summary": summary,
        "keyphrases": keyphrases[:15],
        "num_sentences": len(sentences),
        "num_chunks": len(chunks),
        "top_sentence_scores": sorted(scores, reverse=True)[:10]
    }