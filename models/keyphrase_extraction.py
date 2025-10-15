from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import re

sbert = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sbert)

def extract_keyphrases(doc_text, top_n=25, use_mmr=True):
    """Extract keyphrases with filtering for quality"""
    # Extract candidates
    candidates = kw_model.extract_keywords(
        doc_text,
        keyphrase_ngram_range=(1, 4),
        stop_words="english",
        top_n=top_n * 2,  # Get more to filter
        use_mmr=use_mmr,
        diversity=0.6,
        nr_candidates=50
    )
    
    # Filter low-quality keyphrases
    filtered = []
    for phrase, score in candidates:
        phrase = phrase.strip()
        
        # Skip if too short or too long
        if len(phrase) < 3 or len(phrase) > 50:
            continue
        
        # Skip if mostly numbers
        if sum(c.isdigit() for c in phrase) > len(phrase) * 0.5:
            continue
        
        # Skip common academic stopwords
        if phrase.lower() in ['et al', 'fig', 'figure', 'table', 'section']:
            continue
        
        filtered.append(phrase)
        
        if len(filtered) >= top_n:
            break
    
    return filtered