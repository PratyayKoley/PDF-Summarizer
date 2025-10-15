import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

def compute_comprehensive_scores(sentences, chunk_embeddings, chunks, keyphrases, text):
    """Compute multi-factor sentence scores"""
    
    scores = np.zeros(len(sentences))
    
    for i, sent in enumerate(sentences):
        score = 0.0
        sent_lower = sent.lower()
        
        # Factor 1: Keyphrase matching (HIGHEST WEIGHT)
        kp_matches = sum(1 for kp in keyphrases if kp.lower() in sent_lower)
        score += kp_matches * 5.0
        
        # Factor 2: Important section keywords
        important_sections = [
            'abstract', 'introduction', 'conclusion', 'result', 'finding',
            'method', 'approach', 'propose', 'present', 'demonstrate',
            'show', 'achieve', 'improve', 'framework', 'model', 'system'
        ]
        section_matches = sum(1 for keyword in important_sections if keyword in sent_lower)
        score += section_matches * 2.0
        
        # Factor 3: Position bias (favor earlier sentences slightly)
        position_score = 1.0 / (1.0 + i * 0.01)
        score += position_score * 1.0
        
        # Factor 4: Sentence length (prefer medium-length sentences)
        words = sent.split()
        word_count = len(words)
        if 10 <= word_count <= 35:
            score += 2.0
        elif 8 <= word_count <= 40:
            score += 1.0
        elif word_count < 5:
            score -= 2.0  # Penalize very short sentences
        
        # Factor 5: Sentence structure quality
        # Prefer sentences that start with capital and end with period
        if sent[0].isupper() and sent[-1] in '.!?':
            score += 1.0
        
        # Penalize sentences with too many numbers/symbols
        non_alpha = sum(1 for c in sent if not c.isalnum() and c not in ' .,;:!?')
        if non_alpha > len(sent) * 0.2:
            score -= 1.0
        
        # Factor 6: Chunk embedding relevance
        # Find which chunk contains this sentence
        for chunk_idx, chunk_info in enumerate(chunks):
            if i in chunk_info['sentence_indices']:
                # Use chunk embedding magnitude as relevance signal
                chunk_emb_score = np.linalg.norm(chunk_embeddings[chunk_idx])
                score += chunk_emb_score * 0.3
                break
        
        # Factor 7: Avoid reference/citation sentences
        if re.search(r'\[\d+\]|\(\d{4}\)|et al\.', sent):
            score -= 1.5
        
        # Factor 8: Avoid table/figure captions
        if re.match(r'^(Table|Figure|Fig\.|Equation)\s+\d', sent, re.I):
            score -= 3.0
        
        scores[i] = max(0, score)  # Ensure non-negative
    
    return scores