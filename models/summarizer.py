import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def score_sentences_improved(text, chunk_embeddings, chunk_positions, keyphrases):
    """Improved sentence scoring using chunk embeddings and keyphrases"""
    sentences = nltk.sent_tokenize(text)
    
    # Map sentences to their positions in text
    sent_positions = []
    current_pos = 0
    for sent in sentences:
        start = text.find(sent, current_pos)
        if start != -1:
            sent_positions.append((start, start + len(sent)))
            current_pos = start + len(sent)
        else:
            sent_positions.append((current_pos, current_pos))
    
    sent_scores = []
    chunk_embs_np = chunk_embeddings.squeeze(0).cpu().numpy()
    
    for i, (sent, (s_start, s_end)) in enumerate(zip(sentences, sent_positions)):
        score = 0.0
        
        # 1. Keyphrase matching (high weight)
        kp_score = sum(1 for kp in keyphrases if kp.lower() in sent.lower())
        score += kp_score * 3.0
        
        # 2. Position-based scoring (favor earlier sentences slightly)
        position_score = 1.0 / (1.0 + i * 0.05)
        score += position_score * 0.5
        
        # 3. Sentence length penalty (avoid very short/long sentences)
        words = sent.split()
        if 5 <= len(words) <= 40:
            score += 1.0
        elif len(words) < 5:
            score -= 0.5
        
        # 4. Chunk embedding similarity
        relevant_chunks = []
        for j, (chunk_ids, chunk_offsets, chunk_start) in enumerate(chunk_positions):
            if chunk_offsets:
                chunk_text_start = chunk_offsets[0][0]
                chunk_text_end = chunk_offsets[-1][1]
                
                # Check if sentence overlaps with chunk
                if not (s_end <= chunk_text_start or s_start >= chunk_text_end):
                    relevant_chunks.append(j)
        
        if relevant_chunks:
            chunk_score = np.mean([np.linalg.norm(chunk_embs_np[j]) for j in relevant_chunks])
            score += chunk_score * 0.3
        
        # 5. Contains important keywords (abstract, conclusion, method, result)
        important_keywords = ['abstract', 'conclusion', 'result', 'method', 'propose', 
                             'framework', 'introduce', 'demonstrate', 'show', 'find']
        if any(kw in sent.lower() for kw in important_keywords):
            score += 1.5
        
        sent_scores.append(score)
    
    return sentences, np.array(sent_scores)

def get_extractive_summary_mmr(sentences, sent_scores, sent_embeddings=None, top_k=5, lambda_param=0.7):
    """Extract summary using MMR for diversity"""
    if len(sentences) == 0:
        return ""
    
    selected_indices = []
    remaining_indices = list(range(len(sentences)))
    
    # Select first sentence with highest score
    first_idx = np.argmax(sent_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Use simple token overlap if embeddings not provided
    def similarity(i, j):
        s1_tokens = set(sentences[i].lower().split())
        s2_tokens = set(sentences[j].lower().split())
        if len(s1_tokens | s2_tokens) == 0:
            return 0
        return len(s1_tokens & s2_tokens) / len(s1_tokens | s2_tokens)
    
    # Select remaining sentences using MMR
    while len(selected_indices) < top_k and remaining_indices:
        mmr_scores = []
        
        for idx in remaining_indices:
            relevance = sent_scores[idx]
            
            # Calculate max similarity to already selected sentences
            max_sim = max([similarity(idx, sel_idx) for sel_idx in selected_indices])
            
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim * 10
            mmr_scores.append((idx, mmr))
        
        # Select sentence with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Sort by original order
    selected_indices.sort()
    selected_sentences = [sentences[i] for i in selected_indices]
    
    return " ".join(selected_sentences)