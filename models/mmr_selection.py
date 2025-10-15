from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def mmr_select_sentences(sentences, scores, sentence_embeddings, top_k=5, lambda_param=0.7):
    """Select sentences using Maximal Marginal Relevance for diversity"""
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) <= top_k:
        return sentences
    
    selected_indices = []
    remaining_indices = list(range(len(sentences)))
    
    # Start with highest scoring sentence
    first_idx = np.argmax(scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Select remaining sentences
    while len(selected_indices) < top_k and remaining_indices:
        mmr_scores = {}
        
        for idx in remaining_indices:
            # Relevance component
            relevance = scores[idx]
            
            # Diversity component (similarity to already selected)
            if sentence_embeddings is not None:
                selected_embs = sentence_embeddings[selected_indices]
                candidate_emb = sentence_embeddings[idx].reshape(1, -1)
                
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_similarity = np.max(similarities)
            else:
                # Fallback: token overlap
                candidate_tokens = set(sentences[idx].lower().split())
                max_similarity = 0
                for sel_idx in selected_indices:
                    selected_tokens = set(sentences[sel_idx].lower().split())
                    if len(candidate_tokens | selected_tokens) > 0:
                        sim = len(candidate_tokens & selected_tokens) / len(candidate_tokens | selected_tokens)
                        max_similarity = max(max_similarity, sim)
            
            # MMR formula
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity * 10
            mmr_scores[idx] = mmr
        
        # Select best MMR score
        best_idx = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Return sentences in original order
    selected_indices.sort()
    return [sentences[i] for i in selected_indices]