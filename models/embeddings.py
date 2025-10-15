import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_sentence_embedding(text):
    """Get BERT embedding for a sentence"""
    inputs = bert_tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding.cpu().numpy()

def get_chunk_embeddings_batch(chunks, keyphrases):
    """Get embeddings for chunks with keyphrase weighting"""
    embeddings = []
    
    for chunk_info in chunks:
        chunk_text = chunk_info['text']
        
        # Count keyphrases in chunk
        kp_count = sum(1 for kp in keyphrases if kp.lower() in chunk_text.lower())
        
        # Get base embedding
        emb = get_sentence_embedding(chunk_text)
        
        # Weight by keyphrase presence
        weight = 1.0 + (kp_count * 0.5)
        weighted_emb = emb * weight
        
        embeddings.append(weighted_emb)
    
    return np.array(embeddings)
