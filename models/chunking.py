from transformers import AutoTokenizer
import nltk

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
nltk.download('punkt', quiet=True)

def smart_chunk_by_sentences(text, max_tokens=384, overlap_sentences=2):
    """Chunk text by sentences to preserve semantic boundaries"""
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for i, sent in enumerate(sentences):
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_token_count = len(sent_tokens)
        
        # If single sentence exceeds max, split it
        if sent_token_count > max_tokens:
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'sentence_indices': list(range(i - len(current_chunk), i))
                })
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence into smaller parts
            words = sent.split()
            temp_chunk = []
            temp_tokens = 0
            
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if temp_tokens + word_tokens > max_tokens:
                    if temp_chunk:
                        chunks.append({
                            'text': ' '.join(temp_chunk),
                            'sentence_indices': [i]
                        })
                    temp_chunk = [word]
                    temp_tokens = word_tokens
                else:
                    temp_chunk.append(word)
                    temp_tokens += word_tokens
            
            if temp_chunk:
                chunks.append({
                    'text': ' '.join(temp_chunk),
                    'sentence_indices': [i]
                })
            continue
        
        # Check if adding this sentence exceeds limit
        if current_tokens + sent_token_count > max_tokens:
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'sentence_indices': list(range(i - len(current_chunk), i))
                })
            
            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:] + [sent]
            current_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) 
                               for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_tokens += sent_token_count
    
    # Add remaining chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'sentence_indices': list(range(len(sentences) - len(current_chunk), len(sentences)))
        })
    
    return chunks, sentences
