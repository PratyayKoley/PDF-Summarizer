import fitz  # PyMuPDF
import re
from collections import Counter

def pdf_to_text_pymupdf(path):
    """
    Extract text using PyMuPDF - often handles spacing better than pdfplumber
    """
    doc = fitz.open(path)
    all_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with proper spacing
        text = page.get_text("text")
        
        if text:
            all_text.append(text)
    
    doc.close()
    
    full_text = "\n".join(all_text)
    
    # Clean the text
    full_text = clean_extracted_text(full_text)
    
    return full_text


def clean_extracted_text(text):
    """
    Comprehensive text cleaning
    """
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    
    # Remove DOIs
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.I)
    
    # Remove page numbers (various formats)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\b(page|p\.)\s*\d+\b', '', text, flags=re.I)
    
    # Remove copyright symbols
    text = re.sub(r'[©®™].*?\n', '', text)
    
    # Remove figure/table references that are standalone
    text = re.sub(r'\n\s*(Figure|Table|Fig\.)\s+\d+[:\.].*?\n', '\n', text, flags=re.I)
    
    # Remove author affiliations (common patterns)
    text = re.sub(r'\d+\s*(Department|University|Institute|College).*?\n', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove non-ASCII
    text = re.sub(r'[^\x00-\x7F\n]+', '', text)
    
    # Remove headers/footers (repeated lines)
    lines = text.split('\n')
    line_counts = Counter(lines)
    
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        
        # Skip empty or very short lines
        if len(line) < 5:
            continue
        
        # Skip repeated lines (headers/footers)
        if line_counts[line] > 3:
            continue
        
        cleaned_lines.append(line)
    
    # Rejoin with single space
    full_text = ' '.join(cleaned_lines)
    
    # Final cleanup
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return full_text


# Alternative Solution 2: Use pdfminer.six - Most reliable for text extraction
# Install: pip install pdfminer.six
