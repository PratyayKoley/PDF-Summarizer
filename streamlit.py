import streamlit as st
import requests
import time
import json

# Configuration
st.set_page_config(
    page_title="Academic PDF Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .summary-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #1e88e5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.05rem;
        line-height: 1.8;
    }
    .keyphrase-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        margin: 5px;
        border-radius: 25px;
        font-weight: 500;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        color: #333;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://127.0.0.1:8000"

# Header
st.markdown('<h1 class="main-header">📚 Academic PDF Summarizer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Transform research papers into coherent, extractive summaries using advanced NLP</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    summary_length = st.slider(
        "Summary Length (sentences)",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of sentences to extract for the summary"
    )
    
    st.markdown("---")
    
    st.subheader("🎯 Key Features")
    st.markdown("""
    - ✅ **Smart Chunking**: Preserves sentence boundaries
    - ✅ **Keyphrase Weighting**: Focuses on important concepts
    - ✅ **Multi-Factor Scoring**: 8 different relevance signals
    - ✅ **MMR Selection**: Ensures diverse, non-redundant output
    - ✅ **Academic Optimized**: Trained on scientific papers
    """)
    
    st.markdown("---")
    
    st.subheader("📖 How It Works")
    with st.expander("View Pipeline"):
        st.markdown("""
        1. **Text Extraction**: Clean PDF parsing
        2. **Keyphrase Detection**: Identify key concepts (KeyBERT)
        3. **Smart Chunking**: Sentence-aware segmentation
        4. **Embedding**: BERT-based representations
        5. **Scoring**: Multi-factor relevance calculation
        6. **Selection**: MMR for diversity (λ=0.7)
        """)
    
    st.markdown("---")
    
    # API Health Check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Connected")
            data = health.json()
            st.caption(f"Device: {data.get('device', 'N/A')}")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.caption("Start with: `python main.py`")

# Main Content
st.markdown("### 📤 Upload Your Document")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a PDF file (max 50MB)",
        type=["pdf"],
        help="Upload academic papers, research articles, or technical documents"
    )

with col2:
    st.markdown("**Supported Files:**")
    st.markdown("- 📄 PDF documents")
    st.markdown("- 📊 Research papers")
    st.markdown("- 📚 Technical reports")

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > 50:
        st.error(f"❌ File too large: {file_size_mb:.2f} MB (max 50MB)")
    else:
        st.markdown(
            f'<div class="success-box">✅ <strong>{uploaded_file.name}</strong> uploaded successfully ({file_size_mb:.2f} MB)</div>',
            unsafe_allow_html=True
        )
        
        process_button = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True
        )
        
        if process_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Check API
                status_text.text("🔍 Checking API connection...")
                progress_bar.progress(10)
                
                health_response = requests.get(f"{API_URL}/health", timeout=5)
                if health_response.status_code != 200:
                    st.error("❌ API not responding. Please start the FastAPI server.")
                    st.stop()
                
                # Upload and process
                status_text.text("📤 Uploading document...")
                progress_bar.progress(20)
                
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                params = {"summary_sentences": summary_length}
                
                status_text.text("🔄 Processing (this may take 30-60 seconds)...")
                progress_bar.progress(40)
                
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    params=params,
                    timeout=180
                )
                processing_time = time.time() - start_time
                
                progress_bar.progress(100)
                status_text.empty()
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.balloons()
                    st.success(f"✅ Summary generated in {processing_time:.1f} seconds!")
                    
                    # Summary Section
                    st.markdown("---")
                    st.markdown("## 📝 Extractive Summary")
                    
                    summary_text = data.get("summary", "")
                    if summary_text:
                        st.markdown(
                            f'<div class="summary-box">{summary_text}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("⚠️ No summary generated")
                    
                    # Keyphrases Section
                    st.markdown("---")
                    st.markdown("## 🔑 Key Concepts")
                    
                    keyphrases = data.get("keyphrases", [])
                    if keyphrases:
                        keyphrase_html = "".join([
                            f'<span class="keyphrase-tag">{kp}</span>'
                            for kp in keyphrases
                        ])
                        st.markdown(keyphrase_html, unsafe_allow_html=True)
                    else:
                        st.info("No keyphrases extracted")
                    
                    # Statistics
                    st.markdown("---")
                    st.markdown("## 📊 Document Analytics")
                    
                    stats = data.get("stats", {})
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    
                    with col_s1:
                        st.metric(
                            "Total Sentences",
                            stats.get("num_sentences", "N/A")
                        )
                    
                    with col_s2:
                        st.metric(
                            "Document Chunks",
                            stats.get("num_chunks", "N/A")
                        )
                    
                    with col_s3:
                        st.metric(
                            "Summary Sentences",
                            stats.get("summary_length", "N/A")
                        )
                    
                    with col_s4:
                        compression_ratio = (
                            stats.get("summary_length", 0) / stats.get("num_sentences", 1) * 100
                        )
                        st.metric(
                            "Compression",
                            f"{compression_ratio:.1f}%"
                        )
                    
                    # Quality Indicators
                    st.markdown("---")
                    st.markdown("## ✨ Quality Metrics")
                    
                    col_q1, col_q2 = st.columns(2)
                    
                    with col_q1:
                        st.markdown("**Summary Length**")
                        summary_words = len(summary_text.split())
                        st.info(f"📊 {summary_words} words, {len(summary_text)} characters")
                    
                    with col_q2:
                        st.markdown("**Keyphrase Coverage**")
                        kp_in_summary = sum(1 for kp in keyphrases if kp.lower() in summary_text.lower())
                        coverage = (kp_in_summary / len(keyphrases) * 100) if keyphrases else 0
                        st.info(f"🎯 {kp_in_summary}/{len(keyphrases)} keyphrases ({coverage:.0f}% coverage)")
                    
                    # Download Section
                    st.markdown("---")
                    st.markdown("## 💾 Export Results")
                    
                    # Prepare download content
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    download_content = f"""
ACADEMIC PDF SUMMARY
{'='*80}

Document: {uploaded_file.name}
Generated: {timestamp}
Processing Time: {processing_time:.2f} seconds

EXTRACTIVE SUMMARY
{'='*80}

{summary_text}

KEY CONCEPTS
{'='*80}

{', '.join(keyphrases)}

DOCUMENT STATISTICS
{'='*80}

Total Sentences: {stats.get("num_sentences", "N/A")}
Document Chunks: {stats.get("num_chunks", "N/A")}
Summary Sentences: {stats.get("summary_length", "N/A")}
Compression Ratio: {compression_ratio:.1f}%
File Size: {stats.get("file_size_kb", "N/A")} KB

Summary Word Count: {summary_words}
Summary Character Count: {len(summary_text)}
Keyphrase Coverage: {coverage:.1f}%

METHODOLOGY
{'='*80}

- Extraction Method: Extractive Summarization
- Keyphrase Detection: KeyBERT with MMR
- Embeddings: BERT-base-uncased
- Chunking: Sentence-aware with overlap
- Selection: Multi-factor scoring + MMR (λ=0.7)
- Scoring Factors: 8 (keyphrases, position, length, structure, etc.)

Generated by Academic PDF Summarizer v2.0
                    """
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        st.download_button(
                            label="📄 Download Summary (TXT)",
                            data=download_content,
                            file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col_d2:
                        # JSON export
                        json_data = {
                            "filename": uploaded_file.name,
                            "generated_at": timestamp,
                            "summary": summary_text,
                            "keyphrases": keyphrases,
                            "statistics": stats
                        }
                        
                        st.download_button(
                            label="📊 Download Data (JSON)",
                            data=json.dumps(json_data, indent=2),
                            file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                else:
                    error_data = response.json()
                    st.error(f"❌ Error {response.status_code}: {error_data.get('detail', 'Unknown error')}")
                    
                    with st.expander("🔍 View Error Details"):
                        st.json(error_data)
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The document might be too large or complex.")
                st.info("💡 Try: Reduce summary length or upload a smaller document")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server")
                st.markdown("""
                    <div class="warning-box">
                    <strong>⚠️ Server Not Running</strong><br>
                    Please start the FastAPI backend:<br>
                    <code>python main.py</code><br>
                    The server should be available at http://127.0.0.1:8000
                    </div>
                """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                with st.expander("🔍 View Traceback"):
                    st.code(str(e))

else:
    # Show example/instructions when no file uploaded
    st.markdown("---")
    
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("### 📖 Sample Use Cases")
        st.markdown("""
        **Perfect for:**
        - 📚 Research paper summarization
        - 📊 Technical report digests
        - 🔬 Scientific article abstracts
        - 📄 Academic literature reviews
        - 🎓 Thesis/dissertation summaries
        """)
    
    with col_ex2:
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        **Recommendations:**
        - Use well-formatted PDF documents
        - Text-based PDFs work best (not scanned images)
        - Papers with clear structure yield better summaries
        - 5-7 sentences provide balanced summaries
        - Longer papers may take 30-60 seconds
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 What Makes This Different?")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        **🧠 Intelligent Scoring**
        
        Uses 8 different factors:
        - Keyphrase matching
        - Section keywords
        - Position bias
        - Sentence quality
        - Length optimization
        - Structure validation
        - Reference filtering
        - Embedding relevance
        """)
    
    with col_f2:
        st.markdown("""
        **🎨 Diversity Guarantee**
        
        MMR algorithm ensures:
        - No redundant sentences
        - Broad topic coverage
        - Balanced representation
        - Original document order
        - Coherent flow
        """)
    
    with col_f3:
        st.markdown("""
        **⚡ Smart Processing**
        
        Advanced techniques:
        - Sentence-aware chunking
        - Overlap for context
        - Keyphrase weighting
        - BERT embeddings
        - Multi-head attention
        - Academic optimization
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 30px;'>
        <h3>🚀 Powered by State-of-the-Art NLP</h3>
        <p>
            <strong>Technologies:</strong> BERT • KeyBERT • Sentence-Transformers • PyTorch • FastAPI • Streamlit
        </p>
        <p style='margin-top: 10px;'>
            <em>Extractive Summarization with Transformer-Enhanced Graph Networks</em>
        </p>
        <p style='margin-top: 20px; font-size: 0.9rem;'>
            Built with ❤️ for academic research and technical documentation
        </p>
    </div>
""", unsafe_allow_html=True)