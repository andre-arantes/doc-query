import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Configurações iniciais
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = "google/flan-t5-large"

@st.cache_resource
def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return tokenizer, model

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            st.error(f"Erro ao extrair texto do PDF {pdf.name}: {e}")
    return text

def get_text_chunks(text, max_chunk_size=300):
    chunks = []
    current_chunk = ""
    for sentence in text.split('. '):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if chunk]

def generate_embeddings(text_chunks):
    if not text_chunks:
        return np.array([])
    embeddings = EMBEDDING_MODEL.encode(text_chunks, show_progress_bar=False)
    return embeddings.astype('float32')

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def generate_response_flant5(query, context_chunks, tokenizer, model, max_input_length=512):
    # Junta os contextos numa string
    context_str = "\n".join(context_chunks)
    prompt = f"Responda a pergunta com base no texto abaixo:\n\n{context_str}\n\nPergunta: {query}\nResposta:"
    
    # Tokenizar entrada e truncar para max tokens permitidos
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_input_length, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remover o próprio prompt da resposta, se aparecer
    response = response.replace(prompt, "").strip()
    return response

# Interface Streamlit
st.set_page_config(page_title="Assistente PDF com FLAN-T5-Large", layout="centered")
st.title("📄 Assistente PDF com FLAN-T5-Large")

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed" not in st.session_state:
    st.session_state.processed = False

pdf_files = st.file_uploader("Envie seus arquivos PDF", accept_multiple_files=True, type=["pdf"])

if st.button("Processar PDFs"):
    if pdf_files:
        with st.spinner("Extraindo texto e criando índice..."):
            raw_text = extract_text_from_pdf(pdf_files)
            chunks = get_text_chunks(raw_text)
            embeddings = generate_embeddings(chunks)
            faiss_index = create_faiss_index(embeddings)
            
            st.session_state.text_chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = faiss_index
            st.session_state.processed = True
            
        st.success(f"Processados {len(pdf_files)} arquivos com {len(chunks)} chunks.")
    else:
        st.warning("Por favor, envie ao menos um arquivo PDF.")

tokenizer, model = load_tokenizer_and_model()

query = st.text_input("Faça sua pergunta sobre os PDFs processados")

if st.session_state.processed and query:
    # Gerar embedding da pergunta
    query_embedding = EMBEDDING_MODEL.encode([query]).astype('float32')
    D, I = st.session_state.faiss_index.search(query_embedding, k=3)
    # Recuperar chunks relevantes
    relevant_chunks = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]
    
    with st.spinner("Gerando resposta..."):
        answer = generate_response_flant5(query, relevant_chunks, tokenizer, model)
    st.markdown(f"**Resposta:** {answer}")
elif query:
    st.warning("Por favor, faça o upload e processe os PDFs antes de perguntar.")

st.markdown("---")
st.caption("Made by André Arantes")
