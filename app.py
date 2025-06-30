import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import faiss
import requests

# --- Configurações da Hugging Face ---
HF_API_TOKEN = st.secrets["hf_token"]
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # modelo para embeddings
LLM_MODEL_ID = "google/flan-t5-base"  # modelo para geração de texto

EMBEDDING_API_URL = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL_ID}"
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_ID}"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Funções auxiliares ---

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text, max_chunk_size=300):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embedding_via_api(text):
    payload = {"inputs": text}
    response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        # resposta geralmente é uma lista de floats
        return np.array(response.json()).astype("float32")
    else:
        st.error(f"Erro ao gerar embedding: {response.text}")
        return None

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        emb = generate_embedding_via_api(chunk)
        if emb is not None:
            embeddings.append(emb)
        else:
            embeddings.append(np.zeros(384, dtype="float32"))  # fallback (dimensão do MiniLM é 384)
    return np.vstack(embeddings)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_huggingface_flant5(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 128}
    }
    response = requests.post(LLM_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    else:
        return f"Erro: {response.text}"

# --- App Streamlit ---

st.set_page_config(page_title="Assistente PDF com FLAN-T5", layout="centered")
st.title("📚 Assistente PDF com LLM")
st.markdown("Faça upload de PDFs e faça perguntas com base no conteúdo.")

# Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload e processamento
with st.sidebar:
    st.header("📂 Upload de PDF")
    pdf_files = st.file_uploader("Escolha seus arquivos", accept_multiple_files=True, type="pdf")
    if st.button("🔍 Processar PDFs"):
        raw_text = extract_text_from_pdf(pdf_files)
        chunks = get_text_chunks(raw_text)
        embeddings = generate_embeddings(chunks)
        index = create_faiss_index(embeddings)

        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.success("PDFs processados com sucesso!")

# Input do usuário
query = st.chat_input("Digite sua pergunta")
if query and st.session_state.faiss_index is not None:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    question_embedding = generate_embedding_via_api(query)
    if question_embedding is None:
        st.error("Erro ao gerar embedding da pergunta.")
    else:
        D, I = st.session_state.faiss_index.search(question_embedding.reshape(1, -1), k=3)
        context = "\n".join([st.session_state.chunks[i] for i in I[0]])

        with st.spinner("Pensando..."):
            answer = query_huggingface_flant5(query, context)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif query:
    st.warning("Por favor, carregue e processe os PDFs primeiro.")

# Rodapé
st.markdown("---")
st.caption("Made by André Arantes")
