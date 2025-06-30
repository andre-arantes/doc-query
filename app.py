import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_API_TOKEN = st.secrets.get("hf_token")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

if not HF_API_TOKEN:
    st.error("Token da Hugging Face não encontrado no secrets.toml!")
    st.stop()

EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text + "\n"
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
    try:
        embeddings = EMBEDDING_MODEL.encode(text_chunks, show_progress_bar=False)
        return embeddings.astype('float32')
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {e}")
        return np.array([])

def create_faiss_index(embeddings):
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource
def load_llm():
    try:
        pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            use_auth_token=HF_API_TOKEN,
            max_new_tokens=200,
            temperature=0.05,
            do_sample=True
        )
        return pipe
    except Exception as e:
        st.error(f"Erro ao carregar LLM: {e}")
        raise

def generate_response(query, context_chunks):
    if not context_chunks:
        return "Não encontrei informações relevantes nos documentos."

    context_str = "\n".join(context_chunks)
    prompt = f"""Leia o conteúdo abaixo e responda de forma direta à pergunta.

[Conteúdo]
{context_str}

[Pergunta]
{query}

[Resposta]
"""

    llm = load_llm()

    try:
        output = llm(prompt)[0]["generated_text"]
        response = output[len(prompt):].strip()
        if not response:
            response = "Não sei"
        return response
    except Exception as e:
        logger.error("Erro ao gerar resposta: %s", e)
        return "Houve um erro ao tentar gerar a resposta."

st.set_page_config(page_title="Assistente PDF com Hugging Face", layout="centered")
st.title("📚 Assistente PDF - Pergunte sobre seus documentos!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Faça uma pergunta sobre o PDF")

if user_query:
    if not st.session_state.documents_processed or st.session_state.faiss_index is None:
        st.error("Faça o upload e processe seus PDFs na barra lateral antes de perguntar.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        try:
            query_embedding = EMBEDDING_MODEL.encode([user_query]).astype('float32')
            D, I = st.session_state.faiss_index.search(query_embedding, k=3)
            relevant_context = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]
            with st.spinner("Gerando resposta..."):
                response = generate_response(user_query, relevant_context)
            with st.chat_message("assistant"):
                st.markdown(f"**Resposta:**\n{response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")

with st.sidebar:
    st.title("Seus Documentos")
    pdf_docs = st.file_uploader("Carregue seus arquivos PDF aqui e clique em 'Processar'", accept_multiple_files=True, type="pdf")
    if st.button("Processar PDFs"):
        if pdf_docs:
            try:
                with st.spinner("Processando PDFs..."):
                    raw_text = extract_text_from_pdf(pdf_docs)
                    st.session_state.text_chunks = get_text_chunks(raw_text)
                    st.session_state.embeddings = generate_embeddings(st.session_state.text_chunks)
                    st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
                    st.session_state.documents_processed = True
                    st.success(f"{len(pdf_docs)} PDFs processados com sucesso!")
            except Exception as e:
                st.error(f"Erro ao processar PDFs: {e}")
        else:
            st.warning("Carregue pelo menos um arquivo PDF para processar.")

    if st.session_state.faiss_index is not None and st.button("Resetar e fazer upload de novos PDFs"):
        st.session_state.documents_processed = False
        st.session_state.messages = []
        st.session_state.text_chunks = []
        st.session_state.embeddings = None
        st.session_state.faiss_index = None
        st.experimental_rerun()

st.markdown("---")
st.caption("Made by André Arantes")
