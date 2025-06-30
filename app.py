import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, pipeline

# Modelo de embeddings leve e rápido
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
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
    return chunks

def generate_embeddings(text_chunks):
    return EMBEDDING_MODEL.encode(text_chunks, show_progress_bar=False).astype('float32')

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def truncate_context(context, question, max_tokens=800):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    prompt = f"{context}\n\nPergunta: {question}\nResposta:"
    tokens = tokenizer.encode(prompt)
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return truncated

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=200,
        do_sample=False,
        temperature=0.1,
    )

def generate_response(query, context):
    if not context:
        return "Não encontrei informações relevantes nos documentos."

    context_str = "\n".join(context)
    context_str = truncate_context(context_str, query, max_tokens=800)
    prompt = f"Leia o conteúdo abaixo e responda de forma direta à pergunta.\n\n{context_str}\n\nPergunta: {query}\nResposta:"

    llm = load_llm()
    output = llm(prompt)[0]['generated_text']
    return output.strip()

# Streamlit UI
st.set_page_config(page_title="Assistente PDF", layout="centered")
st.title("📚 Assistente PDF")

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
    if not st.session_state.documents_processed:
        st.error("Faça upload e processe os PDFs antes de fazer perguntas.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        query_embedding = EMBEDDING_MODEL.encode([user_query]).astype('float32')
        D, I = st.session_state.faiss_index.search(query_embedding, k=3)
        relevant_context = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]

        with st.spinner("Gerando resposta..."):
            response = generate_response(user_query, relevant_context)

        with st.chat_message("assistant"):
            st.markdown(f"**Resposta:**\n{response}")
        st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.title("Documentos")
    pdf_docs = st.file_uploader("Carregue arquivos PDF", accept_multiple_files=True, type="pdf")

    if st.button("Processar PDFs"):
        if pdf_docs:
            with st.spinner("Processando PDFs..."):
                raw_text = extract_text_from_pdf(pdf_docs)
                st.session_state.text_chunks = get_text_chunks(raw_text)
                st.session_state.embeddings = generate_embeddings(st.session_state.text_chunks)
                st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
                st.session_state.documents_processed = True
            st.success(f"{len(pdf_docs)} PDFs processados com sucesso!")
        else:
            st.warning("Carregue pelo menos um PDF.")

    if st.session_state.documents_processed and st.button("Resetar"):
        st.session_state.clear()
        st.experimental_rerun()

st.markdown("---")
st.caption("Made by André Arantes")
