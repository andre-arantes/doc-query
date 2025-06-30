# Assistente Conversacional para PDFs
### Link para a aplicação: https://doc-query-skuywzbxibjqoij8vx5yax.streamlit.app/
Este projeto implementa um assistente conversacional baseado em LLM (Large Language Model) com capacidade de ler documentos PDF, indexá-los com embeddings e responder perguntas sobre seu conteúdo.

## Tecnologias usadas

- Python
- Streamlit
- LangChain
- HuggingFace Transformers
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- PyPDF2

## O que ele faz

1. Lê PDFs carregados pelo usuário
2. Separa o texto em trechos curtos
3. Gera vetores (embeddings) com `sentence-transformers`
4. Indexa os vetores usando FAISS
5. Usa um modelo LLM (`flan-t5-base`) para responder perguntas com base nos PDFs

## Como usar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
2. Rode a aplicação
    ```bash
    streamlit run seu_script.py
3. Carregue seus PDFs na barra lateral, processe e comece a fazer perguntas no chat.


### obs: 
Para enviar pelo drive, o venv de execução local teve que ser excluido. Não foi possivel rodar localmente com o torch local no meu caso, uma vez que o pacote `torch` é muito pesado e grande, mas a aplicação que está rodando no link no inicio deste arquivo está em concordância com o código e os requirements dessa pasta.