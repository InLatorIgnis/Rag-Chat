import logging
import enums
from langChainWrapper import LangChainSQLiteRetriever
from sentence_transformers import SentenceTransformer
from retrieverSQL import retrieverSQL
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)  # avoid duplicate root handlers

error_logger = logging.getLogger("errors")
rag_logger = logging.getLogger("rag")

error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
error_logger.addHandler(error_handler)

rag_handler = logging.FileHandler("rag.log")
rag_handler.setLevel(logging.INFO)
rag_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
rag_logger.addHandler(rag_handler)

# --- Embedding model ---
embedding_model = SentenceTransformer(enums.modelNames.EMBEDDING_MODEL)

# --- Retriever ---
retriever_instance = retrieverSQL(db_path=enums.filePaths.DB_PATH.value)
lc_retriever = LangChainSQLiteRetriever(
    retriever=retriever_instance,
    embedding_model=embedding_model,
    top_k=5
)

# --- LLM ---
llm = ChatOllama(
    model=enums.modelNames.LLM_MODEL,
    temperature=0.3,
    num_predict=256,
    request_timeout=120 
)

# --- Prompt ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Context:
{context}

Question:
{question}

Answer based ONLY on the above context."""
)

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=lc_retriever,
    return_source_documents=True
)


def run_query(question: str):
    try:
        rag_logger.info("Running query: %s", question)
        result = qa_chain.invoke(question)
        
        # Log RAG pipeline details
        rag_logger.info("Answer: %s", result["result"])
        if "source_documents" in result:
            for i, doc in enumerate(result["source_documents"], 1):
                rag_logger.info("Source %d: %s", i, doc.page_content[:200])  # snippet
        return result["result"]
    except Exception:
        error_logger.error("Failed to run query: %s", question, exc_info=True)
        raise


if __name__ == "__main__":
    answer = run_query("what is the text about?")
    print(answer)
