import logging, json
import enums
from langChainWrapper import LangChainSQLiteRetriever
from sentence_transformers import SentenceTransformer
from retrieverSQL import retrieverSQL
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#
logging.basicConfig(level=logging.INFO)

error_logger = logging.getLogger("errors")
rag_logger = logging.getLogger("rag")

error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
error_logger.addHandler(error_handler)



rag_json_logger = logging.getLogger("rag.json")
rag_json_handler = logging.FileHandler("rag.jsonl")  # JSON Lines format
rag_json_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
rag_json_logger.addHandler(rag_json_handler)
rag_json_logger.setLevel(logging.INFO)


def log_rag_event(event_type, **kwargs):
    rag_json_logger.info(json.dumps({"event": event_type, **kwargs}))

#
embedding_model = SentenceTransformer(enums.modelNames.EMBEDDING_MODEL)

#
retriever_instance = retrieverSQL(db_path=enums.filePaths.DB_PATH.value)
lc_retriever = LangChainSQLiteRetriever(
    retriever=retriever_instance,
    embedding_model=embedding_model,
    top_k=5
)

#
llm = ChatOllama(
    model=enums.modelNames.LLM_MODEL,
    temperature=0.3,
    num_predict=256,
    request_timeout=120 
)

#
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Context:
{context}

Question:
{question}

Answer based ONLY on the above context."""
)

#
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=lc_retriever,
    return_source_documents=True
)

def run_query(question: str):
    try:
        rag_json_logger.info(json.dumps({"event": "query_started", "question": question}))
        result = qa_chain.invoke(question)
        # Log RAG pipeline details
        rag_json_logger.info(json.dumps({
            "event": "query_completed",
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get("source_documents", [])
            ]
        }))
        return result["result"]
    except Exception:
        error_logger.error(("Failed to run query: {q}".format( q = question)),exc_info=True)
        raise

if __name__ == "__main__":
    answer = run_query("what year is the text about?")
    print(answer)
