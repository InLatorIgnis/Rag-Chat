import enums
#
from LangChainWrapper import LangChainSQLiteRetriever

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#
from SQLiteRetriever import SQLiteRetriever
retriever_instance = SQLiteRetriever(db_path="embeddings.db")

#
lc_retriever = LangChainSQLiteRetriever(
    retriever=retriever_instance,
    embedding_model=embedding_model,
    top_k=5
)
#
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model=enums.modelNames.LLM_MODEL,
    temperature=0.3,    # lower = less verbose
    num_predict=256,    # hard cap on tokens in output
#    stop=["\n\n"]       # optional, to cut off after a double newline
)

# 5️⃣ Optional prompt template
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Context:
{context}

Question:
{question}

Answer based ONLY on the above context."""
)

# 6️⃣ Create RAG chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # simple "stuff" template
    retriever=lc_retriever,
    return_source_documents=True
)

# 7️⃣ Ask a question
result = qa_chain.invoke("how do we solve question 6?")
print(result["result"])
