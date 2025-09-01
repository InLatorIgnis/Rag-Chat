import enums
#
from langChainWrapper import LangChainSQLiteRetriever

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(enums.modelNames.EMBEDDING_MODEL)

#
from retrieverSQL import retrieverSQL
retriever_instance = retrieverSQL(db_path=enums.filePaths.DB_PATH.value)

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

#
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Context:
{context}

Question:
{question}

Answer based ONLY on the above context."""
)

#
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # simple "stuff" template
    retriever=lc_retriever,
    return_source_documents=True
)

#
result = qa_chain.invoke("create a latex bullet point list of the main steps of the timeline, make it on the format of the included bullet points in the context")
print(result["result"])
