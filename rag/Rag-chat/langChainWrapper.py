from langchain.schema import BaseRetriever, Document
from pydantic import Field
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document

import embedd


class LangChainSQLiteRetriever(BaseRetriever):
    """Custom LangChain Retriever using SQLite as the vector store."""
    retriever: object
    embedding_model: object
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = self.embedding_model.encode([query])[0]
        top_chunks = self.retriever.retrieve(query_embedding, top_k=self.top_k)
        return [Document(page_content=chunk) for chunk in top_chunks]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def get_relevant_texts(self, query: str):
        docs = self.get_relevant_documents(query)
        return [d.page_content for d in docs]
