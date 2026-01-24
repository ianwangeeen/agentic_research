# pdf_rag_agent.py

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
import os
from anthropic import Anthropic
from dotenv import load_dotenv


class PDFRAGAgent:
    load_dotenv()
    def __init__(self, api_key=os.getenv('API_KEY')):
        self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        self.llm = ChatAnthropic(
                    model="claude-sonnet-4-5",
                    anthropic_api_key=api_key,
                    temperature=0.3,
                    max_tokens=4096
                )
        self.vectorstore = None
        self.qa_chain = None
        
    def load_and_process_pdf(self, pdf_path):
        """Load PDF, chunk it, and create vector store"""
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="../data/chroma_db"
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 chunks
            ),
            return_source_documents=True
        )
        
        return f"✅ Processed {len(chunks)} chunks from PDF"
    
    def query_pdf(self, question):
        """Query the PDF using RAG"""
        if not self.qa_chain:
            return "⚠️ Please upload a PDF first!"
        
        result = self.qa_chain({"query": question})
        
        # Format response with sources
        answer = result['result']
        sources = result['source_documents']
        
        response = f"**Answer:**\n{answer}\n\n**Sources:**\n"
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get('page', 'unknown')
            response += f"\n{i}. Page {page}: {doc.page_content[:200]}..."
        
        return response