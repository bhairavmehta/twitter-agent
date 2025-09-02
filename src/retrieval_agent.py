import os
from glob import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

class RetrievalAgent:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-small",
        docs_path: str = "docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        openrouter_api_key: str = None,
        openai_api_key: str = None,  # Added separate OpenAI API key
        temperature: float = 0.0,
        system_prompt: str = (
                "You are a highly focused crypto market analyst. "
                "Rely exclusively on the provided context to answer the question—do not introduce external information. "
                "Summarize the key market trends, events, and statistics concise sentences. "
                "Give the data related to the query given"
                "Context: {context}"
        )
    ):
        """
        Initialize the RetrievalAgent.

        Args:
            model_name: Name of the LLM (served via OpenRouter) to use (e.g. "gpt-4o").
            embedding_model_name: Name of the OpenAI embedding model.
            docs_path: Folder path where text documents reside.
            chunk_size: Maximum length of each text chunk.
            chunk_overlap: Overlap between consecutive chunks.
            openrouter_api_key: OpenRouter API key for LLM (if None, loaded from env).
            openai_api_key: OpenAI API key for embeddings (if None, loaded from env).
            temperature: Temperature setting for LLM generation.
            system_prompt: The system prompt template for retrieval.
        """
        self.model_name = model_name
        self.embedding_model_name = "text-embedding-3-small"
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Handle API keys
        if openrouter_api_key is None:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        self.openrouter_api_key = openrouter_api_key
        self.openai_api_key = openai_api_key
        self.temperature = temperature

        # Initialize the embedding model with OpenAI API key
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.openai_api_key  # Use OpenAI API key for embeddings
        )
        
        # Initialize the LLM with OpenRouter API key
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_api_key,  # Use OpenRouter API key for LLM
            temperature=self.temperature
        )

        # Set up the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.system_prompt = system_prompt

        self.vectorstore = None
        self.chain = None
        self.update_vector_store()

    def load_and_split_documents(self) -> list:
        """
        Load all text files from the fixed docs_path and split them into chunks.
        """
        file_paths = glob(os.path.join(self.docs_path, "**", "*.txt"), recursive=True)
        documents = []
        for path in file_paths:
            loader = TextLoader(path)
            docs = loader.load()
            documents.extend(docs)
        return self.text_splitter.split_documents(documents)

    def update_vector_store(self):
        """
        Rebuild the FAISS vector store from the documents in the fixed folder,
        and recreate the retrieval chain using the custom prompt.
        """
        print("Updating vector store from documents in folder...")
        doc_chunks = self.load_and_split_documents()
        if doc_chunks:
            self.vectorstore = FAISS.from_documents(doc_chunks, self.embeddings)
            retriever = self.vectorstore.as_retriever()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{input}"),
                ]
            )
            qa_chain = create_stuff_documents_chain(self.llm, prompt)
            self.chain = create_retrieval_chain(retriever, qa_chain)
            print(f"Vector store updated with {len(doc_chunks)} document chunks.")
        else:
            print("No documents found in folder. Vector store not updated.")

    def query(self, input_query: str) -> str:
        """
        Invoke the retrieval chain to answer a query.
        """
        if self.chain is None:
            raise ValueError("Retrieval chain is not initialized. Please update the vector store.")
        chain_response = self.chain.invoke({"input": input_query})
        return chain_response.get("answer", chain_response)

    def refresh(self):
        """
        Refresh the vector store and retrieval chain.
        """
        self.update_vector_store()


# if __name__ == "__main__":
#     agent = RetrievalAgent(
#         model_name="gpt-4o",
#         embedding_model_name="text-embedding-ada-002",
#         docs_path=r"docs\automated",
#         chunk_size=1000,
#         chunk_overlap=200,
#         openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
#         openai_api_key=os.getenv("OPENAI_API_KEY"),  # Added OpenAI API key
#         temperature=0.0,
#     )

#     query = (
#         "Analyze the provided documents and extract key insights about current crypto market trends. "
#         "Only include information that is present in the documents. If there is insufficient data, state that the "
#         "information is limited. Provide a clear, concise summary labeled 'Crypto Market Summary'."
#     )

#     response = agent.query(query)
#     print("Agent response:\n", response)