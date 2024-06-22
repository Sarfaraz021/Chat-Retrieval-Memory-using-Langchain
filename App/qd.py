from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_community.document_loaders import TextLoader

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define the relative and absolute paths for the data file
relative_path = 'data'
filename = 'dummy.txt'  # just to initialize the retriever
absolute_path = os.path.join(relative_path, filename)

# Load and split the document
loader = TextLoader(absolute_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Define the Qdrant URL
url = "http://localhost:6333"

# Initialize Qdrant
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=False,  # Disable gRPC to use HTTP
    collection_name="my_documents",
)

# Perform a similarity search
query = "What could make the tanker car implode?"
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)
