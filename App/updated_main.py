import os
import logging
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2Model
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
import time
import uuid
from langchain_qdrant import Qdrant

# Load environment variables from the .env file
load_dotenv('var.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GPT-2 model and tokenizer for embeddings
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2Model.from_pretrained('gpt2')

# Initialize Qdrant client
client = QdrantClient("http://localhost:6333")

# Check if the collection exists, and create it if it doesn't
if not client.collection_exists("memories"):
    client.create_collection(
        collection_name="memories",
        vectors_config=models.VectorParams(
            size=768, distance=models.Distance.COSINE),
    )

# Define a class for the chatbot


class Main:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # Define the relative and absolute paths for the data file
        self.relative_path = 'data'
        self.filename = 'dummy.txt'  # just to initialize the retriever
        self.absolute_path = os.path.join(self.relative_path, self.filename)
        self.documents = self.load_documents()
        self.docs = self.split_documents(self.documents)
        self.embeddings = OpenAIEmbeddings()
        self.vectbd = self.initialize_qdrant()
        self.retriever = self.vectbd.as_retriever()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.memory = ConversationBufferMemory(
            memory_key="history", input_key="question")
        self.prompt_template = self.create_prompt_template()
        self.chain = self.create_retrieval_qa_chain()

    def load_documents(self):
        """Load documents from the specified directory."""
        loader = TextLoader(self.absolute_path)
        return loader.load()

    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def initialize_qdrant(self):
        """Initialize Qdrant vector store with the documents and embeddings."""
        qdrant = Qdrant.from_documents(
            self.docs,
            self.embeddings,
            url="http://localhost:6333",
            prefer_grpc=False,  # Disable gRPC to use HTTP
            collection_name="my_documents",
        )
        return qdrant

    def create_prompt_template(self):
        """Create a prompt template for the conversation."""
        from prompt import template
        return PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

    def create_retrieval_qa_chain(self):
        """Create a RetrievalQA chain with the specified components."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={
                "verbose": False,
                "prompt": self.prompt_template,
                "memory": self.memory,
            }
        )

    def vectorize_text(self, text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = gpt2_model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).detach().numpy()

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def exponential_decay(self, relevance, elapsed_time, decay_rate):
        return relevance * np.exp(-decay_rate * elapsed_time)

    def recall_probability(self, relevance, elapsed_time, decay_rate):
        return 1 - np.exp(-self.exponential_decay(relevance, elapsed_time, decay_rate))

    def recall_memory(self, user_id, input_vec, threshold=0.86, decay_rate=0.01):
        current_time = time.time()
        response = client.search(
            collection_name="memories",
            query_vector=input_vec.flatten().tolist(),
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )]
            ),
            limit=10
        )
        logger.info(f"Recall memory response: {response}")
        for result in response:
            memory_vec = result.vector
            if memory_vec is None:
                logger.warning(f"Memory vector is None for result: {result}")
                continue
            relevance = self.cosine_similarity(memory_vec, input_vec)
            elapsed_time = current_time - result.payload['timestamp']
            prob = self.recall_probability(relevance, elapsed_time, decay_rate)
            if prob > threshold:
                return result.payload['content']
        return None

    def store_memory(self, user_id, content):
        vector = self.vectorize_text(content).flatten().tolist()
        timestamp = time.time()
        memory_id = str(uuid.uuid4())
        client.upsert(
            collection_name="memories",
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload={"user_id": user_id,
                             "content": content, "timestamp": timestamp},
                )
            ]
        )

    def get_response(self, user_id, user_input):
        input_vec = self.vectorize_text(user_input)
        recalled_memory = self.recall_memory(user_id, input_vec)
        context = "\n".join([doc.page_content for doc in self.docs])
        if recalled_memory:
            context += f"\nRecalled Memory: {recalled_memory}"
        prompt = self.prompt_template.format(
            history=self.memory.load_memory_variables({})['history'],
            context=context,
            question=user_input
        )
        response = self.chain.invoke({"query": user_input, "context": context})
        self.store_memory(user_id, user_input)
        return response['result']

# Main function to run the chatbot in a loop


def main():
    chatbot = Main()
    # Replace 'your_name_here' with your actual name or desired identifier
    user_id = 'Ahmed'
    while True:
        prompt = input("User> ")
        if prompt.lower() == 'exit':
            break
        else:
            response = chatbot.get_response(user_id, prompt)
            print(f"AI Assistant: {response}")
            print("*********************************")

# Uncomment the following lines to use Streamlit for UI
# def run_streamlit_ui():
#     import streamlit as st
#     st.title("AI-Driven Chatbot")
#     chatbot = Main()
#     user_id = 'example_user'  # Placeholder user ID
#     user_input = st.text_input("Your query:")
#     if user_input:
#         response = chatbot.get_response(user_id, user_input)
#         st.write(response)


if __name__ == "__main__":
    main()
    # Uncomment the following line to run Streamlit UI
    # run_streamlit_ui()
