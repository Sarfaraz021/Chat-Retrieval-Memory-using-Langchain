import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
import openai
from langchain_qdrant import Qdrant

# Load environment variables from the .env file
load_dotenv('var.env')

# Define a class for the chatbot


class UniversityAdmissionChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.directory_path = "data"
        self.documents = self.load_documents()
        self.docs = self.split_documents(self.documents)
        self.embeddings = OpenAIEmbeddings()
        self.vectbd = self.initialize_pinecone()
        self.retriever = self.vectbd.as_retriever()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.prompt_template = self.create_prompt_template()
        self.chain = self.create_retrieval_qa_chain()

    def load_documents(self):
        """Load documents from the specified directory."""
        loader = DirectoryLoader(self.directory_path)
        return loader.load()

    def split_documents(self, documents):
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    def initialize_pinecone(self):
        """Initialize Pinecone vector store with the documents and embeddings."""
        Pinecone(api_key=self.pinecone_api_key,
                 environment='REPLACE WITH YOUR ENV NAME OF PINECONE (e.g gcp-starter) ')
        return PineconeVectorStore.from_documents(self.docs, self.embeddings, index_name=self.index_name)

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
                "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
            }
        )

    def get_response(self, user_input):
        """Get a response from the chatbot based on user input."""
        return self.chain.invoke(user_input)['result']

# Main function to run the chatbot in a loop


def main():
    chatbot = UniversityAdmissionChatbot()
    while True:
        prompt = input("User> ")
        if prompt.lower() == 'exit':
            break
        else:
            response = chatbot.get_response(prompt)
            print(f"AI Assistant: {response}")
            print("*********************************")

# Uncomment the following lines to use Streamlit for UI
# def run_streamlit_ui():
#     import streamlit as st
#     st.title("AI-Driven Chatbot for University Admission & Enrollment")
#     chatbot = UniversityAdmissionChatbot()
#     user_input = st.text_input("Your query:")
#     if user_input:
#         response = chatbot.get_response(user_input)
#         st.write(response)


if __name__ == "__main__":
    main()
    # Uncomment the following line to run Streamlit UI
    # run_streamlit_ui()
