from livekit_agents import Agent, OpenAIChatLanguageModel
from livekit_agents.retrievers import DocumentMemoryRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("lesson_konspekt.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

retriever = DocumentMemoryRetriever.from_documents(docs)

llm = OpenAIChatLanguageModel(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    language="pl",
)

agent = Agent(
    name="Asystent Lekcji",
    purpose="Pomagam w zrozumieniu treści lekcji na podstawie konspektu Batyskaf.",
    retriever=retriever,
    llm=llm,
    initial_message="Cześć! Jestem asystentem lekcji. O co chcesz zapytać?",
)

agent.serve()
