import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from livekit.agents import Agent, AgentSession, WorkerOptions
from livekit.plugins import openai, silero

# --- Wczytanie zmiennych środowiskowych ---
load_dotenv()

# --- Automatyczne wykrywanie pierwszego PDF ---
pdf_files = glob.glob("*.pdf")
if not pdf_files:
    raise FileNotFoundError("Nie znaleziono żadnych plików PDF w katalogu")
pdf_path = pdf_files[0]
print(f"Wczytuję plik PDF: {pdf_path}")

# --- Wczytanie dokumentu i podział na fragmenty ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# --- Tworzenie bazy wektorowej z konspektu ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
vector_store = Chroma.from_documents(docs, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = openai.LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")


class LessonAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            instructions=(
                "Jesteś edukacyjnym asystentem AI. "
                "Odpowiadasz wyłącznie na podstawie dostarczonego konspektu lekcji. "
                "Jeżeli odpowiedź nie jest zawarta w konspekcie, napisz: 'Nie mam takiej informacji w konspekcie'. "
                "Po każdej odpowiedzi zaproponuj użytkownikowi kolejne pytanie związane z lekcją."
            ),
            **kwargs
        )
        self.retriever = retriever
        self.llm = llm
        self.name = "Asystent Lekcji"

    async def process_message(self, message, context):
        # Pobierz istotne fragmenty z konspektu
        relevant_docs = self.retriever.get_relevant_documents(message)
        context_text = "\n".join([d.page_content for d in relevant_docs])

        prompt = (
            "Odpowadaj wyłącznie na podstawie poniższego konspektu lekcji.\n"
            "Jeżeli odpowiedź nie jest zawarta w konspekcie, napisz: 'Nie mam takiej informacji w konspekcie.'\n\n"
            f"Konspekt:\n{context_text}\n\n"
            f"Pytanie: {message}\n"
            "Odpowiedź:"
        )

        response = await self.llm.run(prompt)
        return f"Z konspektu wynika, że {response.strip()} \n\nCo jeszcze chcesz wiedzieć?"


async def entrypoint(ctx):
    await ctx.connect()
    session = AgentSession(
        llm=llm,
        tts=openai.TTS(),
        vad=silero.VAD.load()
    )
    agent = LessonAgent()
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions="Cześć! Jestem asystentem lekcji. O co chcesz mnie zapytać?"
    )


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
