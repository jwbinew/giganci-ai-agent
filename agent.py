import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob

from livekit.agents import Agent, AgentSession, WorkerOptions
from livekit.plugins import openai, silero

load_dotenv()

# Automatyczne wykrywanie pierwszego pliku PDF
pdf_files = glob.glob("*.pdf")
if not pdf_files:
    raise FileNotFoundError("Nie znaleziono żadnych plików PDF w katalogu")
pdf_path = pdf_files[0]

print(f"Wczytuję plik PDF: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

llm = openai.LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")


class LessonAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, instructions="Twoje instrukcje dla agenta", **kwargs)
        self.docs = docs
        self.name = "Asystent Lekcji"

    async def process_message(self, message, context):
        #obsługa komunikacji
        response = await self.llm.chat(message)
        return response


async def entrypoint(ctx):
    await ctx.connect()
    session = AgentSession(
        llm=llm,
        tts=openai.TTS(),         # TTS od OpenAI
        vad=silero.VAD.load()     # Opcjonalnie VAD
    )
    agent = LessonAgent(llm=llm)
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Cześć! Jestem asystentem lekcji. O co chcesz mnie zapytać?")

if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
