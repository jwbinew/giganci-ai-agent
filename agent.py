import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from livekit.agents import Agent, AgentSession, WorkerOptions
from livekit.plugins import openai, silero
import asyncio

load_dotenv()

# Wczytanie pierwszego PDF
pdf_files = glob.glob("*.pdf")
if not pdf_files:
    raise FileNotFoundError("Nie znaleziono żadnych plików PDF w katalogu")
pdf_path = pdf_files[0]
print(f"Wczytuję plik PDF: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# Stworzenie streszczenia / kontekstu z dokumentu
# (tutaj prosta metoda: połącz teksty, daj agentowi do instrukcji)
material_text = " ".join([doc.page_content for doc in docs[:5]])  # np. pierwsze 5 chunków
base_instructions = (
    "Jesteś asystentem lekcji. Poniżej znajduje się konspekt i treść materiału do nauki:\n"
    f"{material_text}\n"
    "Odpowiadaj na pytania odnosząc się do materiału. "
    "Podpowiadaj, zadawaj kolejne pytania, by pomóc w nauce."
)

llm = openai.LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")


class LessonAgent(Agent):
    def __init__(self, *args, **kwargs):
        # Przekazujemy do agenta pełne instrukcje z konspektem
        super().__init__(*args, instructions=base_instructions, **kwargs)
        self.docs = docs
        self.name = "Asystent Lekcji"
        self.memory = []  # Pamięć rozmowy lub kontekst, można rozbudować

    async def process_message(self, message, context):
        # Dodajemy wiadomość użytkownika do pamięci, by kontekst zachować
        self.memory.append({"role": "user", "content": message})

        # Tworzymy prompt z pamięcią i instrukcjami
        conversation_prompt = base_instructions + "\n\n"
        for entry in self.memory:
            role = entry["role"]
            content = entry["content"]
            conversation_prompt += f"{role}: {content}\n"

        conversation_prompt += "assistant:"

        # Wywołanie LLM z pełnym promptem
        response = await self.llm.chat(conversation_prompt)

        # Dodajemy odpowiedź do pamięci
        self.memory.append({"role": "assistant", "content": response})

        # Proaktywne podpowiedzi do rozmowy
        hints = (
            "\n\nMożesz zapytać mnie o: plan zajęć, szczegóły materiału, "
            "ćwiczenia do wykonania lub inne kwestie związane z lekcją."
        )
        return response + hints


async def entrypoint(ctx):
    await ctx.connect()
    session = AgentSession(
        llm=llm,
        tts=openai.TTS(),  # TTS od OpenAI
        vad=silero.VAD.load()  # Opcjonalnie VAD
    )
    agent = LessonAgent(llm=llm)
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Cześć! Jestem asystentem lekcji. O co chcesz mnie zapytać?")


if __name__ == "__main__":
    from livekit.agents import cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
