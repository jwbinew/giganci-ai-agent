# Giganci AI Agent

**Polskojęzyczny AI agent wspierający tutorów i wykładowców w przygotowaniu się do lekcji i seminariów.**  

Agent korzysta z konspektów lub plików PDF **dostarczonych przez użytkownika**, odpowiada na pytania i sugeruje kolejne zagadnienia, pomagając prelegentowi uporządkować materiał i przygotować spójną narrację.

## Najważniejsze cechy
* Integracja z LiveKit – szybka komunikacja głosowa i tekstowa.
* Asynchroniczna architektura (asyncio) – obsługuje wiele zapytań równocześnie.
* LangChain + lokalne embeddingi – pełna kontrola nad danymi PDF.
* PDF jako źródło wiedzy – agent odpowiada na podstawie treści dostarczonego dokumentu.
* Obsługa języka polskiego – naturalna konwersacja i proaktywne podpowiedzi.
* Tryb PoC (Proof of Concept) – lekka implementacja, łatwa integracja z innymi systemami.
* Możliwość integracji z własnym LLM offline – pełna niezależność od zewnętrznych usług (wymaga podmiany instancji LLM w kodzie).

## Język i komunikacja

Agent prowadzi **rozmowę w języku polskim**, rozumie kontekst i naturalne zwroty.
Pomaga w przygotowaniu się do lekcji, podsumowywaniu materiału i planowaniu zagadnień, zgodnie z treścią dostarczonego konspektu.

## Prywatność i lokalność danych

- **Warstwa wektorowa (embeddingi) działa lokalnie**, dzięki czemu materiały PDF pozostają w pełni pod kontrolą użytkownika.  
- **Użycie zewnętrznego LLM (np. OpenAI)** wymaga przesłania promptów do chmury, co oznacza ograniczoną prywatność.  
- **Pełna prywatność i kontrola** jest zachowana w przypadku podmiany modelu LLM na lokalny, działający offline.  

## Dlaczego to stworzyłem

Projekt powstał, aby umożliwić **tutorom i wykładowcom szybkie, skuteczne i praktyczne przygotowanie do lekcji lub seminariów.**

Agent pozwala pracować **bezpośrednio na własnych konspektach i dokumentach PDF**, przyspiesza przyswajanie materiału, pomaga w planowaniu zagadnień i formułowaniu pytań, a także wspiera tworzenie spójnej narracji lekcji lub prezentacji.

Celem jest **ułatwienie codziennej pracy edukacyjnej**, przy zachowaniu pełnej kontroli nad treścią i prywatnością danych.  

## Uruchomienie

```bash
# Skopiuj plik .env.example do .env i uzupełnij klucze
cp .env.example .env

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom agenta
python agent.py

```

Po uruchomieniu połącz się z agentem na [LiveKit Agents Playground](https://agents-playground.livekit.io).

## Prawa autorskie

Projekt stworzony wyłącznie na potrzeby procesu rekrutacyjnego.  
Wszelkie prawa do kodu źródłowego i implementacji agenta AI pozostają przy autorze.  
Projekt nie może być wykorzystywany ani rozpowszechniany poza kontekstem rekrutacji bez pisemnej zgody autora.

