# Giganci AI Agent
**Polskojęzyczny AI agent wspierający tutorów i wykładowców w szybkim przygotowaniu lekcji i seminariów na podstawie własnych konspektów lub PDF.**

Agent odpowiada na pytania, sugeruje kolejne zagadnienia i pomaga w tworzeniu spójnej narracji lekcji, przy zachowaniu pełnej kontroli nad danymi.

Struktura projektu umożliwia podmianę modelu LLM na lokalny lub offline przy minimalnej modyfikacji kodu, przy zachowaniu lokalnych embeddingów i pełnej kontroli nad danymi.


## Kluczowe cechy
* Integracja z LiveKit – szybka komunikacja głosowa i tekstowa.
* Asynchroniczna architektura (asyncio) – obsługuje wiele zapytań równocześnie.
* LangChain + lokalne embeddingi – pełna kontrola nad danymi PDF.
* PDF jako źródło wiedzy – agent odpowiada na podstawie treści dostarczonego dokumentu.
* Obsługa języka polskiego – naturalna konwersacja i proaktywne podpowiedzi.
* Tryb PoC (Proof of Concept) – lekka implementacja, łatwa integracja z innymi systemami.
* Możliwość integracji z własnym LLM offline – pełna niezależność od zewnętrznych usług (wymaga podmiany instancji LLM w kodzie).

## 🇵🇱 Język i komunikacja

Agent prowadzi **rozmowę w języku polskim**, rozumie kontekst i naturalne zwroty.
Pomaga w przygotowaniu się do lekcji, podsumowywaniu materiału i planowaniu zagadnień, zgodnie z treścią dostarczonego konspektu.

## Dlaczego to stworzyłem

Projekt powstał, aby umożliwić **tutorom i wykładowcom szybkie, skuteczne i praktyczne przygotowanie do lekcji lub seminariów.**
Agent pozwala pracować **bezpośrednio na własnych konspektach i dokumentach PDF**, przyspiesza przyswajanie materiału, pomaga w planowaniu zagadnień i formułowaniu pytań, a także wspiera tworzenie spójnej narracji lekcji lub prezentacji.
Celem jest **ułatwienie codziennej pracy edukacyjnej**, przy zachowaniu pełnej kontroli nad treścią i prywatnością danych.  

## Uruchomienie

1. Skopiuj `.env.example` do `.env` i uzupełnij swoje klucze.
2. Zainstaluj zależności:
`pip install -r requirements.txt`
3. Uruchom agenta:
`python agent.py`
4. Wejdź na [LiveKit Agents Playground](https://agents-playground.livekit.io) i połącz się z agentem.

## Prawa autorskie

Projekt stworzony wyłącznie na potrzeby procesu rekrutacyjnego.

Wszelkie prawa autorskie do kodu źródłowego i implementacji agenta AI pozostają przy autorze.  
Nie zezwalam na jego dalsze rozpowszechnianie ani wykorzystanie poza kontekstem procesu rekrutacyjnego bez mojej pisemnej zgody.
