# RAG-based Question Answering su Materiale Universitario

Progetto per il corso di Deep Learning Applications.

L’obiettivo del progetto è sviluppare un sistema basato su Retrieval-Augmented Generation (RAG) per il Question Answering su materiale universitario, come slide, dispense e appunti in formato PDF/TXT.

Il sistema permette all’utente di caricare documenti e porre domande in linguaggio naturale, generando risposte contestualizzate tramite retrieval semantico e Large Language Models open-source eseguiti localmente.

Il progetto contiene anche un report in markdown dove viene spiegato più approfonditamente.

---

# Features

Il sistema è in grado di:

- caricare documenti PDF/TXT;
- preprocessare e suddividere i documenti in chunk;
- generare embeddings vettoriali;
- salvare gli embeddings in un vector database;
- recuperare i contenuti più rilevanti rispetto a una query;
- generare risposte contestualizzate;
- confrontare una pipeline baseline e una pipeline migliorata;
- visualizzare i chunk recuperati;
- mostrare similarity scores;
- eseguire una valutazione sperimentale.

---

# Stack tecnologico

Il progetto utilizza:

- Python
- LlamaIndex
- ChromaDB
- Ollama
- Llama 3
- HuggingFace Embeddings
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

# Architettura della pipeline RAG

La pipeline implementata segue questi step:

1. document ingestion;
2. chunking;
3. embedding generation;
4. vector storage;
5. retrieval;
6. answer generation.

---

# Versioni del sistema

Il progetto confronta due pipeline differenti.

---

## Baseline RAG

Pipeline standard basata su:

- fixed-size chunking;
- chunk overlap;
- retrieval standard;
- top_k = 3.

Configurazione principale:

- chunk_size = 350
- overlap = 70

---

## Improved RAG

Pipeline migliorata basata su:

- semantic chunking;
- retrieval ottimizzato;
- top_k = 2.

La pipeline improved utilizza:

- SemanticSplitterNodeParser

per preservare meglio la continuità semantica tra slide consecutive.

---

# Modelli utilizzati

## Large Language Model (LLM)

Per la generazione delle risposte viene utilizzato un modello open-source eseguito localmente tramite Ollama.

Modello utilizzato:

- llama3

---

## Embedding Model

Per la generazione degli embeddings viene utilizzato un modello HuggingFace locale.

Modello utilizzato:

- BAAI/bge-small-en-v1.5

---

# Esecuzione locale

L’intera pipeline RAG viene eseguita localmente:

- il modello LLM gira tramite Ollama;
- gli embeddings vengono generati localmente;
- il vector database viene salvato in locale con ChromaDB.

Questo approccio permette:

- nessun costo API;
- maggiore privacy dei documenti;
- utilizzo di modelli open-source;
- funzionamento offline.

---

# Demo Streamlit

La demo Streamlit permette di:

- selezionare la pipeline;
- confrontare baseline e improved RAG;
- visualizzare le risposte generate;
- osservare i chunk recuperati;
- analizzare i similarity scores;
- confrontare retrieval e generation side-by-side.

---

# Valutazione sperimentale

La valutazione viene effettuata tramite un benchmark di domande sui documenti caricati.

Aspetti valutati:

- accuratezza delle risposte;
- qualità del retrieval;
- coerenza semantica;
- riduzione delle hallucinations;
- confronto tra baseline e improved RAG.

---

# Risultati

| Pipeline | Chunking Strategy | Score medio |
|---|---|---|
| Baseline RAG | Fixed-size chunking | 0.536 |
| Improved RAG | Semantic chunking | 0.583 |

La pipeline improved ottiene risultati mediamente migliori rispetto alla baseline.

---

# Struttura del progetto
```
project-DLA-RAG/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
├── images/
│
├── src/
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── pipeline.py
│   ├── retrieval.py
│   └── vector_store.py
│
├── app.py
├── main_base.py
├── main_improved.py
├── run_evaluation.py
│
├── requirements.txt
├── README.md
└── report.md
```
---

# Installazione

## 1. Clonare il repository

```git clone <repository_url> cd project-DLA-RAG ```

---

## 2. Creare ambiente virtuale

```python -m venv .venv ```

Attivazione ambiente virtuale:

### macOS/Linux

```source .venv/bin/activate ```

### Windows

```.venv\Scripts\activate ```

---

## 3. Installare le dipendenze

```pip install -r requirements.txt ```

---

## 4. Installare Ollama

https://ollama.com/

---

## 5. Scaricare il modello Llama3

```ollama pull llama3 ```

---

# Utilizzo del progetto

Inserire i documenti PDF/TXT nella cartella:
```
data/raw/ 
```
---

## Installare requirements

``` pip install -r requirements.txt ```

---

## Avvio pipeline baseline

```python main_base.py ```

---

## Avvio pipeline improved

```python main_improved.py ```

---

## Avvio demo Streamlit

```streamlit run app.py ```

---

## Esecuzione evaluation

```python run_evaluation.py ```

---

# Esempio di utilizzo

Domanda:
```
What is a Convolutional Neural Network? 
```
Il sistema:

1. recupera i chunk più rilevanti;
2. genera una risposta basata sul contesto;
3. mostra i chunk recuperati;
4. mostra i similarity scores.

---

# Possibili sviluppi futuri

- hybrid retrieval;
- reranking;
- supporto multimodale;
- retrieval su immagini;
- conversational memory;
- evaluation avanzata;
- dataset più estesi.

---

# Repository

```
https://github.com/annasetzu/project-DLA-RAG.git
```

