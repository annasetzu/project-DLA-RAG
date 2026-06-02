# RAG-based Question Answering su Materiale Universitario

Progetto per il corso di **Deep Learning Applications**.

L’obiettivo del progetto è sviluppare un sistema basato su **Retrieval-Augmented Generation (RAG)** per il question answering su materiale universitario, come slide, dispense e appunti in formato PDF/TXT.

Il sistema permette all’utente di caricare documenti e porre domande in linguaggio naturale, generando risposte contestualizzate attraverso una pipeline RAG.

---

# Obiettivi del progetto

Il sistema deve essere in grado di:

- caricare documenti PDF/TXT;
- preprocessare e suddividere i documenti in chunk;
- generare embeddings vettoriali;
- salvare gli embeddings in un vector database;
- recuperare i contenuti più rilevanti rispetto a una domanda;
- generare una risposta tramite un Large Language Model utilizzando il contesto recuperato.

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

---

# Architettura della pipeline RAG

La pipeline implementata segue questi step:

## 1. Document ingestion

Caricamento di documenti:

- PDF
- TXT

I documenti vengono letti ed estratti in formato testuale.

---

## 2. Chunking

Il testo viene suddiviso in chunk di dimensione controllata.

Parametri principali:

- chunk size
- chunk overlap

Questo permette di migliorare la qualità del retrieval.

---

## 3. Embedding generation

Ogni chunk viene trasformato in un embedding vettoriale utilizzando un modello HuggingFace locale.

Embedding model utilizzato:

- `BAAI/bge-small-en-v1.5`

---

## 4. Vector storage

Gli embeddings vengono salvati in:

- ChromaDB

Il vector database permette di effettuare ricerche semantiche sui documenti.

---

## 5. Retrieval

Data una domanda utente:

1. viene generato l’embedding della query;
2. vengono recuperati i chunk più rilevanti;
3. i chunk recuperati vengono usati come contesto per il modello LLM.

---

## 6. Answer generation

La risposta viene generata tramite un LLM locale eseguito con Ollama.

LLM utilizzato:

- `llama3`

---

# Modelli utilizzati

## Large Language Model (LLM)

Per la generazione delle risposte viene utilizzato un modello open-source eseguito localmente tramite Ollama.

Modello utilizzato:

- Llama 3

---

## Embedding Model

Per la generazione degli embeddings viene utilizzato un modello HuggingFace locale.

Modello utilizzato:

- `BAAI/bge-small-en-v1.5`

---

# Esecuzione locale

L’intera pipeline RAG viene eseguita localmente:

- il modello LLM gira tramite Ollama;
- gli embeddings vengono generati localmente;
- il vector database è salvato in locale con ChromaDB.

Questo approccio permette:

- nessun costo API;
- maggiore privacy dei documenti;
- utilizzo di modelli open-source;
- funzionamento offline.

---

# Versioni del sistema

Il progetto prevede il confronto tra:

## RAG base

Pipeline standard con:

- chunk size fisso;
- overlap fisso;
- retrieval standard;
- prompt base.

---

## RAG migliorato

Versione ottimizzata sperimentando:

- diverse dimensioni dei chunk;
- diversi overlap;
- differenti valori di top-k;
- prompt engineering;
- eventuale reranking.

---

# Valutazione

La valutazione verrà effettuata tramite un insieme di domande preparate sui documenti caricati.

Aspetti valutati:

- accuratezza delle risposte;
- qualità del retrieval;
- riduzione delle hallucinations;
- confronto tra RAG base e RAG migliorato.

---

# Struttura del progetto

```text
rag-university-qa/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evaluation.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Installazione

## 1. Clonare il repository
- `git clone <repository_url>`
- `cd rag-university-qa`

## 2. Creare ambiente virtuale
- `python -m venv .venv`
- `source .venv/bin/activate` (su Windows: `.venv\Scripts\activate`)

## 3. Installare le dipendenze
- `pip install -r requirements.txt`

## 4. Installare Ollama
https://ollama.com/

## 5. Scaricare il modello Llama3
- `ollama pull llama3`

---

# Avvio del progetto
Inserire i documenti PDF/TXT nella cartella: `data/raw/`

Avvio da terminale: `python main.py`

Avvio interfaccia Streamlit: `streamlit run app.py`

---

# Esempio di utilizzo
Domanda: Che cos'è una rete neurale convoluzionale?

Il sistema:

1. recupera i chunk più rilevanti;
2. genera una risposta basata sul contesto;
3. mostra anche i chunk recuperati.