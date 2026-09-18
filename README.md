# RAG-b1
## Overview

This project aims to build a Retrieval-Augmented Generation (RAG) system capable of answering questions about Warhammer 40K using two different knowledge sources:

- the Warhammer 40K core rulebook in PDF format;
- a SQLite database containing unit datasheets.

The system should be able to retrieve information from either source independently and combine information from both sources when necessary.

## Objectives

The main objectives are to:
- parse and process the PDF rulebook;
- split document content into meaningful chunks;
- build a retriever for the rulebook;
- build a retriever for the SQLite database;
- route user questions to the appropriate source;
- generate answers using retrieved context and an LLM;
- evaluate the quality of the RAG pipeline;
- monitor basic system behaviour and performance;
- provide a simple user interface.


## Project Structure
```text
RAG-b1/
│
├── data/
│   ├── pdf/
│   │   └── Core Rules.pdf
│   └── sqlite/
│       └── wahadb.sqlite
│
├── ingestion/
│   ├── pdf_parser.py
│   └── chunker.py
│
├── retrieval/
│   ├── text_retriever.py
│   ├── sql_retriever.py
│   └── reranker.py
│
├── rag/
│   ├── router.py
│   └── pipline.py
│
├── llm/
│   └── client.py
│
├── eval/
│
├── monitor/
│
├── tests/
│ 
├── docs/
│   ├── journal.md. ----> change_log
│   └── plan.md
│
└── README.md

