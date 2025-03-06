# Setup Instructions

## Python Environment Setup

1. Create a virtual environment and activate it by running:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

2. Install packages

```shell
uv sync
```

## Elasticsearch Setup with Docker

This project uses Elasticsearch through Docker Compose to facilitate document indexing and searching.

### Prerequisites

- Docker and Docker Compose installed on your system
- Curl (optional, to verify Elasticsearch status)

### Starting Elasticsearch

To start the Elasticsearch service, run the following command from the project root:

```shell
docker-compose up -d
```

The `-d` parameter runs the containers in the background.

### Verifying Elasticsearch Status

To check that Elasticsearch is working properly:

```shell
curl http://localhost:9200
```

You should receive a JSON response with information about the Elasticsearch instance.

### Stopping Elasticsearch

To stop the services:

```shell
docker-compose down
```

If you also want to remove the volumes (this will delete all indexed data):

```shell
docker-compose down -v
```

# Using the system

## Using the Indexer

Once Elasticsearch is running, you can run the indexing script:

```shell
python index_docs.py
```

## Using the Decomposed Search System

The decomposed search system breaks down complex questions into simpler sub-questions, processes each sub-question separately, and returns the results for each. This approach can improve the quality of answers for complex, multi-part questions.


```shell
python run_decomposed_search.py
```

# RESULTS

# Github repo with all code and documentation: 
- https://github.com/davidgil/enterprise-rag/ (main branch with basic implementation).  This basic implementation is a basic RAG without query breakdown.
- https://github.com/davidgil/enterprise-rag/tree/query-breakdown (branch with query breakdown implementation). This implementation is a basic RAG with query breakdown using LLM.

# Benchmark Dataset: 
- Data is based in Apple and Microsoft 10K filings in PDF
- Converted in markdown using docling
- Data is stored in /data folder
- Benchmark dataset is stored in benchmark/test-dataset.csv file (questions and answers crafted by ChatGPT)

# Codebase: 
- index_docs.py used to index documents in ElasticSearch DB
- run_decomposed_search.py used to run the inference phase 

# Results: 
- In benchmark/text-dataset.xlsx file you can find the results of the execution using the basic implementation. In summary, the RAG without query breakdown is able to answer 12/14 basic questions correctly (question about 1 company). With complex questions (involving several PDFs), the RAG is not able to answer correctly.
- In benchmark/text-dataset-query-breakdown.xlsx file you can find the results of the execution using the query breakdown implementation. With complex questions (involving several PDFs), the RAG is able to answer better than the basic implementation. Probably the PROMPT used for query breakdown can be improved.

# Local LLM: 
- All the tests have been done using a local LLM based in Microsoft Phi-4 hosted locally using LM Studio.



# Useful material

- [Hybrid search with opensearch](https://opensearch.org/blog/hybrid-search/)
- [LLM as a judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge#:~:text=LLM%2Das%2Da%2DJudge%20is%20an%20evaluation%20method%20to,%2C%20Q%26A%20systems%2C%20or%20agents.) 
- [Query breakdown](https://haystack.deepset.ai/blog/query-decomposition)