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

# Useful material

- [Hybrid search with opensearch](https://opensearch.org/blog/hybrid-search/)
- [LLM as a judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge#:~:text=LLM%2Das%2Da%2DJudge%20is%20an%20evaluation%20method%20to,%2C%20Q%26A%20systems%2C%20or%20agents.) 
- [Query breakdown](https://haystack.deepset.ai/blog/query-decomposition)