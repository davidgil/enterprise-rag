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

## Using the Indexer

Once Elasticsearch is running, you can run the indexing script:

```shell
python index_docs.py
```

### Common Troubleshooting

- **Connection Error**: Make sure Elasticsearch is running before executing the indexer.
- **Insufficient Memory**: If Docker reports memory issues, you can adjust the `ES_JAVA_OPTS` parameters in the `docker-compose.yml` file.




