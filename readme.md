# Retrieval-Augmented-Generation Workshop GenAi-Fertiglobe
A workshop form Fertiglobe


## Installations

### Install the required packages
```bash
$ pip install -r requirements.txt
```

### Setup the environments variables
```bash
$ cp .env.example .env
```
### Run Docker Compose Services
```bash
$ cd docker
$ cp .env.example .env
```

### Run FastAPI server
```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```