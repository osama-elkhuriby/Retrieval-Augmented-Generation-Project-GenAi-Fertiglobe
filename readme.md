# Retrieval-Augmented-Generation Workshop GenAi-Fertiglobe
A workshop form Fertiglobe


## Installations

### Download Miniconda form https://www.anaconda.com/download

### Setup the environment
```bash
$ conda create --name myenv python=3.11
$ conda activate myenv
```

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