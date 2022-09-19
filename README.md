# peer-x

## Pre-requisite
1. Install [poetry](https://python-poetry.org/docs/):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
2. Clone this repo and install dependencies using poetry:
```bash
git clone https://github.com/grmvvr/peer-x.git
cd peer-x
poetry install
```

## Run model training
Training a deep learning model.
```bash
poetry run python main.py
```

## Post processing
Post-processing outputs
```bash
poetry run python scripts/post_processing_pipeline.py
```

## Start TensorBoard server
```bash
poetry run tensorboard --logdir=./logs --port=8888
# Open in web browser: http://127.0.0.1:8888
```