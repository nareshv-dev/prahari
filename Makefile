.PHONY: data train evaluate serve test all
data:
	python data/download.py
train:
	python -m src.train
evaluate:
	python -m src.train
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000
test:
	python -m pytest -q
all: train test
