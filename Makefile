.PHONY: test lint format build clean

test:
	python -m pytest tests/ -m "not integration" -v

test-all:
	python -m pytest tests/ -v

lint:
	python -m ruff check kognios/ tests/

format:
	python -m ruff format kognios/ tests/

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/ __pycache__/
