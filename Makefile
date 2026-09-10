.PHONY: install test run serve clean

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

run:
	python -m crm_intelligence.cli run --customers 1200 --seed 42

serve:
	python -m crm_intelligence.cli serve --port 8000

clean:
	python -c "from pathlib import Path; import shutil; p=Path('artifacts'); shutil.rmtree(p) if p.exists() else None"
