MODEL := model.pkl
DATA := data/spam.csv

.PHONY: train test clean

$(MODEL): train.py $(DATA)
	python train.py

train: $(MODEL)
	@echo "Training finished."

test:
	python -m pytest -q

clean:
	@if exist $(MODEL) del /q $(MODEL)

