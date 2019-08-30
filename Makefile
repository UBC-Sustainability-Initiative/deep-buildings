data/processed/processed.pkl: 
	python3 src/data/preprocess.py "data/interim/CIRS_data_joined_LSTM.pkl" $@

data/processed/processed4clf.pkl: 
	python3 src/data/preprocess.py "data/interim/CIRS_interim.pkl" $@



models/LSTM/lstm.model: data/processed/processed.pkl
	python3 src/modeling/train_lstm.py --n_epochs 5 $< $@
