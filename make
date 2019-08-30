data/processed/processed.pkl: 
	python3 src/data/preprocess.py "data/interim/CIRS_data_joined_LSTM.pkl" $@

models/LSTM/lstm.model: data/processed/processed.pkl
	python3 src/modeling/train_lstm.py $< $@