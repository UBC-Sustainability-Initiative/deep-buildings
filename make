data/processed/processed.pkl: 
	python3 src/data/preprocess.py "data/interim/CIRS_data_joined_LSTM.pkl" "data/processed/processed.pkl" $@

clean:
	rm -f data/processed/*.pkl

models/LSTM/lstm.model: data/processed/processed.pkl
   python3 src/modeleling/run.py $< $@