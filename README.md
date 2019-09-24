# Deep learning for forecasting the impact of climate change on a naturally-ventilated building
This repository employs several different machine learning methods (LSTM, gradient boosting, Gaussian process regression and classification) to predict the future indoor thermal comfort performance of a naturally-ventilated test building under different future climate weather scenarios.

[Google CoLab notebook](https://colab.research.google.com/drive/1fEqECoHNKjTxuT4LC6wgYS1oVaH63IeE)

To-Do:

- Need to fix click arguments to ensure for any model, any list of parameters and manipulations can be passed (maybe it's better to provide the option to pass a .txt file with these specifications?)
- Need to fix 'H' returning error for CatBoost

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile commands
    ├── README.md          
    ├── data
    │   ├── interim        <- Data from that has been transformed, joined, etc.
    │   ├── output         <- Predictions from each model. 
    │   ├── processed      <- The final test/train data sets for used for each model.
    │   └── raw            <- Datafiles from variety of sources.  
    ├── figures            <- Final figures saved by user.
    ├── models             <- Trained models.  
    ├── src                <- Directory for processing and modeling code.
    │   ├── data           <- Contains preprocessing functions.
    │   ├── evaluation     <- For producing output plots and evaluating models.
    │   ├── modeling       <- Specific files for data processing and training each model.
    │   ├── visualization  <- Exploratory visualizations of the data.
    │   
    ├── notebooks          <- Jupyter notebooks for interacting with most of 'src'. Naming 
    │                         convention is a number (for ordering), the creator's initials, 
    │                         and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
                              
                       
