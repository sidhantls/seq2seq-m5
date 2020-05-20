# seq2seq forecasting 
Contains versions of my ongoing experimental work with RNNs on the m5 competition. 

## Model 
* 2 layer encoder-decoder model with GRU units. 
* Uses embedding layers for categorical features
* Pipeline to incoporate local and global categorial conditioning
* Incoporates teacher forcing decay while training to help convergence and test performance 
* Context-Vector (output hidden of encoder network) doesnt capture seasonality-week over week effects 

## Observations 
* Incoporating all categorical features as input into the encoder leads to overfitting as opposed to using some amount of global conditioning
* Modeling seems to be challenging for RNN because of the sporadic count data mixed with continuous count series data  
* Training batch data is skewed towards low velocity items (sporadic and low magnitude sales)
* Model struggles with adjusting to different output scales for different items 
* Batch training of NNs helps tackle the constraint of internal memory and number of features seen with GBMs. 
* Method of stationary processing to subtract trendline from timeseries doesnt work as well because of intermittant magnitude of most items 

## To Try:
* Attension 
* Batch Sampling 
