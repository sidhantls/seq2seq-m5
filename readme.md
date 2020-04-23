# seq2seq forecasting (ongoing)
Contains versions of my ongoing experimental work on the m5 competition. Batch training of NNs will help tackle the constraint of number of features due to 
memory limits I've faced using LightGBM. 

## Model 
* 2 layer symmetric encoder-decoder model with GRU units. 
* Uses embedding layers for categorical features
* Incoporates regular and teacher-forcing training
