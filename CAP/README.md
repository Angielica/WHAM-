
This is the accompanying code for the paper **CAP: A Prompt Generation Mechanisms for Detecting Unauthorized Data Usage in Generative Models**

The repository contains all code to re-execute our models.

The repository contains the following folders:
- **config**, contains experiment configurations for each dataset and for each parameter configuration;
- **data**, contains the dataset
- **dataloader**, contains the code to load dataset;
- **models**, with the source code of the models;
- **utility**, with utility code.

**main.py** contains the code to run experiments. You have to pass the experiment configuration:

'''
python3 main.py config/{config_name}.json
'''
where config_name is the name of the experimental configuration file.

main.py module imports the needed libraries and set the environment variable to activate the GPU device

In the configuration, you need to set all the parameters needed for the experiments. Specifically:
- **dataset_name** : the name of the dataset;
- **dataset_path**: the path where the dataset is stored;
- **seq_len**: the sequence length;
- **seed**: the seed for replication;
- **perc_split_m**: the percentage of data used for the training (D_train_M) and validation set (D_val_M) used for training and evaluating the model \Theta 
- **split_train**: the percentage of data extracted from D_train_M for creating the D_{+} subset (used together with D_{-} for training the model \Phi)
- **split_val**: the percentage of data extracted from D_val_M for creating D_{-}
- **n_clusters**: the number of clusters
- embed_dim_m: the dimension of the embedding for model \Theta
- num_heads_m: the number of heads for model \Theta
- num_layers_m: the number of layers for model \Theta
- embed_dim_g: the dimension of the embedding for model \Phi
- num_heads_g: the number of heads for model \Phi
- num_layers_g: the number of layers for model \Phi
- SAVE_FOLDER: folder for saving the results
- batch_size: the batch size
- n_runs: the number of run
- lr_generator: the learning rate of \Theta
- lr_model: the learning rate of \Phi
- clip_grad : the clipping for the gradient
- n_epochs_m: the number of epochs for training \Theta
- n_epochs_g: the number of epochs for training \Phi
- train_m: =1 to train \Theta
- train_g: =1 to train \Phi
- test: =1 to make the inference
- compute_prediction: =1 to compute the predictions
- compute_elbow_metric: =1 to compute the elbow metric
- compute_stats_all: = 1 to compute the metric on the union of D_{+} and D_{-}
- compute_stats_sep: = 1 to compute the metric separately on the two sets (D_{+} and D_{-})
- compute_low_k: =1 to compute the low@k metric
- k: the number of the lowest predictions
- clustering: =1 to apply the clustering strategy for splitting
- n_gpu: = the number of the GPU