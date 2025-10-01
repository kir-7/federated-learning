# Tasks

* __Kireeti__ : Implement other algorithms: FedAVG, FedSVRG, CO-OP.
* __Fariz__ : Implement  medical datasets related to vision (image classification is simple and good) modify existing functions for iid and non iid distribution of this dataset.    
* __Charitha__ : Implement CNN model for this dataset.
* __Kireeti__ : run the new datasets, models, algos and aggregate the results.


# FedSVRG, AVG, COOP

* all of these are implemented as 2 classes each Client and Server
* To use any of these algo, pass the model, dataset, n_clients, sample to the respective server class
* Model is pytorch model 
* dataset is a pytorch dataset, this is the entire dataset, not split for ach client
* n_clients is number of clients
* sample is a function that will take dataset as input and output data partitions for each client the output of this should be a dictionary with keys as client ids(0-n_clients-1) and values as the pytorch dataset for that particular client

* sample script to use FedAVG is there in fed_main.py file.

# TODO: 
* modify get_dataset function in utils.py to return data partitions in correct format
* update the models.py file with new models 
* finish the data sampling part.
    