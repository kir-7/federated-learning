# Overview

## Part - I : 
* The first part of the project consisted of implementing basic fed algos (FedAvg, FedSVRG, CooP) to test performance on basic datasets like MNIST, Cancer Dataset. 
* Performance across these datasets were noted and inferences about the importance of data distribution along with effects of class level skewness in data distribution were made.
* The implementation is roughly based around this [paper](https://dl.acm.org/doi/pdf/10.1145/3286490.3286559) 
* Code for this part can be found in ./groundwork/ folder, although it is not properly structured, but most of the results/algotihms are available in the .ipynb file


* Contributors - KK Kireeti, C Sai Charitha, Fariz Sameer

## Part - II :
* The second part of the project consists of exploring different cases of Non IIdness, different ways of creating a Non IID distribution (class label based, feature based) and comming up with algorithms to improve performance.

* Code can be found at ./fed_nidd/
* Contributors - KK Kireeti, Fariz Sameer, Sri Teja.

* Path is :-
    - For Class label based Non IID:
        - First implement clustering with K Means, and test out performance for both MNIST and Cifar10
        - K Means causes an issue where some outlier that is grouped with a similar clients, will bringdown the overall performance of the cluster leading to poor performnce, so wee need a clustering algo where outliers will be their own cluster, so it should be a distance based clustering, and should a distance threshold.
        - Testing with DBSCAN. 