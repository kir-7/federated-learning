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

