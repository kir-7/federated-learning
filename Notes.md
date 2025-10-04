## Non IID Fed
 
 * We are trying to implement federated learning algorithms for Non IID Datasets through clustering
    - Few ways of clustering clients
        - via class skew non iid distribution        
        - letting models train and converge; cluster models that have similar paramters(converged to similar locations)
    - Non IID dataset can be from the class level skewness or from different features for same class level distribution. 

    - **Evaluation procedure:** For each of the client there will be a train/test set, which those clients will be evaluated on, and there is going to be a global test set consisting of all classes/labels; for each data point of the global set, infer all the cluster models, and pick the class that has highest probability across all clusters, this will be the global accuracy(as will be used during inference as well), apart from this each cluster will evaluate its model on the global set this is cluster accuracy and then each cluster will have a win rate which will be depict the percentage of times that cluster's output was selected for the global accuracy.

<br>

* ### Class Skew Non IID distribution
    - We create Non IID datasets through Dirichlet distribution by adjusting alpha (0=non iid; ∞=iid)
    - Use algorithms like KMeans, GMM, Agglomarative clustering to cluster clients based on the number of samples of each label a client has


    - #### Observations / Problems: 
        - We can divide the dataset to each client using the dirichlet distribution, we can get good degree of non IID distributions.
        - On MNIST dataset, it is n noted that the participation percentage(m) has a huge impact on the overall accuracy, as m = 1 means all clients participate and in ideal case this would be no different from centralized training. Using a lower m (< 0.5) significantly effects the overall performance. It is noted that, m=0.7 gave 92% accuracy in non IID case for base FedAVG and gave 94% by clusering. (not much difference, due to hyperparamters)
        
        - **One thing to test out is that let all clients be in their own cluster and check the overall accuracy** : leads to too many models (one per each client), performance on non IID mnist is slightly less than regular non IID setting in both base FedAVG and clustering.

        - In clustering, one problem that is faced is that if we fix the number of clusters, then some clusters will have clients that are tightly bounded, but some clusters will have clients that are enough far apart such that tey still belong to same cluster, but have different charecteristics so that that cluster's performance decreases.
        - Needs testing on lower values of m, different datasets and see if there will be a significant difference in performance between base FedAVG and clustering.
        
        - With Agglomarative clustering, and 60 clients,  m=0.4 the clustering method performed very well obtaining 95.2% gloabl accuracy; the corresponding base fedavg performance for 60 clients and m=0.4 achieved 96.5% as well.

        - **Hypothesis:** When clustering is done, the method inherently makes it so that the clusters will have high class imbalance issue, so when we global eval: for a particular cluster the data point that doesn't belong to this cluster's data distribution will cause issues, but this should be taken care of by the other cluster to which this data point belongs to. On the other hand, when base FedAVG is used since there is one globa model and it sees all the classes it might perform better.  
        **Emperical Verification:** To verify this hypothesis, need to first get the component for m=0,4; 10 clusters; 60 clients, and for data point of each client need to verify the probability distribution for each cluster model and compare and see the cluster probability and win rates. 

        - Another important thing is to test all these hypothesis on cifar10 dataset, as MNIST is a relatively simple dataset, it ois only for establishing the fact that the approach if applicable, not to test and consider the actual outputs

        - While Evaluating for Cifar10, it was very frustrating as in all initial experiements the accuracy never crossed 15%, The issue was in BatchNormalizaion, when using CNN models, it is very common to use Normalization layers, but these layers shoudn't be aggregated as these layer's parameters(running mean, var) depend highly on the data that is fed into the model and averaging these parameters will cause model collapse and after only few iterations model will start to predict only one class for all images, so while aggregating client parameters batch norm weights should be ignored. So **BatchNorm layers are poisonous to federated learning**, which is obvious in hindsight, as majority implementations of fed lear found online don't use CNNs.   

        - When experimenting on CIFAR10 dataset, base fed avg got 60% accuracy by 20 global epochs, and data distribution based clustering got similar performance. But when we evaluate different clusters performance and win rates, we notice that for some clusters the training doesn't seem to happen at all, and for cluster having only 1 client also doesn't seem to improve at all.


        - Partial Experiments can be found in __fed_niid/clustering/mnist__.


* ### Model Convergence based clustering:
    - **Hypothesis:** First need to establish that clients having similar data will converge to spatially similar locations.
