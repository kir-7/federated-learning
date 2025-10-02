from cluster import FedCluster, FedClusterConfig
from visualize import visualize_clustering


config = FedClusterConfig(
    n_clients=10,
    n_clusters=3,
    n_classes=10,
    partition_column='label',
    client_lr=1e-3,
    dataset='mnist',
    method='dirichlet',
    dirichlet_alpha=1.0,
    m=0.5,
    global_rounds=10,
    client_bs=32,
    global_bs=32,
    client_epochs=5,

    use_agglomarative=True,
)

component = FedCluster(config)

fig = visualize_clustering([component], method='pca')
fig.show()