COM --  Embeddings, Random Walks, Link prediction
===============================================================================

About
-----

Code for [Efficient Link Prediction in Continuous-Time Dynamic Networks using Optimal Transmission and Metropolis Hastings Sampling]. COM is a novel continuous-time dynamic network link prediction model that integrates optimal transmission and Metropolis-Hastings sampling. Specifically, it leverages optimal transmission theory to compute the Wasserstein distance between a given node and its temporally efficient candidate neighbors, minimizing information loss during node information propagation. Additionally, the Metropolis-Hastings algorithm captures both the local structural characteristics and global spatial correlations of target links within the network.


## Installation

```python setup.py install```

## Requirements

  * Python  3.8

## Usage

```python link_prediction.py```

## Data 

All datasets are sourced from https://snap.stanford.edu/.

## Cite

If you find the code useful, please cite our paper:

```
@article{zhang2023efficient,
  title={Efficient Link Prediction in Continuous-Time Dynamic Networks using Optimal Transmission and Metropolis Hastings Sampling},
  author={Zhang, Ruizhi and Wei, Wei and Yang, Qiming and Shi, Zhenyu and Feng, Xiangnan and Zheng, Zhiming},
  journal={arXiv preprint arXiv:2309.04982},
  year={2023}
}```
