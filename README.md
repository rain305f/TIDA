<h2 align="center"> <a href="https://openreview.net/forum?id=zrLxHYvIFL">Discover and Align Taxonomic Context Priors for Open-world Semi-Supervised Learning</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>
<h5 align="center">

[![arxiv](https://img.shields.io/badge/Arxiv-2403.09513-red)](https://openreview.net/forum?id=zrLxHYvIFL)
![Genarizalized Novel Class Discovery](https://img.shields.io/badge/Generalized-NovelClassDiscovery-yellow.svg?style=plastic)
![Open-set Learning](https://img.shields.io/badge/Openset-Learning-orange.svg?style=plastic)
![Semi-supervised Learning](https://img.shields.io/badge/Semi-supervised-Learning.svg?style=plastic)

 The official implementation of our paper "[Discover and Align Taxonomic Context Priors for Open-world Semi-Supervised Learning](https://openreview.net/forum?id=zrLxHYvIFL)", by *[Yu Wang](https://rain305f.github.io/), [Zhun Zhong](https://zhunzhong.site/), Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, Jie Chen.



## 📰 News
 We will release all source codes in three weeks.

| Date       | Event    |
|------------|----------|
| **2024/04/15** | 🔥 We have released our full training and inference codes.|
| **2023/11/03** | 🔥 Our TIDA is acccepted by NeurIPS 2023! |



## 💡 Abstract
Open-world Semi-Supervised Learning (OSSL) is a realistic and challenging task, aiming to classify unlabeled samples from both seen and novel classes using partially labeled samples from the seen classes. Previous works typically explore the relationship of samples as priors on the pre-defined single-granularity labels to help novel class recognition. In fact, classes follow a taxonomy and samples can be classified at multiple levels of granularity, which contains more underlying relationships for supervision. We thus argue that learning with single-granularity labels results in sub-optimal representation learning and inaccurate pseudo labels, especially with unknown classes. In this paper, we take the initiative to explore and propose a uniformed framework, called Taxonomic context prIors Discovering and Aligning (TIDA), which exploits the relationship of samples under various granularity. It allows us to discover multi-granularity semantic concepts as taxonomic context priors (i.e., sub-class, target-class, and super-class), and then collaboratively leverage them to enhance representation learning and improve the quality of pseudo labels. Specifically, TIDA comprises two components: i) A taxonomic context discovery module that constructs a set of hierarchical prototypes in the latent space to discover the underlying taxonomic context priors; ii) A taxonomic context-based prediction alignment module that enforces consistency across hierarchical predictions to build the reliable relationship between classes among various granularity and provide additions supervision. We demonstrate that these two components are mutually beneficial for an effective OSSL framework, which is theoretically explained from the perspective of the EM algorithm. Extensive experiments on seven commonly used datasets show that TIDA can significantly improve the performance and achieve a new state of the art. The source codes are publicly available at https://github.com/rain305f/TIDA.



##  🗝️ Training & Validating
```shell

# For CIFAR100 10% Labels and 50% Novel Classes 
python3 train_ours.py --dataset cifar100 --lbl-percent 10 --novel-percent 50 --arch resnet18 --num_protos 200 --num_concepts 20 --lr 0.4


For training on the other datasets, please download the dataset and put under the "name_of_the_dataset" folder and put the train and validation/test images under "train" and "test" folder. After that, please set the value of data_root argument as "name_of_the_dataset".

```

## 👍 Acknowledgement
* [Towards Realistic Semi-Supervised Learning](https://github.com/nayeemrizve/TRSSL) The main codebase we built upon and it is an wonderful open-set semi-supervised learning algorithm.



## Citation
```
@inproceedings{
wang2023discover,
title={Discover and Align Taxonomic Context Priors  for Open-world Semi-Supervised Learning},
author={Yu Wang and Zhun Zhong and Pengchong Qiao and Xuxin Cheng and Xiawu Zheng and Chang Liu and Nicu Sebe and Rongrong Ji and Jie Chen},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=zrLxHYvIFL}
}
```
