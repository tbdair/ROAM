
# ROAM: Random Layer Mixup for Semi-Supervised Learning in Medical Imaging
Medical image segmentation is one of the major challenges addressed by machine learning methods. Yet, deep learning methods profoundly depend on a huge amount of annotated data which is time-consuming and costly. Though semi-supervised learning methods approach this problem by leveraging an abundant amount of unlabeled data along with a small amount of labeled data in the training process. Recently, MixUp regularizer has been successfully introduced to semi-supervised learning methods showing superior performance. MixUp augments the model with new data points through linear interpolation of the data at the input space. In this paper, we argue that this option is limited, instead, we propose ROAM, a random layer mixup, which encourages the network to be less confident for interpolated data points at randomly selected space. Hence, avoids over-fitting and enhances the generalization ability. We validate our method on publicly available datasets on whole-brain image segmentation (MALC) achieving state-of-the-art results in fully supervised (89.8%) and semi-supervised (87.2%) settings with relative improvement up to 2.75% and 16.73%, respectively.
<br/> -- Link to the paper
[https://arxiv.org/pdf/2003.09439.pdf](https://arxiv.org/pdf/2003.09439.pdf) 

## Getting Started

This repository contains a PyTorch code for implementing and training ROAM. 
<br/> Also, this package provides trained models for MRI whole brain segmentation,  and CT COVID-19 lung segmentation. 
- [Whole Brain Segmentation](#Whole-Brain-Segmentation)
- [COVID-19 Segmentation](#COVID-19-Segmentation)


### Prerequisites

-Python <br/> 
-PyTorch

### Training 
- To train our model in supervised manner for whole brain segmentation, run the following code.
```
python train_ROAM_MALC_Supervised.py --data_path='your data path' --checkpoint_dir='path to save the trained model'
```
### Evaluation

### Prediction 


# Whole Brain Segmentation
In this package we opt for MRI a publicly available datasets; (i) [22], which
consists of 30 T1 MRI volumes; 15 volumes split into 3 labeled (∼500 slices),
9 unlabeled (∼1500 slices), 3 validation (∼500 slices), and 15 testing volumes
(∼2500 slices), (ii) IBSR [26], which consists of 18 T1 MRI volumes (∼2000
slices), and (iii) CANDI [19] which consists of 13 T1 MRI volumes (∼1500 slices).
# COVID-19 Segmentation

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


