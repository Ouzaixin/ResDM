# A Prior-information-guided Residual Diffusion Model for Multi-modal PET Synthesis from MRI

This repo contains the official Pytorch implementation of the paper: A Prior-information-guided Residual Diffusion Model for Multi-modal PET Synthesis from MRI

## Contents

1. [Summary of the Model](#1-summary-of-the-model)
2. [Setup instructions and dependancies](#2-setup-instructions-and-dependancies)
3. [Running the model](#3-running-the-model)
4. [Some results of the paper](#4-some-results-of-the-paper)
5. [Contact](#5-contact)
6. [License](#6-license)

## 1. Summary of the Model

The following figure shows the overview for our proposed model

<img src= image\framework.png>

We propose a novel unified model to simultaneously synthesize multi-modal PET images from MRI, to achieve low-cost and time-efficient joint multi-biomarker diagnosis of AD. Specifically, we incorporate residual learning into the diffusion model to emphasize inter-domain differences between PET and MRI, thereby forcing each modality to maximally reconstruct its modality-specific details. Furthermore, we leverage prior information, such as age and gender, to guide the diffusion model in synthesizing PET images with semantic consistency, enhancing their diagnostic value. Additionally, we develop an intra-domain difference loss to ensure that the intra-domain differences among synthesized PET images closely match those among real PET images, promoting more accurate synthesis.

## 2. Setup instructions and dependancies

For training/testing the model, you must first download ADNI dataset. You can download the dataset [here](https://adni.loni.usc.edu/data-samples/access-data/). Also for storing the results of the validation/testing datasets, checkpoints and loss logs, the directory structure must in the following way:

    ├── data                # Follow the way the dataset has been placed here
    │   ├── whole_Abeta       # Here Abeta-PET images must be placed
    │   ├── whole_Tau          # Here Tau-PET images must be placed
    │   └── whole_MRI          # Here MR images must be placed
    ├── data_info          # Follow the way the data info has been placed here
    │   ├── data_info.csv       # This file contains labels, age and gender information for each ID
    │   ├── train.txt           # This file contains IDs of training dataset, like '037S6046'
    │   └── validation.txt      # This file contains IDs of validation dataset
    │   └── test.txt            # This file contains IDs of test dataset
    ├── result             # Follow the way the result has been placed here
    │   ├── exp_1              # for experiment 1
    │   │   └── CHECKPOINT_Unet.pth.tar     # This file is the trained checkpoint for Unet
    │   │   └── CHECKPOINT_encoder.pth.tar     # This file is the trained checkpoint for text_encoder
    │   │   └── loss_curve.csv              # This file is the loss curve
    │   │   └── validation.csv              # This file is the indicator files in the validation set
    │   │   └── test.csv                    # This file is the indicator files in the test set
    ├── config.py          # This is the configuration file, containing some hyperparameters
    ├── dataset.py         # This is the dataset file used to preprocess and load data
    ├── main.py            # This is the main file used to train and test the proposed model
    ├── model.py           # This is the model file, containing two models (text_encoder and Unet)
    ├── README.md
    ├── utils.py           # This file stores the helper functions required for training

## 3. Running the model

Users can modify the setting in the config.py to specify the configurations for training/validation/testing. For training/validation/testing the our proposed model:

```
python main.py
```

## 4. Some results of the paper

Some of the results produced by our proposed model and competitive models are as follows. *For more such results, consider seeing the main paper and also its supplementary section*

<img src=image\result.png>

## 5. Contact

If you have found our research work helpful, please consider citing the original paper.

If you have any doubt regarding the codebase, you can open up an issue or mail at ouzx2022@shanghaitech.edu.cn

## 6. License

This repository is licensed under MIT license