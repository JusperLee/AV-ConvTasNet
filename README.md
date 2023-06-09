# Unofficial Time Domain Audio Visual Speech Separation Implementation

> This repository contains the unofficial implementation of the Time Domain Audio Visual Speech Separation (AV-ConvTasNet) algorithm. This project aims to provide an open-source, easy-to-understand implementation that is accessible to researchers, engineers, and enthusiasts.

# Table of Contents
- Introduction
- Installation
- Usage
- Training
- Evaluation
- Contribute
- License


# Introduction
This paper proposes a new time-domain audio-visual architecture for target speaker extraction from monaural mixtures, utilizing audio-visual multi-modal modeling. The proposed structure is compared with three typical separation models, uPIT (frequency-domain audio-only), Conv-TasNet (time-domain audio-only), and Conv-FavsNet (frequency-domain audio-visual), and is shown to bring significant improvements compared to all other baseline models. This work is the first to perform audio-visual separation directly on the time-domain, and highlights that previous visual features are not well designed for speech separation.

This implementation is based on the following research paper:

```bibtex
@inproceedings{wu2019time,
  title={Time domain audio visual speech separation},
  author={Wu, Jian and Xu, Yong and Zhang, Shi-Xiong and Chen, Lian-Wu and Yu, Meng and Xie, Lei and Yu, Dong},
  booktitle={2019 IEEE automatic speech recognition and understanding workshop (ASRU)},
  pages={667--673},
  year={2019},
  organization={IEEE}
}
```

# Installation

To install the required dependencies for this project, follow the instructions below:

1. Clone this repository:
```bash
git clone https://github.com/JusperLee/AV-ConvTasNet.git
cd AV-ConvTasNet
```

2. Create and activate a virtual environment:
```bash
conda env create -f env.yml
conda activate nichang
```

# Usage
## Data Preparation

Before you can train or evaluate the model, you'll need to prepare the data. Download the dataset (e.g., LRS2, LRS3 and Voxceleb2) and extract it into the Data/ directory. Follow the instructions in the Data/ to preprocess and generate the necessary files.

The generated datasets (LRS2-2Mix, LRS3-2Mix, and VoxCeleb2-2Mix) can be downloaded at the links below.

| Datasets |  Links  | Pretrained Models  |
| ------------ | ------------ |------------ |
| LRS2-2Mix  | [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi**  | [Google Driver](https://drive.google.com/file/d/1WtcpYYr8nMiIpJ1epnuGNk2DtiacUXDf/view?usp=sharing)|
| LRS3-2Mix  |  [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi** |[Google Driver](https://drive.google.com/file/d/19OqNxY1jaO8USM-KnAIJ9B0Rh_Uwh5Ji/view?usp=sharing)|
| VoxCeleb2-2Mix |  [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi** |[Google Driver](https://drive.google.com/file/d/1jFHC6R51tpqyUd81LYM8Tg45NbbjWkwV/view?usp=sharing) |

# Training
To train the AV-ConvTasNet model, use the following command:

```bash
cd Trainer
python train.py --opt config/train.yml
```

You can customize the training parameters by modifying the train.yaml file or creating your own configuration file.

# Evaluation
To evaluate the performance of a trained TDAVSS model, use the following command:

```bash
cd Test
python evaluate.py config/train.yml model_path save_path
```

You can customize the evaluation parameters by modifying the eval_config.yaml file or creating your own configuration file.

# Contribute
Contributions are welcome! 

# Thanks
- [Jian Wu](https://scholar.google.com/citations?user=AGAyldkAAAAJ&hl=en) (Provide ideas for model implementation)
- [Asteroid](https://github.com/asteroid-team/asteroid) (Loss fuction and training pipline)

# License
This project is released under the [Apache-2.0 license](https://github.com/JusperLee/AV-ConvTasNet/blob/main/LICENSE).
