# Boost Your Image Encoder with Radiologic Sentences for Interval Change of Chest Radiographs

This is a PyTorch implementation of the MICCAI under reivew paper:

## How to use

1. docker build
2. docker run
3. `pip install -r requirements.txt`  in your container

---

## Requirements

```
# --------- pytorch --------- #
# torch==1.10.
# torhvision==0.13.1+cu113
pytorch-lightning==1.8.6
torchmetrics==0.9.1
transformers==4.18.0
# ---- python packages ---- #
albumentations==1.1.0
pydicom
tqdm==4.64.1
tensorboard
pycocotools
pandas
setuptools==59.5.0
nltk
einops

# --------- hydra --------- #
hydra-core==1.1.0  # hydra-core is a dependency of hydra-cli, stable
hydra-colorlog==1.1.0
hydra-optuna-sweeper==1.1.0

# --------- loggers --------- #
wandb
tensorboard

# --------- linters --------- #
pre-commit      # hooks for applying linters on commit
black           # code formatting
isort           # import sorting
flake8          # code analysis
nbstripout      # remove output from jupyter notebooks

# --------- others --------- #
rich            # beautiful text formatting in terminal
pytest          # tests
sh              # for running bash commands in some tests
pudb            # debugger
jupyter
```

# Model architecture

![Figure_1](https://user-images.githubusercontent.com/108312461/228401129-9d34dd68-2eaa-4db3-b6cb-9a6e79483096.png)

# Model training

```
# NLP
python3 -u train.py +experiment=FU.yaml

# Vision
python3 -u train.py +experiment=FU_vision.yaml

# MM
python3 -u train.py +experiment=FU_mm.yaml
```

The config management used for learning follows the rules in the repository above. (https://github.com/ashleve/lightning-hydra-template)

# Result

![CAM_fig](https://user-images.githubusercontent.com/108312461/228400744-9a51ef53-0e56-403f-97f5-5f2ed2becfa9.png)


## Contact

![https://user-images.githubusercontent.com/108312461/212851640-3e52332d-5346-4c1a-ab32-e337854afe71.png](https://user-images.githubusercontent.com/108312461/212851640-3e52332d-5346-4c1a-ab32-e337854afe71.png)


Page: [https://mi2rl.co](https://mi2rl.co/)

Email: [kjcho](mailto:kjcho@amc.seoul.kr).amc@gmail.com