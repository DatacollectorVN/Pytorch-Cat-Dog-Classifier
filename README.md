# Pytorch-Cat-Dog-Classifier
This repository is the practice room with goals:
+ Use pytorch with custom pretrained model
+ Organize code structure 

## INSTALLATION 
1. Create virtual environment
```bash
conda create -n CatDogClassifier python=3.8
conda activate CatDogClassifier
```
2. Clone this repository 
3. Install required packages 
```bash 
pip install -r requirements.txt
```
4. In the repository, execute `bash setup_dataset.sh` for create folder and download small dataset.
5. In the repository, execute `bash download_model.sh` for download model in to folder ./saved_model/best_model.pth

## FOR TRAINING
```bash
python train.py
```