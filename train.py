import numpy as np 
from src.Trainer import CatDogTrainer
import json
from pathlib import Path
import os

FILE_TRAIN_CONFIG = os.path.join('config', 'TrainConfigSmall.json')
f = open(FILE_TRAIN_CONFIG)
params = json.load(f)
model = CatDogTrainer(**params)
def main():
    model.train()

if __name__ == "__main__":
    main()