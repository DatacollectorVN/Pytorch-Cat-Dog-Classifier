import numpy as np 
from src.Evaluator import EvalTest
import json
from pathlib import Path
import os

FILE_TRAIN_CONFIG = os.path.join('config', 'InferConfig.json')
f = open(FILE_TRAIN_CONFIG)
params = json.load(f)
tester = EvalTest(**params)
def main():
    tester.save_predicts(save_small = True)

if __name__ == "__main__":
    main()