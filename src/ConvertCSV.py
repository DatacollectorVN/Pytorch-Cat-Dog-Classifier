import pandas as pd
import numpy as np
from PIL import Image
import os 

ROOT_DIR = 'Dog_Cat_dataset/train/train'
SAVE_CSV = 'Dog_Cat_dataset/data_after_convert.csv'

def main():
    obj = os.walk(ROOT_DIR)
    list_dir = list()
    for root, dirs, files in obj:
        for file in files:
            list_dir.append(os.path.join(root, file))
    
    print(f'len(list_dir) = {len(list_dir)}')
    label_0 = np.zeros(shape = 12500)
    label_1 = np.ones(shape = 12500)
    labels = np.concatenate([label_0, label_1], axis = 0)
    data = pd.DataFrame({"img_path" : list_dir, 
                         "labels" : labels})
    data.to_csv(SAVE_CSV)
    print('Done')

if __name__ == "__main__":
    data = pd.read_csv(SAVE_CSV, index_col = 0)
    print(data.iloc[0,1])
    img = Image.open(data.iloc[0,0]).convert('RGB')
    print(img.size)