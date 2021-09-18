import sys
from torch.utils.data import DataLoader
from CustomDataset import CatAndDogDataset, DatasetTransform

ROOT_DIR = 'Dog_Cat_dataset/train/train'
CSV_FILE = 'Dog_Cat_dataset/data_after_convert.csv'
def main():
    transformer = DatasetTransform(resize = (224, 224))
    dataset = CatAndDogDataset(CSV_FILE, transformer)
    
    len_data = len(dataset)
    print(f'len_data = {len_data}')
    image, label = dataset[100]
    print(f'image = {image.size()}')
    print(f'label = {label}')
    train_loader = DataLoader(dataset, batch_size = 32, 
                                  shuffle = False, num_workers = 0)
    print(train_loader)

    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)
        print(labels[:5])
        sys.exit()

if __name__ == "__main__":
    main()
