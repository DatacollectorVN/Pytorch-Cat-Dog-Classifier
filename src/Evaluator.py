import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.CustomDataset import CatAndDogDatasetTest, DatasetTransform
from src.Model import ResNet
import os


class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])


class EvalTest(SettingConfig):
    ''' Evaluate testset (dataset without label)
    '''

    def __init__(self, **args):
        super(EvalTest, self).__init__(**args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def transform(self):
        transformer = DatasetTransform(resize = (self.SIZE, self.SIZE))
        
        return transformer

    def load_data(self):
        transformer = self.transform()
        test_dataset = CatAndDogDatasetTest(base_dir = self.TEST_IMAGE_DIR, 
                                            transform = transformer)

        return test_dataset

    def get_testloader(self):
        test_dataset = self.load_data()
        test_loader = DataLoader(test_dataset, batch_size = self.BATCH_SIZE, 
                                 shuffle = 0, num_workers = self.NUM_WORKERS)

        return test_loader
    def load_model(self):
        model = ResNet(is_trained = False).to(self.device)

        # load checkpoint:
        checkpoint = torch.load(self.CHECKPOINT_DIR, map_location = self.device)

        # model load state
        #load_state_dict(chech_point['state'])
        model.load_state_dict(checkpoint['state'])

        return model
    
    def predict(self, save_small = False):
        test_loader = self.get_testloader()
        model = self.load_model()
        files_lst = list()
        pred_proba_lst = list()
        for step, (images, files) in enumerate(test_loader):
            images = images.to(self.device)
            
            pred_proba = model(images) # pred_proba with shape (BATCH_SIZE, 1)
            pred_proba = pred_proba.squeeze(1) # (BATCH_SIZE, )

            # convert to list 
            pred_proba = pred_proba.cpu().detach().tolist()
            pred_proba = [round(x, 3) for x in pred_proba]
            pred_proba_lst.extend(pred_proba)
            files_lst.extend(files)

            if save_small:
                if step == 3:
                    break
        
        return files_lst, pred_proba_lst
    
    def save_predicts(self, save_small = False):
        file_lst, pred_proba_lst = self.predict(save_small)
        indices = [i for i in range(len(file_lst))]
        os.makedirs("predicts", exist_ok = True)
        df = pd.DataFrame({"Index" : indices,
                           "ID_image": file_lst, 
                           "pred_proba" : pred_proba_lst})
        df.to_csv(self.SAVE_PREDICTS_CSV, index = False)

        print('DONE')


