import torch
import sys
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.CustomDataset import CatAndDogDataset, DatasetTransform
from src.utils import EarlyStopping
from src.MetricEvaluation import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.Model import ResNet
from torch import nn, optim


class SettingConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])


class CatDogTrainer(SettingConfig):
    def __init__(self, **args):
        super(CatDogTrainer, self).__init__(**args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def feedforward(self, model, loss_fn, X, y):
        ''' feedforward the model and get probabilites and loss
        Args:
            models: (nn.Module) model to be used
            loss_fn: (nn.modules.loss) loss function to be used
            X: (torch.tensor) data used to train model
            y: (torch.tensor) label is used to calculate loss
        
        Output:
            probabilities: (torch.tensor) probabilities (from 0 to 1)
            lossses: (torch.tensor) total losses
        '''

        probabilities = model(X)
        probabilities = probabilities.squeeze(dim = 1)
        losses = loss_fn(probabilities, y)
        probabilities_metric = probabilities.cpu().detach().numpy()
        y_metric = y.cpu().detach().numpy()
        metric = Metric(probabilities_metric, y_metric, thresh = 0.5)
        acc = metric.acc()

        return probabilities, losses, acc

    def get_training_objects(self):
        '''Create object for training
        Outputs:
            model: (nn.Module) model for training 
            loss: (nn.module.loss) loss for training
            optimizer: (torch.optim) optimizer object 
            lr_scheduler: (torch.optim.lr_scheduler) learning rate scheduler
            early_stopping: (object) early stopping object
        '''

        model = ResNet(net_type='resnet50').to(self.device)
        loss_fn = nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr = self.LEARNING_RATE_START, momentum = 0.8)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor = self.LEARNING_RATE_SCHEDULE_FACTOR, 
                                         patience = self.LR_PATIENCE, mode = 'min', verbose = True)
        
        early_stopping = EarlyStopping(mode = 'max', patience = self.EARLY_STOPPING_PATIENCE)
        
        return model, loss_fn, optimizer, lr_scheduler, early_stopping
    
    def save_model(self, model, val_loss, optimizer):
        ''' saving model
        Args:
            model: (nn.Module) model to save
            val_loss: (float) loss on validation set
            optimizer: (torch.optim) optimizer object
        '''

        os.makedirs(self.SAVE_MODEL_DIR, exist_ok = True)
        model_path = self.SAVE_MODEL_PATH
        check_point = {'state' : model.state_dict(), 
                       'loss' : val_loss, 
                       'classes' : ['cat', 'dog'],
                       'optimizer' : optimizer.state_dict()}
        torch.save(check_point, model_path)
        
        return model_path
    
    def get_train_loader(self):
        ''' Build dataloader for train set
        '''

        train_transformer = DatasetTransform(resize = (self.IMAGE_SIZE, self.IMAGE_SIZE))
        train_dataset = CatAndDogDataset(self.DATA_DIR_TRAIN_LABEL, train_transformer, self.DATA_DIR_TRAIN_IMAGES)
        train_loader = DataLoader(train_dataset, batch_size = self.BATCH_SIZE, 
                                  shuffle = False, num_workers = self.NUM_WORKERS)
                
        return train_loader

    def train_val_split(self, shuffle=True):
        ''' Split train set into train and validation dataloader
        '''

        train_transformer = DatasetTransform(resize = (self.IMAGE_SIZE, self.IMAGE_SIZE))
        test_transformer = DatasetTransform(resize = (self.IMAGE_SIZE, self.IMAGE_SIZE))

        annotation_csv = pd.read_csv(self.DATA_DIR_TRAIN_LABEL, index_col = 0)
        num_train = annotation_csv.shape[0]
        indices = list(range(num_train))

        if shuffle:
            np.random.seed(self.SEED)
            # shuffle indices
            np.random.shuffle(indices) #Modify a sequence in-place by shuffling its contents.
        
        split = int(np.floor(self.VAL_RATIO * num_train)) # 20 % of num_train
        train_idx, val_idx = indices[split:], indices[:split]

        # SubsetRandomSampler. Samples elements randomly from a given list of indices, without replacement.
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx) 

        train_dataset = CatAndDogDataset(self.DATA_DIR_TRAIN_LABEL, train_transformer, self.DATA_DIR_TRAIN_IMAGES)
        val_dataset = CatAndDogDataset(self.DATA_DIR_TRAIN_LABEL, test_transformer, self.DATA_DIR_TRAIN_IMAGES)

        train_loader = DataLoader(train_dataset, batch_size = self.BATCH_SIZE, 
                                  shuffle = False, sampler = train_sampler, num_workers =self.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size = self.BATCH_SIZE, 
                                  shuffle = False, sampler = val_sampler, num_workers =self.NUM_WORKERS)

        return train_loader, val_loader
    
    def train(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Cuda is not available. Training on CPU ...")
        else:
            print("Cuda is available. Training on GPU ...")
        
        [model, loss_fn, optimizer, lr_scheduler, early_stopping] = self.get_training_objects()
        epochs = self.MAX_EPOCHS
        train_losses_lst, val_losses_lst = list(), list()
        best_loss = 999999
        train_loader, val_loader = self.train_val_split()

        print("Using Model as resnet50")

        for epoch in range(epochs):
            running_loss = 0
            running_acc = 0
            model.train() # mode training

            # for/else. https://book.pythontips.com/en/latest/for_-_else.html
            for step, (images, labels) in enumerate(train_loader):
                
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                probs, losses, acc = self.feedforward(model, loss_fn, images, labels)
                losses.backward()
                optimizer.step()
                running_loss += losses.item()
                running_acc += acc
                sys.stdout.write(f"\rEpoch {epoch + 1}/{epochs}... Training step {step + 1}/{len(train_loader)}... Loss {running_loss / (step + 1):.3f} with acc_train = {running_acc / (step+ 1):.3f}")
                
            else:
                val_running_loss = 0
                val_running_acc = 0
                model.eval() # mode evaluation

                with torch.no_grad():
                    for step, (images,labels) in enumerate(val_loader):
                        if train_on_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        
                        val_probs, val_losses, val_acc = self.feedforward(model, loss_fn, images, labels)
                        val_running_loss += val_losses
                        val_running_acc += val_acc
                        sys.stdout.write(f"\rEpoch {epoch+1}/{epochs}... Validating step {step + 1}/{len(val_loader)}... Loss {val_running_loss / (step + 1):.3f} with acc_val = {val_running_acc / (step+ 1):.3f}")

                train_losses_lst.append(running_loss / len(train_loader))
                new_val_loss = val_running_loss / len(val_loader)
                val_losses_lst.append(new_val_loss)

                if best_loss > new_val_loss:
                    print(f"Improve loss from {best_loss} to {new_val_loss}")
                    best_loss = new_val_loss
                    model_path = self.save_model(model, val_losses_lst, optimizer)