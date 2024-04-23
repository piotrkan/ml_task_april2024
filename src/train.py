"""functionalities that are training-related (trainer, cross validation etc)"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm.notebook import tqdm
from . import utils
from . import data
from .models import ThermoNet

class ThermoNetTrainer:
    '''class for training and inference for neural network'''
    def __init__(self,
                 train_loader,
                 val_loader,
                 epochs=1,
                 lr=0.001,
                 runname='thermonet'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = ThermoNet().to(self.device)
        self.epochs = epochs
        self.lr = lr
        self.patience = 5
        self.best_val_loss = float('inf')
        self.train_loss = []
        self.val_loss = []
        self.runname = runname

    def _train_epoch(self, x, y_true, optimizer):

        criterion = nn.MSELoss()

        # Use autocast to handle data type casting
        with autocast(enabled=True):
            logits = self.model(x)
            loss = criterion(logits, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    def _validate_epoch(self, x, y_true):
        x = x.to(self.device)
        y_true = y_true.to(self.device)

        with torch.no_grad():
            y_pred = self.model(x)
            criterion = nn.MSELoss()
            loss = criterion(y_pred, y_true)

        return loss.item()


    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            input = x.to(self.device)
            output = self.model(input)
            return output
    
    def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr=float(self.lr))
        num_epochs_no_improvement = 0

        for epoch in range(1, self.epochs + 1):
            print(epoch)
            for phase in ['train','val']:
                if phase=='train':
                    for inputs, labels in tqdm(self.train_loader):
                        loss=[]
                        loss.append(self._train_epoch(inputs, labels, optimizer))
                    self.train_loss.append(np.mean(loss))
                    print('Train MSE ',self.train_loss[-1])
                else:
                    for inputs, labels in tqdm(self.val_loader):
                        loss=[]
                        loss.append(self._validate_epoch(inputs, labels))
                    self.val_loss.append(np.mean(loss))
                    print('Val MSE ',self.val_loss[-1])

            if self.val_loss[-1] < self.best_val_loss:
                self.best_val_loss = self.val_loss[-1]
                num_epochs_no_improvement = 0

                torch.save(self.model.state_dict(),
                           os.path.join('models',f"best_{self.runname}_ckpt.pth"))
            else:
                num_epochs_no_improvement += 1

            #early stopping
            if num_epochs_no_improvement >= self.patience:
                print(f"Early stopping after {epoch} epochs.")
                break  # Stop training
            torch.save(self.model.state_dict(), os.path.join('models', f"{self.runname}_ckpt.pth"))
            
        print(self.train_loss)
        print(self.val_loss)
        utils.plot_train_val_curve(self.train_loss, self.val_loss,self.runname)

def train_sklearn_with_cv(regressor,
                          input_data:pd.DataFrame,
                          target:np.array,
                          cv:int=5) -> tuple:
    """ function for cross validation for the sklearn regressors to check generelizability
    input
        regressor - regressor model for cv
        input_data - pd.DataFrame used for training, one-hot encoded
        target - target variable in np.array
        cv - cross validation
    out
        list of mean scores obtained for each validation, list of predictions for error analysis
    """
    kfold = KFold(cv, shuffle=True, random_state=42)
    total_score=[]
    predictions = []
    real_target = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(input_data)):
        print(fold)
    
        #fit & predict
        regressor.fit(input_data[train_ids], target[train_ids])
        outputs = regressor.predict(input_data[val_ids])
    
        #assess
        mse=mean_squared_error(target[val_ids], outputs)
        r2=r2_score(target[val_ids], outputs)
    
        print('MSE: ',mse)
        print('R2: ',r2)
        predictions.append(outputs)
        real_target.append(target[val_ids])
        total_score.append(mse)
    print(total_score)
    return predictions, real_target

def train_with_cv(trainer_class,
                  input_data:pd.DataFrame,
                  target:np.array,
                  cv:int=5,
                  lr:int=0.001,
                  no_epochs:int=5)-> list:
    """ function for cross validation for the neural nets to check generelizability
    input
        trainer class - trainer class for training & validation
        input_data - pd.DataFrame used for training, one-hot encoded
        target - target variable in np.array
        cv - cross validation
        no_epochs - no of epochs for training
    out
        list of mean scores obtained for each validation
    """
    #choose fold
    kfold = KFold(cv, shuffle=True, random_state=42)
    total_score=[]
    for fold, (train_ids, val_ids) in enumerate(kfold.split(input_data)):
        print(fold)
    
        #define datasets for the fold
        val_dataset = data.SequenceDataset(input_data[val_ids], target[val_ids])
        train_dataset = data.SequenceDataset(input_data[train_ids], target[train_ids])
        train_dataloader = DataLoader(train_dataset, batch_size=128)
        val_dataloader = DataLoader(val_dataset, batch_size=64)
    
        #define trainer and train
        trainer = trainer_class(train_dataloader, val_dataloader, epochs=no_epochs, lr=lr, runname=f'{fold}_net')
        trainer.train()
    
        #validate
        predictions = []
        for inputs, _ in val_dataloader:
            outputs = trainer.predict(inputs)
            predictions.extend(outputs.tolist())
        mse=mean_squared_error(target[val_ids], predictions)
        print(mse)
        total_score.append(mse)
    print(total_score)
    return total_score
