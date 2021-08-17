from genericpath import exists
import os
import pdb

import torch
from torch.utils.data import Dataset

import yfinance as yf
import pandas as pd


company_tickers = "SPY AAPL MSFT AMZN TSLA GOOG NVDA ACN ADBE INTC AMD".split(" ")

class StockDataset(torch.utils.data.Dataset):

    def __init__(self, history_win, pred_win, pred_ticker, context_tickers, 
                        history_tform="totensor", pred_tform="totensor"):
        self.fname = "./data/stocks.csv"

        if not os.path.exists(self.fname):
            self.data = yf.download(company_tickers, period="20y", interval="1d", group_by="ticker")
            self.data = self.data.dropna()
            os.makedirs("./data", exist_ok=True)
            self.data.to_csv(self.fname)
        else:
            self.data = pd.read_csv(self.fname, header=[0,1], index_col=0)

        self.data = self.data.loc[:, (context_tickers+[pred_ticker], ["Close", "Adj Close"])]


        self.history_tform = history_tform
        self.pred_tform = pred_tform
        self.history_win = history_win
        self.pred_win = pred_win
        self.pred_ticker = pred_ticker
        self.context_tickers = context_tickers

    def __len__(self):
        return len(self.data) - self.history_win - self.pred_win

    def __getitem__(self, idx):
        features = self.data[idx:idx+self.history_win]

        future_val = self.data.loc[:, (self.pred_ticker, ["Close"])][idx+self.history_win:idx+self.history_win+self.pred_win]

        if self.history_tform == "totensor":
            features_tensor = []
            for col in features:
                features_tensor.append(torch.tensor(features[col])[:, None])
            
            features_tensor = torch.cat(features_tensor, dim=1)
            features = features_tensor.permute(1,0)

        if self.pred_tform == "totensor":
            future_val = torch.tensor(future_val.loc[:, (self.pred_ticker, "Close")])

        return features, future_val

    @staticmethod
    def create_splits(split_date='2021-01-01', **dset_args):
        train_dset = StockDataset(**dset_args)
        val_dset = StockDataset(**dset_args)

        train_dset.data = train_dset.data.loc[:split_date, :]
        val_dset.data = val_dset.data.loc[split_date:, :]

        return train_dset, val_dset


    def _normalize(self, mean=None, std=None):
        """
        normalize all columns separately
        """
        if mean is None:
            self.col_mean = []
            self.col_std = []
            compute_mean = True
        else:
            self.col_mean = mean
            self.col_std = std
            compute_mean = False

        for idx, col in enumerate(self.data):

            if compute_mean:
                self.col_mean.append(self.data[col].mean())
                self.col_std.append(self.data[col].std())
            
            self.data[col] = (self.data[col] - self.col_mean[idx]) / self.col_std[idx]
    
        return self.col_mean, self.col_std
        

def _unnormalize(self):
    """
    unnormalize all columns separately
    """
    for idx, col in enumerate(self.data):
        print(self.data[col] * self.col_max[idx])

def get_loaders(batch_size,**dset_args):
    train_dset, val_dset = StockDataset.create_splits(**dset_args)

    col_mean, col_std = train_dset._normalize()

    val_dset._normalize(col_mean, col_std)
    
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    # last 2 columns belong to the ticker to be predicted
    col_mean, col_std = col_mean[-2], col_std[-2]
    return train_loader, val_loader, col_mean, col_std

def test():
    
    dset = StockDataset(30, 5, "MSFT", "SPY AAPL AMZN GOOG NVDA".split(" "), pred_tform=None)

    hist, trg = dset[0]
    print(hist, trg)
    pdb.set_trace()

if __name__ == "__main__":
    test()