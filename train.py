import torch
import data_utils
import numpy as np
import pandas as pd
import pdb


class stock_CNN(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool1d(2, 2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.AvgPool1d(3, 3),
            torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
        )
    
    def forward(self, inp):
        features = self.conv_layers(inp)
        return torch.squeeze(features)
    
    def load_wt(self, weights):
        self.conv_layers.load_state_dict(torch.load(weights))
    

def get_loss(model, hist, trg, loss_func, device):
    hist = hist.to(device).float()
    trg = trg.to(device).float()

    pred = model(hist)

    loss = loss_func(pred, trg)

    return loss, pred

def main():
    device="cuda"
    batch_size = 64

    dset_args = {"history_win":30, 
            "pred_win":5, 
            "pred_ticker":"GOOG", 
            "context_tickers":[]}
            # "SPY AAPL AMZN GOOG NVDA".split(" ")}


    train_loader, val_loader, col_mean, col_std = data_utils.get_loaders(batch_size=batch_size, **dset_args)

    model = stock_CNN(len(dset_args["context_tickers"])*2+2).to(device)
    

    loss_func = torch.nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print_interval=30

    num_epochs = 30
    best_val_loss = np.inf
    for epoch in range(num_epochs):

        for idx, (train_hist, train_trg) in enumerate(train_loader):
            opt.zero_grad()
            

            loss, _ = get_loss(model, train_hist, train_trg, loss_func, device)
            loss.backward()
            opt.step()

            if idx % print_interval == 0:
                print(f"iter:{idx}::=::train loss:{loss.item()}")
        
        val_loss = 0
        with torch.no_grad():
            for idx, (val_hist, val_trg) in enumerate(val_loader): 
                
                loss, pred = get_loss(model, val_hist, val_trg, loss_func, device)

                val_loss += loss.item()
                
            val_loss /= idx
            print(f"val loss:{val_loss}")
        
            if val_loss < best_val_loss:
                torch.save(model.conv_layers.state_dict(), f"./weights/{dset_args['pred_ticker']}_model_wt.pt")
                best_val_loss = val_loss

    

if __name__ == "__main__":
    main()

