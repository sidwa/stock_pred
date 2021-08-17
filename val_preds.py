import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdb

import train
import data_utils

def get_unnormed_val_predictions(val_dset, pred_win, pred_ticker, model, col_mean, col_std, device):

    val_target = pd.DataFrame([])
    val_preds = []

    with torch.no_grad():
        for idx in range(0, len(val_dset), pred_win):
            val_hist, val_trg = val_dset[idx]
            val_hist = val_hist[None, :, :]
        
            pred = model(val_hist.to(device).float())

            val_target = val_target.append(val_trg)
            val_preds.append(pred)
    
    val_preds = torch.cat(val_preds).reshape(-1).cpu()

    val_target = (val_target * col_std) + col_mean
    val_preds = (val_preds * col_std) + col_mean

    diff = (val_target[(pred_ticker, "Close")] - val_preds.numpy())

    val_target.insert(len(val_target.columns), "preds", val_preds.numpy())
    val_target.insert(len(val_target.columns), "diff", diff)

    return val_target    

def main():
    device="cuda"
    batch_size = 64

    dset_args = {"history_win":30, 
            "pred_win":5, 
            "pred_ticker":"GOOG", 
            "context_tickers": [],
            "pred_tform":None}
            # "SPY AAPL AMZN GOOG NVDA".split(" ")}


    # train_loader, val_loader, col_mean, col_std = data_utils.get_loaders(batch_size=batch_size, **dset_args)
    train_dset, val_dset = data_utils.StockDataset.create_splits(**dset_args)
    col_mean, col_std = train_dset._normalize()
    val_dset._normalize(col_mean, col_std)
    # last 2 columns belong to the ticker to be predicted
    col_mean, col_std = col_mean[-2], col_std[-2]

    model = train.stock_CNN(len(dset_args["context_tickers"])*2+2).to(device)
    
    model.load_wt(f"./weights/{dset_args['pred_ticker']}_model_wt.pt")

    val_df = get_unnormed_val_predictions(val_dset, dset_args["pred_win"], dset_args["pred_ticker"], 
                                            model, col_mean, col_std, device)

    # pdb.set_trace()
    print(plt.hist(val_df["diff"], bins=20))
    plt.savefig(f"{dset_args['pred_ticker']}_pred_diff.png")

    val_df.to_csv(f"{dset_args['pred_ticker']}_predictions.csv")
    
    print(f"average difference from actual stock price:{val_df['diff'].mean():.2f}")

if __name__ == "__main__":
    main()



