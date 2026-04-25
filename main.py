import numpy as np 
import pandas as pd
from src.model import AQIModel
import torch 
import torch.nn as nn 
from src.merge import data_merge
from src.data_preprocessing import DataPreprocessing
from src.dataset import TimeSeriesDataset
from src.train import train #func
from src.evaluate import evaluate #func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    in_path = "/home/reda/AQI/data/raw"
    out_path = "/home/reda/AQI/data/raw/merge_df.csv"

    All_data = data_merge(
        in_path,
        out_path
    )

    data = DataPreprocessing(out_path, "PM2.5")
    output_path_name = "/home/reda/AQI/data/processed/processed_v1"
    processed_data = data.process("/home/reda/AQI/data/processed/processed_v1")

    dataset = TimeSeriesDataset(output_path_name)
    X_train,y_train, X_test, y_test = dataset.get_data()


    # config
    input_size = X_train.shape[2]
    hidden_size = 100
    output_size = 1
    num_layers = 2

    batch_size = 32
    epochs = 100
    lr = 0.001


    # model
    model = AQIModel(input_size, hidden_size, output_size, num_layers)
    model.to(device)

    #train 
    train(device, model, X_train, y_train, batch_size, lr, epochs)

    #eval
    evaluate(model ,X_test, y_test)












