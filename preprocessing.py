import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from loader.normals_loader import CustomKitti, CustomKittiProcessing
from loader.normals_loader import prep_long_2_right_up, prep_long_2_half_right_up, prep_stock, prep_cross_aver


if __name__ == "__main__":
    training_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/", prep_stock) 
    validation_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/", prep_stock, mode="val") 
    test_data = CustomKittiProcessing("/home/polosatik/mnt/kitty/dataset/sequences/00/", prep_stock, mode="test") 

    training_loader = DataLoader(training_data, batch_size=1, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    name = "data/normals_simple"
    with open(f'{name}_train_data.npy', 'wb') as d:
        with open(f'{name}_train_labels.npy', 'wb') as l:
            length = 3700
            data = np.empty((length, 3, 64, 1024))
            labels = np.empty((length, 64, 1024))
            for i in tqdm(range(length)):
                n = next(iter(training_loader))
                np.append(data, n[0].squeeze())
                labels[i] = n[1].squeeze()
            np.save(d, data)
            np.save(l, labels)
    with open(f'{name}_val_data.npy', 'wb') as d:
        with open(f'{name}_val_labels.npy', 'wb') as l:
            length = 300
            data = np.empty((length, 3, 64, 1024))
            labels = np.empty((length, 64, 1024))
            for i in tqdm(range(length)):
                n = next(iter(training_loader))
                np.append(data, n[0].squeeze())
                labels[i] = n[1].squeeze()
            np.save(d, data)
            np.save(l, labels)
    with open(f'{name}_test_data.npy', 'wb') as d:
        with open(f'{name}_test_labels.npy', 'wb') as l:
            length = 541
            data = np.empty((length, 3, 64, 1024))
            labels = np.empty((length, 64, 1024))
            for i in tqdm(range(length)):
                n = next(iter(training_loader))
                np.append(data, n[0].squeeze())
                labels[i] = n[1].squeeze()
            np.save(d, data)
            np.save(l, labels)