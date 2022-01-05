import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as data

np.random.seed(1500)
torch.manual_seed(7)


class DREAMERDataset(data.Dataset):
    def __init__(
            self,
            train=True,
            transfer=False
    ):
        raw = sio.loadmat('./PR/Data/sub_test.mat')
        train_data = raw['train_data']
        train_label = raw['train_label'][0]
        transfer_data = raw['transfer_data']
        transfer_label = raw['transfer_label'][0]
        test_data = raw['test_data']
        test_label = raw['test_label'][0]
        
        self.train = train
        self.transfer = transfer
        if self.train:
            if self.transfer:
                self.train_data = train_data
                self.train_labels = train_label
            else:
                self.train_data = transfer_data
                self.train_labels = transfer_label
            print(self.train_data.shape)
            print(self.train_labels.shape)

        else:
            self.test_data = test_data
            self.test_labels = test_label
            print(self.test_data.shape)
            print(self.test_labels.shape)


    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index, :, :]
            target = self.train_labels[index]
        else:
            img = self.test_data[index, :, :]
            target = self.test_labels[index]
        img = torch.from_numpy(img).float()
        target = target.astype(np.int64)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
