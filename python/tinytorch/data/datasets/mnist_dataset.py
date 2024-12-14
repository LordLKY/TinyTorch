from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        data_folder = ''
        with gzip.open(data_folder + label_filename, 'rb') as lbpath:
          y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(data_folder + image_filename, 'rb') as imgpath:
          x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
          x_train = x_train.astype(np.float32) / 255.0
        
        self.x_train, self.y_train = x_train, y_train
        self.len = y_train.shape[0]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # return (self.apply_transforms(self.x_train)[index], self.y_train[index])
        X, y = self.x_train[index], self.y_train[index]
        # NOTE: self.transforms need input shape like this.
        # thanks to https://github.com/YuanchengFang/dlsys_solution/blob/master/hw2/python/needle/data.py
        if self.transforms:
            X_in = X.reshape((28, 28, -1))
            X_out = self.apply_transforms(X_in)
            X_ret = X_out.reshape(-1, 28 * 28)
            return X_ret, y
        else:
            return X, y 
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.len
        ### END YOUR SOLUTION