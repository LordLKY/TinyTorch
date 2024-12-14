import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X, y = [], []
        if train:
            for i in range(1, 6):
                with open(os.path.join(base_folder, f'data_batch_{i}'), 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    # NOTE key: b''
                    X.append(dict[b'data'].astype(np.float32))
                    y.append(dict[b'labels'])
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                X.append(dict[b'data'].astype(np.float32))
                y.append(dict[b'labels'])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X /= 255.0
        self.X = X
        self.y = y
        self.len = y.shape[0]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        if self.transforms:
            # original shape is NCHW, should transform it to NHWC before applying transforms
            X_in = np.transpose(X.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
            X_out = self.apply_transforms(X_in)
            X_ret = np.transpose(X_out, (0, 3, 1, 2)).reshape((-1, 3, 32, 32))
            return X_ret, y
        else:
            return np.squeeze(X.reshape((-1, 3, 32, 32))), y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.len
        ### END YOUR SOLUTION