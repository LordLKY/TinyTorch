import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an N x H x W x C NDArray.
        Args:
            img: N x H x W x C NDArray of an image
        Returns:
            N x H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        assert len(img.shape) == 4, "img should have 4 dimensions (N, H, W, C)"
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          return img[:, :, ::-1, :]
        else:
          return img
        ### END YOUR SOLUTION


class RandomFlipVertical(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Vertically flip an image, specified as an N x H x W x C NDArray.
        Args:
            img: N x H x W x C NDArray of an image
        Returns:
            N x H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        assert len(img.shape) == 4, "img should have 4 dimensions (N, H, W, C)"
        flip_img = np.random.rand() < self.p
        if flip_img:
          return img[:, :, :, ::-1]
        else:
          return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: N x H x W x C NDArray of an image
        Return 
            N x H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        assert len(img.shape) == 4, "img should have 4 dimensions (N, H, W, C)"
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        pad_img = np.pad(img, self.padding, 'constant', constant_values=0)
        return pad_img[
           self.padding:self.padding+img.shape[0],
           self.padding+shift_x:self.padding+shift_x+img.shape[1],
           self.padding+shift_y:self.padding+shift_y+img.shape[2],
           self.padding:self.padding+img.shape[3]]
        ### END YOUR SOLUTION
