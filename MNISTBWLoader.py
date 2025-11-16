from Data_loader import DataLoader
import numpy as np

class MNISTBWLoader(DataLoader):
    """
    Loads and preprocesses the grayscale MNIST dataset.

    Overrides load_raw_data() to download the required files, normalize 
    pixel values to [0, 1], and reshape each 28×28 image into a 784-dimensional 
    vector. Uses shared download utilities from the DataLoader superclass.

    Parameters
    data_dir : str, optional
        Directory where dataset files are stored. Defaults to "OOP_PROJECT2025".

    Returns (from load_raw_data)
    tuple[np.ndarray, np.ndarray, np.ndarray]
        x_train, x_test, y_test arrays.
    """
    def __init__(self, data_dir="OOP_PROJECT2025"):
        super().__init__(data_dir)

        self._urls={
                    "train": ("https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0", "mnist_bw.npy"),
                    "test": ("https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0", "mnist_bw_te.npy"),
                    "labels": ("https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0", "mnist_bw_y_te.npy"),
                }
    
    def load_raw_data(self):
        paths = self._download_dataset(self._urls)

        #scaling
        x_train = np.load(paths["train"]).astype("float32") / 255.0
        x_test  = np.load(paths["test"]).astype("float32") / 255.0
        y_test  = np.load(paths["labels"])

        #Vectorize 28x28 to 784
        x_train = x_train.reshape((x_train.shape[0], 28*28))
        x_test  = x_test.reshape((x_test.shape[0], 28*28))

        return x_train, x_test, y_test
    


        