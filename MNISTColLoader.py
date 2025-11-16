from Data_loader import DataLoader
import numpy as np
import pickle

class MNISTColLoader(DataLoader):
    """
    Loads and preprocesses the color MNIST dataset.
    Extends DataLoader by downloading the dataset files, selecting a specific 
    color version (m0–m4), and extracting the corresponding training and test 
    data from the provided pickle dictionaries.

    Parameters
    color_version : str, optional
        The chosen color subset (default "m1").
    data_dir : str, optional
        Directory where dataset files are stored.

    Returns (from load_raw_data)
    tuple
        x_train, x_test, y_test arrays.
    """
    def __init__(self, color_version="m1", data_dir="OOP_PROJECT2025"):
        super().__init__(data_dir) 

        self._urls={
                    "train":  ("https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0", "mnist_color.pkl"),
                    "test":   ("https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0", "mnist_color_te.pkl"),
                    "labels": ("https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0",  "mnist_color_y_te.npy"),
                }
        self._color_version=color_version
    
    @property
    def color_version(self):
        """
        Returns the currently selected color version (m0–m4)
        used for extracting data from the color-MNIST dictionaries.
        """
        return self._color_version

    @color_version.setter
    def color_version(self, new_version):
        """
        Updates the selected color version with validation.

        Parameters
        new_version : str
            One of {'m0', 'm1', 'm2', 'm3', 'm4'}.

        Raises
        ValueError
            If the provided version is not valid.
        """
        valid_versions = ['m0', 'm1', 'm2', 'm3', 'm4']
        
        #Assertion/Defensive check
        if new_version not in valid_versions:
            raise ValueError(f"Invalid color version '{new_version}'. Must be one of {valid_versions}.")
        self._color_version = new_version

    #overriding
    def load_raw_data(self):
        """
        Downloads and loads the color MNIST dataset.
        Loads the dataset dictionaries from pickle files, extracts the 
        selected color version, and returns the training and test arrays.

        Returns
        tuple
            x_train, x_test, y_test arrays.
        """
        paths = self._download_dataset(self._urls)

        #loading training and test files with pickle
        with open(paths["train"], "rb") as f:
            color_train_dict = pickle.load(f)
        with open(paths["test"], "rb") as f:
            color_test_dict = pickle.load(f)

        #loading labels with numpy
        y_test = np.load(paths["labels"])
        
        x_train = color_train_dict[self._color_version]
        x_test  = color_test_dict[self._color_version]

        return x_train, x_test, y_test
    
