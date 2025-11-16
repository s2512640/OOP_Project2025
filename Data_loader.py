import os
import subprocess
import numpy as np
import tensorflow as tf
import pickle

class DataLoader:
    """
    Superclass for dataset loaders.

    Provides shared functionality for downloading files, managing the data
    directory, and batching data for training. Subclasses must implement
    `load_raw_data()` to define dataset-specific preprocessing.

    Parameters
    data_dir : str, optional
        Directory where dataset files are stored. Default is "OOP_PROJECT2025".
    """
    
    def __init__(self, data_dir="OOP_PROJECT2025"):
        self._data_dir = data_dir
    
    #shared helper-methods
    def _download_file(self, url, filename):
            """
            Downloads a file if it does not already exist.

            Parameters
            url : str
                URL of the file to download.
            filename : str
                Local filename to save.

            Returns
            filepath : str
                Path to the downloaded or existing file.
            """
            os.makedirs(self._data_dir, exist_ok=True)
            filepath = os.path.join(self._data_dir, filename)

            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                subprocess.run(["wget", "-O", filepath, url], check=True)
            else:
                print(f"{filename} already exists.")
            return filepath
    
    def _download_dataset(self, url_dict):
        """
        Downloads all files defined in the URL dictionary.

        Parameters
        url_dict : dict
            Maps dataset parts to (url, filename) pairs.

        Returns
        local : dict
            Maps dataset parts to local file paths.
        """
        local = {}
        for key, (url, filename) in url_dict.items():
            local[key] = self._download_file(url, filename)
        return local #returns local paths
    
    #abstract method 
    def load_raw_data(self):
         """
        Loads and preprocesses dataset files.
        Must be implemented by subclasses.

        Raises
        NotImplementedError
            If called on the superclass.
        """
         raise NotImplementedError("Subclasses must implement load_raw_data()")
    

   #public interface
    def load_data(self, batch_size=128):
        """
        Loads raw data and returns it in batched TensorFlow format.

        Parameters
        batch_size : int, optional
            Batch size for training. Default is 128.

        Returns
        tuple
            (train_ds, x_test, y_test)
        """
        x_train, x_test, y_test = self.load_raw_data()
        train_ds = tf.data.Dataset.from_tensor_slices(x_train)
        train_ds = train_ds.batch(batch_size)

        return train_ds, x_test, y_test
    
    def set_data_dir(self, new_dir):
        """
        Updates the data directory name.

        Parameters
        new_dir : str
            New directory path.

        Raises
        ValueError
            If the directory name is invalid.
        """

        if not isinstance(new_dir, str) or not new_dir:
            raise ValueError("Data directory must be a non-empty string.")
        
        self._data_dir = new_dir
        print(f"Data directory set to: {self._data_dir}")