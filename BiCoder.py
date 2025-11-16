import tensorflow as tf
from tensorflow.keras import layers

class BiCoder(layers.Layer):
    """
    Abstract superclass for both Encoder and Decoder.

    This class provides the shared structure and behavior, including the
    network forward pass and the sampling method, ensuring
    code reuse for components of the VAE. The class inherts from 
    tf.keras.layers.Layer to allow self.trainable_variables to be tracked 
    correctly by TensorFlow.
    """

    def __init__(self, network):
        """
        Initializes the BiCoder with a pre-built neural network architecture.
        
        Parameters
        network : tf.keras.Sequential
            A prebuilt neural network (MLP or CNN) that serves as either the
            encoder network or the decoder network.
        """
        super().__init__()
        self._network = network

 
    def forward_network(self, x):
        """
        Passes input through the assigned neural network.
        This method encapsulates the common initial step for both the Encoder 
        and Decoder

        Parameters
        x : tf.Tensor
            The input data, which is either the input data (x) for the Encoder
            or the latent sample (z) for the Decoder.

        Returns
        tf.Tensor
            The output of the neural network, which contains the parameters mu and sigma
            for the distribution calculation.
        """
        return self._network(x) #output thta can be broken down to mu and sigma 


    def sample_gaussian(self, mu, sigma):
        """
        Implements the sampling method used for the VAE's 
        reparameterization trick.

        This method is shared to prevent redundant code in the Encoder and Decoder.

        Parameters
        mu : tf.Tensor - Mean of the Gaussian distribution.
        sigma : tf.Tensor or float - Standard deviation of the Gaussian.

        Returns
        sample : tf.Tensor- Sample from N(mu, sigma^2).
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + sigma * eps


    def call(self, inputs):
        """
        This method must be overridden by the Encoder and Decoder subclasses to 
        provide their specific behavior.

        Parameters
        inputs : tf.Tensor
            The input data (x) or latent sample (z).
            
        Raises
        NotImplementedError
            This error is raised because this method is intended to be abstract 
            and must be implemented.
        """
        raise NotImplementedError("Subclasses must implement load_raw_data()")