import tensorflow as tf
from losses import kl_divergence, log_diag_mvn
import numpy as np

class VAE(tf.keras.Model):
    """
    Implements the Variational Autoencoder (VAE).

    This class connects the Encoder and Decoder components to compute the VAE objective function (ELBO). It inherits from 
    tf.keras.Model for gradient tracing. 
    """
    def __init__(self, encoder, decoder):
        """
        Initializes the VAE model.

        It stores private instances of the specialized 
        Encoder and Decoder classes. 

        Parameters
        encoder : Encoder
            An instance of the probabilistic Encoder class.
        decoder : Decoder
            An instance of the probabilistic Decoder class.
        """
        super().__init__()
        self._encoder = encoder #instance of encoder class
        self._decoder = decoder #instance of decoder class
        self._vae_loss= None
    
    #public
    def encode(self, x):
        return self._encoder(x)
    
    #public
    def decode(self, z):
        return self._decoder(z)
    
    
    def call(self, x):
        """
        Performs the VAE forward pass and computes the VAE loss.

        This method is essential for optimization. It calculates the 
        negative of the ELBO, which is the objective function being minimized.

        Parameters
        x : tf.Tensor
            The input data batch.

        Returns
        tf.Tensor
            The computed VAE loss (negative ELBO).
        """
        z, mu, log_var=self._encoder(x) #uses the call method in encoder 
        mu_x, log_sigma_x, x_hat=self._decoder(z) #send the latent sample to the decoder

        kl_div=kl_divergence(mu, log_var) #double-check parameters
        log_diag=log_diag_mvn(x, mu_x, log_sigma_x) 

        #Calculating the Total VAE Loss
        elbo = log_diag - kl_div
        self.vae_loss = -elbo 
        return self.vae_loss 
    
    @tf.function
    def train(self, x, optimizer):
        """
        Computes gradients and updates the network's trainable variables.

        This method uses the @tf.function decorator for improved performance.

        Parameters
        x : tf.Tensor
            The input data batch.

        optimizer : tf.keras.optimizers.Optimizer
            The optimizer used for applying gradients.

        Returns
        tf.Tensor
            The computed loss for the batch.
        """
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss
    
    def sample_posterior(self, mu, log_var):
        """
        Sample z from q(z|x) using reparameterization.

        Returns:
            z_post : np.ndarray
        """
        eps = np.random.normal(size=mu.shape)
        return mu + np.exp(0.5 * log_var) * eps
