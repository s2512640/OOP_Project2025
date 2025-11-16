from BiCoder import BiCoder
import tensorflow as tf

class Encoder(BiCoder):
    """
    Probabilistic Encoder (qφ(z|x)).
    This class inherits from BiCoder and implements the specific behavior 
    required to compute the parameters (µ and log(σ²)) for the latent variable z.

    """
    def __init__(self, encoder):
        """
        Initializes the Encoder instance.
        Parameters
        encoder : tf.keras.Sequential
            The deep neural network used to learn the parameters 
            µ and log(σ²) from the input data x.
        """
        super().__init__(encoder) #encoder is either mlp or cnn
       
    #implementing abstarct method from superclass
    def call(self, x):
        """
        Implements the forward pass for the probabilistic encoder qφ(z|x).
        It passes the input through the network, splits the output into mean and 
        log-variance, computes the standard deviation, and performs the reparameterization to sample the latent variable z.

        Parameters
        x : tf.Tensor
            The input data batch.

        Returns
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            z: The sampled latent variable.
            mu: The mean of q(z|x).
            log_var: The logarithm of the variance of q(z|x).
        
        """
        out = self.forward_network(x) 
        latent_dim = out.shape[1] // 2 #works for every architecture
        mu = out[:, :latent_dim]
        log_var = out[:, latent_dim:]
        sigma = tf.exp(0.5 * log_var)
        z = self.sample_gaussian(mu, sigma)  #reparametratrization 
        return z, mu, log_var
        