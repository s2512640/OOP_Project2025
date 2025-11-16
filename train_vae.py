from Encoder import Encoder
from Decoder import Decoder
from MNISTBWLoader import MNISTBWLoader
from MNISTColLoader import MNISTColLoader
from VAE import VAE
from neural_networks import encoder_mlp, decoder_mlp, encoder_conv, decoder_conv #commented out some lines in this file
import argparse
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_grid #modified naming in this method

"""
This script loads data, initializes a Variational Autoencoder (VAE) model
with either MLP or CNN architectures based on the dataset ('mnist_bw' or 
'mnist_color'), trains the VAE, and performs optional visualization tasks 
like latent space analysis and image generation from the prior and posterior.

Usage:
  python train_vae.py --dset [mnist_bw|mnist_color] --epochs [N] 
                      [--visualize_latent] [--generate_from_prior] 
                      [--generate_from_posterior]
"""

parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE) on MNIST datasets")

#telling parser whitch command-line arguments to accept
parser.add_argument("--dset", type=str, required=True,
                    help="Dataset to use: 'mnist_bw' or 'mnist_color'")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs")

#for visualization, adding boolean flags
parser.add_argument("--visualize_latent", action="store_true",
                    help="Visualize latent space after training")
parser.add_argument("--generate_from_prior", action="store_true",
                    help="Generate new images by sampling from prior p(z)")
parser.add_argument("--generate_from_posterior", action="store_true",
                    help="Generate new images by sampling from posterior q(z|x)")


args = parser.parse_args() #reads the command line arguments

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
if args.dset=="mnist_color":
    my_data_loader=MNISTColLoader() #loads dataset specified in commandline
    my_encoder=Encoder(encoder_conv)
    my_decoder=Decoder(decoder_conv)

elif args.dset=="mnist_bw":
    my_data_loader=MNISTBWLoader()
    my_encoder=Encoder(encoder_mlp)
    my_decoder=Decoder(decoder_mlp)


tr_data, test_data, y = my_data_loader.load_data() #default batch_size is set to 128
assert test_data is not None, "Error: test_data could not be loaded." #asserting that the data is loaded
print("Data is laoded") #for own logging 

print("Starting training.")

vae_model=VAE(my_encoder, my_decoder)
for e in range(args.epochs):
    for i, tr_batch in enumerate(tr_data):
        loss=vae_model.train(tr_batch, optimizer)
    print(f"On epoch: {e}")

print("Training completed ")

z, mu, log_var = vae_model.encode(test_data) #Encode test images to latent space
mu_x, log_sigma_x, x_hat = vae_model.decode(z) #Decode latent vectors back to images

#helper method to reshape
def reshape(x):
    if x.ndim == 2:
        if x.shape[1] == 784:
            x = x.reshape(-1, 28, 28, 1)
       
    #Normalize to [0,1] for mnist_color
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


if args.visualize_latent:
    print("Task 1f2: Generating and visualizing latent space")
    z_2d = TSNE(n_components=2).fit_transform(mu)
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap='tab10', s=5)
    plt.savefig('latent_space_'+ args.dset+'sne.pdf')  #Saves the figure to a file
    plt.close()


if args.generate_from_prior:
    print("Task 1f3: sampling z from prior")
    latent_dim = z.shape[1]
    z_prior = np.random.normal(size=(100, latent_dim)) #Prior is sampled from normal distribution
    mu_x, log_sigma_x, x_hat = vae_model.decode(z_prior)
    x_prior = mu_x.numpy()

    x_prior=reshape(x_prior) #Reshaping or normalizing
    plot_grid(x_prior[:100],args.dset, N=10, C=10, name='prior')


if args.generate_from_posterior:
    print("Task 1f4: Generate new images by sampling z from the posterior q(z|x)")
    z_post = vae_model.sample_posterior(mu, log_var)

    mu_x_post, log_sigma_x_post, x_hat_post = vae_model.decode(z_post)
    x_post = reshape(mu_x_post.numpy())
    plot_grid(x_post[:100], args.dset, N=10, C=10, name='posterior')


    