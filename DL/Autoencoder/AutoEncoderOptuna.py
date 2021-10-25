from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

latent_dim = 16

class Autoencoder(Model):

  def __init__(self, latent_dim, n_enc_layers, enc_reductions, n_dec_layers, dec_expansions):

    super(Autoencoder, self).__init__()

    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      layers.Input(x_train.shape[1]),
      layers.Flatten(),
      layers.Dense(256, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(latent_dim, activation='relu'),
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(64, activation='sigmoid'),
      layers.Dropout(0.5),
      layers.Dense(128, activation='sigmoid'),
      layers.Dropout(0.5),
      layers.Dense(256, activation='sigmoid'),
      layers.Dropout(0.5),
      layers.Dense(x_train.shape[1])
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
