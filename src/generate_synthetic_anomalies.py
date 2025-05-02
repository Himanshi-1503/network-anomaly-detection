import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model

# Define dimensions
input_dim = 10  # Number of features in anomalies (match this with the original dataset)

# Build Generator
def build_generator():
    inputs = Input(shape=(input_dim,))
    x = Dense(128)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(input_dim, activation="tanh")(x)
    
    return Model(inputs, x)

# Build Discriminator
def build_discriminator():
    inputs = Input(shape=(input_dim,))
    x = Dense(256)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation="sigmoid")(x)
    
    return Model(inputs, x)

# Compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Connect Generator and Discriminator
discriminator.trainable = False
gan_input = Input(shape=(input_dim,))
generated_anomaly = generator(gan_input)
gan_output = discriminator(generated_anomaly)
gan = Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer="adam")

# Train GAN
batch_size = 32
epochs = 2000  # Increased number of epochs
print_interval = 100  # Increased print interval

for epoch in range(epochs):
    # Generate fake anomalies
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    fake_anomalies = generator.predict(noise, verbose=0)

    # Get real anomalies (replace with actual anomaly dataset)
    real_anomalies = np.random.randn(batch_size, input_dim)

    # Train Discriminator
    X_combined = np.vstack((real_anomalies, fake_anomalies))
    y_combined = np.hstack((np.ones(batch_size), np.zeros(batch_size)))
    d_loss, d_acc = discriminator.train_on_batch(X_combined, y_combined)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    y_fake = np.ones(batch_size)  # Trick discriminator
    g_loss = gan.train_on_batch(noise, y_fake)

    if epoch % print_interval == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss:.4f}, D Acc: {d_acc:.4f}, G Loss: {g_loss:.4f}")

# Generate new synthetic anomalies
num_synthetic = 1000  # Generate enough synthetic anomalies to balance the dataset
noise = np.random.normal(0, 1, (num_synthetic, input_dim))
synthetic_anomalies = generator.predict(noise, verbose=0)

# Save synthetic anomalies
np.save("c:\\Users\\himan\\OneDrive\\Desktop\\network-anomaly-detection-1\\synthetic_anomalies.npy", synthetic_anomalies)

print("Synthetic anomalies generated and saved to synthetic_anomalies.npy")