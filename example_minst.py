import numpy as np
from keras.datasets import mnist
from utils import to_categorical
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime
import matplotlib.pyplot as plt

# Plotting predictions graphically
def plot_predictions(x_test, y_test, predictions, num_samples=10):
    plt.figure(figsize=(10, 5))  # Set the figure size
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)  # Create a 2x5 grid for 10 samples
        image = x_test[i].reshape(28, 28)  # Reshape flat image back to 28x28
        plt.imshow(image, cmap='gray')  # Display the image in grayscale
        predicted_label = threshold(predictions[i])  # Get the predicted label
        actual_label = np.argmax(y_test[i])  # Get the actual label
        
        # Add the predicted and actual labels as the title
        plt.title(f"P: {predicted_label}, A: {actual_label}")
        plt.axis('off')  # Turn off the axis for a cleaner look
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.reshape(-1, 1, 28 * 28).astype("float32") / 255 
x_test = x_test.reshape(-1, 1, 28 * 28).astype("float32") / 255  


# One-hot encode the labels
y_train = to_categorical(y_train, 10)  # 10 classes (0–9)
y_test = to_categorical(y_test, 10)

def threshold(output):
    return np.argmax(output)  # Find the index of the highest value (predicted class)

# network
net = Network()
net.add(FCLayer(28 * 28, 128))  # Input layer size: 784, hidden layer size: 128
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(128, 64))  # Hidden layer size: 64
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(64, 10))  # Output layer size: 10
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=10, learning_rate=0.01)  # Reduced epochs for quick testing

# test
out = net.predict(x_test)

print("Predictions:")
for i, e in enumerate(out[:10]):  # Show predictions for the first 10 test samples
    predicted_label = threshold(e)
    actual_label = np.argmax(y_test[i])
    print(f"Sample {i + 1}: Predicted={predicted_label}, Actual={actual_label}")
    
plot_predictions(x_test, y_test, out)  # Plot the predictions