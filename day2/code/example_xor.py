import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


def threshold(output):
    return 1 if output > 0.5 else 0
    
def final_output(output):
    return [threshold(e[0][0]) for e in output]
    

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)

print("Predictions:")
for i, e in enumerate(final_output(out)):
    print(f"for label: {x_train[i][0]} {e}")
    