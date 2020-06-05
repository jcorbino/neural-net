from nn import *

np.set_printoptions(formatter = {'float': '{:.2f}'.format})

if __name__ == '__main__':
    # Input
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # Expected output
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    #                  I   H   O
    nn = NeuralNetwork(2, [2], 1, initializer = 'rand')

    # Train the network
    nn.train(X, y, epochs = 1000, lr = 2, activations = ['sigmoid', 'sigmoid'])

    # Test the network
    print(nn.predict(X))
