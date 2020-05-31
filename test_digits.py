from nn import *

if __name__ == '__main__':
    # Load training data and normalize it
    training_data = np.genfromtxt('mnist_train_1000.csv', delimiter = ',')
    output_len = 10 # Categories
    labels = np.zeros((training_data.shape[0], output_len))
    labels[np.arange(len(labels)), training_data[:, 0].astype(int)] = 1
    training_data = training_data[:, 1:]
    training_data /= 255

    # Construct the network     I                             H                     O
    nn = NeuralNetwork(training_data.shape[1], [int(training_data.shape[1]/4)], output_len)

    # Train the network
    nn.train(training_data, labels)

    # Load testing data and normalize it
    testing_data = np.genfromtxt('mnist_test_100.csv', delimiter = ',')
    labels = np.zeros((testing_data.shape[0], output_len))
    labels = testing_data[:, 0]
    testing_data = testing_data[:, 1:]
    testing_data /= 255

    # Test the network
    predicted = np.argmax(nn.predict(testing_data), axis = 1)
    accu = np.where(labels != predicted, 0, 1)
    print('Accuracy {:.2f}'.format(accu.sum()/accu.shape[0]*100)+'%')
