import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print training_data
# print '------------------------------'
# print validation_data
# print '------------------------------'
# print test_data
# print '-------------------------------'

net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


