import sound_input_loader
import network1

training_data, validation_data, test_data = sound_input_loader.input_data_wrapper()
training_data = list(training_data)

net = network1.Network([441000, 30, 1])
net.SGD(training_data, 30, 5, 3.0, test_data=test_data)