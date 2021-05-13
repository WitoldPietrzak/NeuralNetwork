
import random
import math


class Neuron:
    def __init__(self, input_length, activation_function_choice='j', bias=True, step=1, momentum=0.8, beta=1):
        self.bias = bias
        self.input_vector = []
        self.weight_vector = []
        for weight in range(input_length):
            self.weight_vector.append(random.randrange(-5000, 5000) / 1000)
        if self.bias:
            self.weight_vector.append(random.randrange(-5000, 5000) / 1000)
        self.previous_iteration_weight_vector = self.weight_vector.copy()
        self.input_sum_value = 0
        self.output_value = 0
        self.expected_output_value = 0
        self.derivative_value = 0
        self.beta = beta
        self.step = step
        self.momentum = momentum
        self.difference = 0
        self.b_error = 0
        self.activation_function_choice = activation_function_choice

    def load_input(self, input_vector):
        self.input_vector.clear()
        if self.bias:
            self.input_vector = [1]
        else:
            self.input_vector = []
        for input_value in input_vector:
            self.input_vector.append(input_value)

    def input_sum_function(self):
        self.input_sum_value = 0
        for i in range(len(self.input_vector)):
            self.input_sum_value += self.weight_vector[i] * self.input_vector[i]
        return self.input_sum_value

    def jump_activation_function(self):
        if self.input_sum_value >= 0:
            self.output_value = 1
        else:
            self.output_value = 0
        return self.output_value

    def linear_activation_function(self):
        self.output_value = self.input_sum_value
        return self.output_value

    def unipolar_activation_function(self):
        self.output_value = 1.0 / (1 + math.exp(-1 * self.beta * self.input_sum_value))
        return self.output_value

    def bipolar_activation_function(self):
        self.output_value = math.tanh(self.beta * self.input_sum_value)
        return self.output_value

    def linear_activation_function_derivative(self):
        self.derivative_value = 1
        return self.derivative_value

    def unipolar_activation_function_derivative(self):
        self.derivative_value = (self.beta * self.output_value) * (1 - self.output_value)
        return self.derivative_value

    def bipolar_activation_function_derivative(self):
        self.derivative_value = 1 - pow(self.output_value, 2)
        return self.derivative_value

    def calculate_derivative(self):
        if self.activation_function_choice == 'u':
            return self.unipolar_activation_function_derivative()
        elif self.activation_function_choice == 'b':
            return self.bipolar_activation_function_derivative()
        elif self.activation_function_choice == 'l':
            return self.linear_activation_function_derivative()

    def use_activation_function(self):
        if self.activation_function_choice == 'j':
            return self.jump_activation_function()
        elif self.activation_function_choice == 'l':
            return self.linear_activation_function()
        elif self.activation_function_choice == 'u':
            return self.unipolar_activation_function()
        elif self.activation_function_choice == 'b':
            return self.bipolar_activation_function()

    def calculate_output(self):
        self.input_sum_function()
        self.use_activation_function()
        return self.output_value

    def calculate_difference(self):
        self.difference = self.output_value - self.expected_output_value
        return self.difference

    def modify_weights(self):
        for i in range(len(self.weight_vector)):
            temp_value = self.weight_vector[i] + -1 * self.step * self.b_error * self.input_vector[
                i] + self.momentum * (self.weight_vector[i] - self.previous_iteration_weight_vector[i])
            self.previous_iteration_weight_vector[i] = self.weight_vector[i]
            self.weight_vector[i] = temp_value


class Layer:
    def __init__(self, input_length, layer_size, activation_function_choice='j', bias=True, step=1, momentum=0.8,
                 beta=1):
        self.neurons = []
        for k in range(layer_size):
            self.neurons.append(Neuron(input_length, activation_function_choice, bias, step, momentum, beta))

        self.output_vector = []

    def load_input(self, input_vector):
        for neuron in self.neurons:
            neuron.load_input(input_vector)

    def load_expected_output(self, expected_output_vector):
        for n in range(len(self.neurons)):
            self.neurons[n].expected_output_value = expected_output_vector[n]

    def calculate_output(self):
        self.output_vector.clear()
        for neuron in self.neurons:
            self.output_vector.append(neuron.calculate_output())
        return self.output_vector

    def modify_weights(self):
        for n in self.neurons:
            n.modify_weights()

    def get_output(self):
        output = []
        max_value = max(self.output_vector)
        for o in self.output_vector:
            if o == max_value:
                output.append(1)
            else:
                output.append(0)
        return output

    def calculate_error(self):
        error = 0
        for neruon in self.neurons:
            error += pow(neruon.calculate_difference(), 2) / 2
        error /= len(self.neurons)
        return error


class Network:
    def __init__(self, input_length, layer_quantity, layer_size_vector, activation_function_choice_vector, bias,
                 step_vector, momentum_vector, beta_vector):
        self.layers = [
            Layer(input_length, layer_size_vector[0], activation_function_choice_vector[0], bias, step_vector[0],
                  momentum_vector[0], beta_vector[0])]
        for q in range(1, layer_quantity):
            self.layers.append(
                Layer(layer_size_vector[q - 1], layer_size_vector[q], activation_function_choice_vector[q],
                      bias, step_vector[q], momentum_vector[q], beta_vector[q]))

        self.output_vector = []
        self.bias = bias

    def load_input(self, input_vector):
        self.layers[0].load_input(input_vector)

    def load_training_input(self, input_vector, expected_output_vector):
        self.load_input(input_vector)
        self.layers[len(self.layers) - 1].load_expected_output(expected_output_vector)

    def calculate_output(self):
        for layer in range(1, len(self.layers)):
            self.layers[layer].load_input(self.layers[layer - 1].calculate_output())
        self.output_vector = self.layers[len(self.layers) - 1].calculate_output()
        return self.output_vector

    def get_output(self):
        return self.layers[len(self.layers) - 1].get_output()

    def backpropagation_algorithm(self):
        layer_counter = len(self.layers) - 1
        while layer_counter >= 0:
            for m in range(len(self.layers[layer_counter].neurons)):
                if layer_counter == len(self.layers) - 1:
                    self.layers[layer_counter].neurons[m].b_error = self.layers[layer_counter].neurons[
                                                                        m].calculate_difference() * \
                                                                    self.layers[layer_counter].neurons[
                                                                        m].calculate_derivative()
                else:
                    sum = 0
                    for j in self.layers[layer_counter + 1].neurons:
                        if self.bias:
                            sum += j.b_error * j.weight_vector[m + 1]
                        else:
                            sum += j.b_error * j.weight_vector[m]
                    self.layers[layer_counter].neurons[m].b_error = sum * self.layers[layer_counter].neurons[
                        m].calculate_derivative()
            layer_counter -= 1

    def calculate_difference(self):
        for neuron in self.layers[len(self.layers) - 1].neurons:
            neuron.calculate_difference()

    def modify_weights(self):
        for l in self.layers:
            l.modify_weights()

    def calculate_error(self):
        return self.layers[len(self.layers) - 1].calculate_error()
