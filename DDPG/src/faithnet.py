import tensorflow as tf
import numpy as np

class network():
    def __init__(self,
                input_states,
                num_layers,
                nurons_list,
                output_states,
                session):
        
        # Check if the number of neurons in each layer is 
        if len(nurons_list) != num_layers:
            raise ValueError("nurons_list != num of layers")
        
        #Initalize the session
        self.session = session
        
        # Initialize the network
        self.input_states = tf.placeholder(dtype=tf.float64, shape=[None, input_states])
        self.target_states = tf.placeholder(dtype=tf.float64, shape=[None, output_states])
        
        # Create a dictonary of weight and states based on the input layers
        # Compute the dimentions of the weight bias matrix
        self.network_depth = num_layers
        
        nurons_list.append(output_states)
        nurons_list = [input_states] + nurons_list
        
        self.network_width = nurons_list # list of nurons in each layer including input and output 
        
        self.weights_bias = {}
        for i in range(0, self.network_depth+1):
            self.weights_bias["w"+str(i)] = tf.Variable(np.random.rand(self.network_width[i], self.network_width[i+1]), dtype=tf.float64)
            self.weights_bias["b"+str(i)] = tf.Variable(np.random.rand(self.network_width[i+1]), dtype=tf.float64)
        
        
        # Set the computation graph for the network
        self.forward_pass = {}
        # First layer
        self.forward_pass["z1"] = tf.tensordot(self.input_states, self.weights_bias["w0"], axes=1) + self.weights_bias["b0"]
        self.forward_pass["y1"] = tf.nn.relu(self.forward_pass["z1"]) # Make this a user choice 
        
        for i in range(2, self.network_depth+1):
            self.forward_pass["z"+str(i)] = tf.tensordot(self.forward_pass["y"+str(i-1)],
                                                         self.weights_bias["w"+str(i-1)],
                                                         axes=1) + self.weights_bias["b"+str(i-1)]
            self.forward_pass["y"+str(i)] = tf.nn.relu(self.forward_pass["z"+str(i)])
            
        # Final Layer with out activation
        self._predict = tf.tensordot(self.forward_pass["y"+str(self.network_depth)],
                                    self.weights_bias["w"+str(self.network_depth)],
                                    axes=1) + self.weights_bias["b"+str(self.network_depth)]
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(self._predict - self.target_states))
        
        # Optimizer 
        self.optimizer = tf.train.RMSPropOptimizer(0.01)
        
        # Training
        self._train = self.optimizer.minimize(self.loss)
        
        # Initialize the variables 
        self.session.run(tf.global_variables_initializer())
        
    def predict_on_batch(self, input_states):
        return self.session.run(self._predict, {self.input_states:input_states})
    
    def fit(self, inp_states, tar_states):
        self.session.run(self._train, {self.input_states:inp_states, self.target_states:tar_states})
        # Return loss function.
    
    def save_weights(self, path):
        data = {}
        for i in range(0, self.network_depth+1):
            data["w"+str(i)] = self.session.run(self.weights_bias["w"+str(i)])
            data["b"+str(i)] = self.session.run(self.weights_bias["b"+str(i)])
        np.save(path, data)
        print("Weights Saved to ", path)
        
    def get_weights(self):
        data = {}
        for i in range(0, self.network_depth+1):
            data["w"+str(i)] = self.session.run(self.weights_bias["w"+str(i)])
            data["b"+str(i)] = self.session.run(self.weights_bias["b"+str(i)])
        return data
    
    def set_weights(self, weigths_bias_load):
        # Make sure the width and depth are equal 
        for i in range(0, self.network_depth+1):    
            self.session.run(self.weights_bias["w"+str(i)].assign(weigths_bias_load["w"+str(i)]))
            self.session.run(self.weights_bias["b"+str(i)].assign(weigths_bias_load["b"+str(i)]))


