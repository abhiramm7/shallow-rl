import numpy as np
import tensorflow as tf

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
        # Hubers labels and predictions
        self.loss = tf.losses.huber_loss(self.target_states, self._predict)
        
        # Optimizer 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
        
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


class replay_stacker():
    def __init__(self, columns, window_length=100):
        self._data = np.zeros((window_length, columns))
        self.capacity = window_length
        self.size = 0
        self.columns = columns

    def update(self, x):
        self._add(x)

    def _add(self, x):
        if self.size == self.capacity:
            self._data = np.roll(self._data, -1)
            self._data[self.size-1, :] = x
        else:
            self._data[self.size, :] = x
            self.size += 1

    def data(self):
        return self._data[0:self.size, :]


class replay_memory_agent():
    def __init__(self, states_len, replay_window):
        self.states_len = states_len
        self.replay_window = replay_window

        # Initialize replay memory
        self.replay_memory = {'states': replay_stacker(self.states_len, self.replay_window),
                              'states_new': replay_stacker(self.states_len,self.replay_window),
                              'rewards': replay_stacker(1,self.replay_window),
                              'actions': replay_stacker(1,self.replay_window),
                              'terminal': replay_stacker(1,self.replay_window)}

    def replay_memory_update(self,
                             states,
                             states_new,
                             rewards,
                             actions,
                             terminal):

        self.replay_memory['rewards'].update(rewards)
        self.replay_memory['states'].update(states)
        self.replay_memory['states_new'].update(states_new)
        self.replay_memory['actions'].update(actions)
        self.replay_memory['terminal'].update(terminal)


def randombatch(sample_size, replay_size):
    indx = np.linspace(0, replay_size-1, sample_size)
    indx = np.random.choice(indx, sample_size, replace=False)
    indx.tolist()
    indx = list(map(int, indx))
    return indx


class deep_q_agent:
    def __init__(self,
                 action_value_model,
                 target_model,
                 states_len,
                 replay_memory,
                 call,
                 batch_size=32,
                 target_update=10000,
                 train=True,):

        self.states_len = states_len
        self.ac_model = action_value_model
        self.target_model = target_model
        self.replay = replay_memory
        self.batch_size = batch_size
        self.train = train
        self.target_update = target_update
        self.call = call 

        self.state_vector = np.zeros((1, self.states_len))
        self.state_new_vector = np.zeros((1, self.states_len))
        self.rewards_vector = np.zeros((1))
        self.terminal_vector = np.zeros((1))
        self.action_vector = np.zeros((1))
 
        self.training_batch = {'states': np.zeros((self.batch_size, self.states_len)),
                               'states_new': np.zeros((self.batch_size, self.states_len)),
                               'actions': np.zeros((self.batch_size, 1)),
                               'rewards': np.zeros((self.batch_size, 1)),
                               'terminal': np.zeros((self.batch_size, 1))}

    def _random_sample(self):
        temp_l= len(self.replay.replay_memory['states'].data())
        indx = randombatch(self.batch_size, temp_l)
        for i in self.training_batch.keys():
            temp = self.replay.replay_memory[i].data()
            self.training_batch[i] = temp[indx]


    def _update_target_model(self):
        self.target_model.set_weights(self.ac_model.get_weights())


    def _train(self):
        temp_states_new = self.training_batch['states_new']
        temp_states = self.training_batch['states']
        temp_rewards = self.training_batch['rewards']
        temp_terminal = self.training_batch['terminal']
        temp_actions = self.training_batch['actions']

        q_values_train_next = self.ac_model.predict_on_batch(temp_states_new)

        target = self.ac_model.predict_on_batch(temp_states)
        
        s_f = np.zeros((32, 4))
        t_f = np.zeros((32, 2))

        for i in range(self.batch_size):
            action_idx = int(temp_actions[i])
            if temp_terminal[i]:
                target[i][action_idx] = temp_rewards[i]
            else:
                target[i][action_idx] = temp_rewards[i] + 0.99 * np.amax(q_values_train_next[i])
            
            temp_s = np.asarray(temp_states[i])
            s_f[i,:] = temp_s.reshape(1,4)

            temp_t = np.asarray(target[i])
            t_f[i,:] = temp_t.reshape(1,2)

        self.ac_model.fit(s_f, t_f, verbose=0, callbacks=self.call)

    def train_q(self, update):
        self._random_sample()
        if update:
            self._update_target_model()
        self._train()

# Function
def epsi_greedy(action_space, q_values, epsilon):
    """Epsilon Greedy"""
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return np.argmax(q_values)


