from iznetwork import IzNetwork

import matplotlib.pyplot as plt
import random
import numpy as np


class myNetwork(object):
    """
    This class is used to simulate small-world modular networks of Izhikevich
    neurons, follow the description of the experiment in Dynamical Complexity 
    Topic.
    """

    def __init__(self, num_ex, num_in, num_ex_module, Dmax):
        """
        Initialise network with given number of neurons and maximum transmission
        delay.

        Inputs:
        num_ex  -- Number of excitatory neurons in total in the network.
        num_in  -- Number of inhibitory neurons in total in the network.
        num_ex_module -- number of excitatory modules.
        Dmax    -- Maximum delay in all the synapses in the network, in ms. Any
                   longer delay will result in failing to deliver spikes.
        """
        self.num_ex = num_ex
        self.num_in = num_in
        self.num_ex_module = num_ex_module
        self.N = self.num_ex + self.num_in      # total number of neurons

        self.net = IzNetwork(self.N, Dmax)      # the network object
        self._W = np.zeros((self.N, self.N))    # the scaled weights
        self.rewire_p = 0   # the probability of rewiring
        self._V = []        # the membrane potential of the neurons


    def set_Ex_to_Ex(self):
        """ 
        Creates a connectivity matrix between excitatory neurons within each
        module. Establishes 1000 unique, randomly assigned connections within 
        each module, ensuring no self-connections.

        Returns:
            ex_to_ex -- np.array. The connectivity matrix for excitatory-to-
                        excitatory connections.
        """
        # initializes a zero matrix
        ex_to_ex = np.zeros((self.num_ex, self.num_ex))
        # x -> the number of excitatory neurons in each module
        x = self.num_ex // self.num_ex_module
        for k in range(self.num_ex_module):
            connections = set()     # track unique connections within the module
            # 1000 randomly assigned one-way excitatoryto-excitatory connections in each module
            while len(connections) < 1000:
                i = random.randint(k * x, (k+1) * x - 1)
                j = random.randint(k * x, (k+1) * x - 1)
                # ensure no self-connections and uniqueness
                if i != j and (i, j) not in connections:
                    connections.add((i, j))
                    # weight: 1
                    ex_to_ex[i, j] = 1
        return ex_to_ex


    def set_Ex_to_In(self):
        """
        Set up the connection between excitatory and inhibitory neurons. Each 
        inhibitory neuron has connections from exactly four excitatory neurons 
        (all within the same module). Each connection is randomly assigned a 
        weight between 0 and 1.

        Returns:
            ex_to_in -- np.array. The connectivity matrix for excitatory-to-
                        inhibitory connections.
        """
        # initializes a zero matrix
        ex_to_in = np.zeros((self.num_ex, self.num_in))
        inhibitory_indices = list(range(self.num_in))  # 200 inhibitory neurons
        # shuffled to ensure random distribution of inhibitory targets
        random.shuffle(inhibitory_indices)

        # x -> calculate the number of excitatory neurons in each module
        x = self.num_ex // self.num_ex_module
        in_index = 0
        for k in range(self.num_ex_module):
            # shuffled the excitatory neuron indices for `k` th module
            cur_ex = list(range(k*x, (k+1)*x))
            random.shuffle(cur_ex)
            for i in range(x):
                ex_idx = cur_ex[i]
                in_idx = inhibitory_indices[in_index]
                ex_to_in[ex_idx][in_idx] = np.random.uniform(0, 1)
                # incremented every four excitatory connections
                if (i+1) % 4 == 0:
                    in_index += 1
        
        return ex_to_in
    

    def set_weights(self):
        """
        Constructs the complete synaptic weight matrix by defining, combining 
        and scaling the connection matrices for different type of neuron-to
        -neuron interaction.

        The concatenated weight matrix has dimensions `(1000, 1000)`
        - the first 800 rows represent connections involving excitatory neurons.
        - the last 200 rows represent connections involving inhibitory neurons.
        """
        # (800, 800) excitatory-to-excitatory 
        ex_to_ex = self.set_Ex_to_Ex()
        # (800, 200) excitatory-to-inhibitory 
        ex_to_in = self.set_Ex_to_In()
        # (200, 800) inhibitory-to-excitatory
        in_to_ex = np.random.uniform(-1, 0, size=(self.num_in, self.num_ex))
        # (200, 200) inhibitory-to-inhibitory
        in_to_in = np.random.uniform(-1, 0, size=(self.num_in, self.num_in))
        # no self-connections
        for i in range(self.num_in):
            in_to_in[i][i] = 0

        # concat to (1000, 1000) and add the scales
        ex_to_all = np.concatenate((ex_to_ex*17, ex_to_in*50), axis=1)
        in_to_all = np.concatenate((in_to_ex*2, in_to_in), axis=1)
        weights = np.concatenate((ex_to_all, in_to_all), axis=0)
        
        self._W = weights
        self.net.setWeights(weights)


    def set_parameters(self):
        """
        Initializes and sets the parameters with some variation for excitatory 
        and inhibitory neurons.
        """
        # 800 excitatory neurons
        ex_r = np.random.rand(self.num_ex)
        ex_a = 0.02*np.ones(self.num_ex)
        ex_b = 0.2*np.ones(self.num_ex)
        ex_c = -65 + 15*(ex_r**2)
        ex_d = 8 - 6*(ex_r**2)

        # 200 inhibitory neurons
        in_r = np.random.rand(self.num_in)
        in_a = 0.02 + 0.08 * in_r
        in_b = 0.25 - 0.05 * in_r
        in_c = -65*np.ones(self.num_in)
        in_d = 2*np.ones(self.num_in)

        # concat to (1000,)
        a = np.concatenate((ex_a, in_a))
        b = np.concatenate((ex_b, in_b))
        c = np.concatenate((ex_c, in_c))
        d = np.concatenate((ex_d, in_d))

        self.net.setParameters(a, b, c, d)


    def set_delays(self):
        """
        Sets up the delay matrix by defining and combining the delay matrices 
        for different type of neuron-to-neuron interaction.
        """
        # (800, 800) Random 1ms to 20ms
        ex_to_ex = np.random.randint(1, 21, size=(self.num_ex, self.num_ex))
        # (800, 200)
        ex_to_in = np.ones((self.num_ex, self.num_in), dtype=int)
        # (200, 1000)
        in_to_all = np.ones((self.num_in, self.N), dtype=int)
        # (800, 1000)
        ex_to_all = np.concatenate((ex_to_ex, ex_to_in), axis=1)

        # concat to (1000, 1000)
        D = np.concatenate((ex_to_all, in_to_all), axis=0)
        self.net.setDelays(D)


    def rewire(self, p):
        """ 
        Rewires connections between excitatory neurons within each module with a
        probability p. Remove an existing connection and create a new one to a 
        random excitatory neuron in a different module if rewiring occurs.

        Input:
            p -- float. The probability of rewiring a connection.
        """
        self.rewire_p = p
        # x -> the number of excitatory neurons in each module
        x = self.num_ex // self.num_ex_module

        for k in range(self.num_ex_module):
            # loop inside each module
            for i in range(k*x, (k+1)*x):
                for j in range(k*x, (k+1)*x):
                    # if there is a connection in the current neuron, rewire
                    if self._W [i, j] == 17:
                        if np.random.rand() < p:
                            # remove the old connection
                            self._W [i, j] = 0
                            # select a random neuron index from other modules
                            if k == 0:
                                remain_idx = random.randint(x, self.num_ex - 1)
                            elif k == self.num_ex_module - 1:
                                remain_idx = random.randint(0, k * x - 1)
                            else:
                                # pick randomly from ranges outside the current module
                                if random.choice([True, False]):
                                    remain_idx = random.randint(0, k * x - 1)
                                else:
                                    remain_idx = random.randint((k + 1) * x, self.num_ex - 1)
                            # create a new connection
                            self._W [i, remain_idx] = 17
        # update the network's weights
        self.net.setWeights(self._W)


    def plot_connectivity(self):
        """ 
        Plots the connectivity matrix of the network.
        Non-zero entries in `self._W` represent connections between neurons, and 
        their positions in the matrix are used to create a scatter plot that 
        shows the structure of connectivity across the network.
        """
        plt.figure(figsize=(5, 5))
        # identify the coords of all non-zero entries -> connections exist
        y, x = np.where(self._W!=0)
        plt.scatter(x, y, s = 0.1)
        plt.gca().invert_yaxis()  # Invert the y-axis

        plt.title("Matrix Connectivity, p=" + str(self.rewire_p))
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.show()


    def set_all(self):
        """ 
        Initializes and sets up the entire network by configuring weights, 
        delays, and neuron parameters.
        (high-level initializer)
        """
        self.set_weights()
        self.set_delays()
        self.set_parameters()


    def plot_raster(self, T):
        """ 
        Generates and plots a raster plot of neuron firing activity over a 
        `T`-millisecond simulation.

        Inputs:
            T -- int. The number of milliseconds to run the simulation.
        """
        V = np.zeros((T, self.N))
        lambda_rate = 0.01      # poisson rate per neuron per ms
        injected_current = 15   # value of current to inject

        for t in range(T):
            extra_I = np.zeros(self.N)
            # for each neuron, generate a random number from a Poisson 
            # distribution with lambda=0.01 -> give a 1% chance of being > 0
            random_events = np.random.poisson(lambda_rate, self.N)
            # inject current I=15 into neurons where the Poisson value is > 0
            extra_I[random_events > 0] = injected_current

            self.net.setCurrent(extra_I)
            self.net.update()
            V[t,:], _ = self.net.getState()     # stores the resulting state

        # extracts excitatory neurons
        V = V[:, :self.num_ex]
        # identifies time and neuron indices where firing events occur
        t, n = np.where(V > 29)
        self._V = V

        plt.figure(figsize=(10, 3))
        plt.scatter(t, n, s=5)
        plt.title("p=" + str(self.rewire_p))
        plt.xticks(np.arange(0, 1000, step=100))
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.show()


    def plot_mean_fire_rate(self, window_size, step_size, num_windows):
        """
        Plots the mean firing rate in each of the eight modules over a 1000 ms 
        run, using a sliding time window.
        Input:
            window_size -- The size of each window in ms.
            step_size   -- Time shift (in ms) between consecutive windows.
            num_windows -- Total number of windows to calculate mean firing
                           rates over.
        """
        # initialize an array to hold the mean firing rates (shape: (50, 8))
        mean_firing_rates = np.zeros((num_windows, 8))
        fire = self._V > 29
        # loop over each time window
        for w in range(num_windows):
            start = w * step_size
            end = start + window_size
            
            # compute the mean firing rate for each module in this window
            x = self.num_ex // self.num_ex_module
            for k in range(self.num_ex_module):
                # select the neurons in the current module (columns of `V`)
                module_neurons = fire[start:end, k * x:(k+1) * x]
                # calculate the mean firing rate over the selected window
                mean_firing_rates[w, k] = np.mean(module_neurons)

        # plot the mean firing rate for each module
        time_points = np.arange(0, num_windows * step_size, step_size)  # time points for x-axis

        plt.figure(figsize=(10, 3))
        for module in range(8):
            plt.plot(time_points, mean_firing_rates[:, module], label=f'Module {module + 1}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Firing Rate')
        plt.xticks(np.arange(0, 1000, step=100))
        plt.title('Mean Firing Rate in Each Module (p =' + str(self.rewire_p) + ')' )
        plt.legend()
        plt.show()

