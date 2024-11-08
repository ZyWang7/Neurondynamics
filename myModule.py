from iznetwork import IzNetwork

import matplotlib.pyplot as plt
import random
import numpy as np
import bct


class myNetwork(object):

    def __init__(self, N, Dmax):
        self.net = IzNetwork(N, Dmax)
        # store the weights before scaling
        self.origin_W = np.zeros((N, N))
        # store the scaled weights
        self._W = np.zeros((N, N))
        self.rewire_p = 0
        self.sigma = 0
        self._V = []


    def set_Ex_to_Ex(self):
        # weight: 1
        ex_to_ex = np.zeros((800, 800))
        # 1000 randomly assigned one-way excitatoryto-excitatory connections in each module
        for k in range(8):
            connections = set()
            while len(connections) < 1000:
                i = random.randint(k*100, k*100+99)
                j = random.randint(k*100, k*100+99)
                # ensure that i != j to avoid self-connections and that (i, j) is unique
                if i != j and (i, j) not in connections:
                    connections.add((i, j))
                    ex_to_ex[i, j] = 1
        return ex_to_ex


    def set_Ex_to_In(self):
        # weight: random 0 - 1
        ex_to_in = np.zeros((800, 200))
        inhibitory_indices = list(range(200))  # 200 inhibitory neurons
        random.shuffle(inhibitory_indices)

        in_index = 0
        for k in range(8):
            cur_ex = list(range(k*100, (k+1)*100))
            random.shuffle(cur_ex)
            for i in range(100):
                ex_idx = cur_ex[i]
                in_idx = inhibitory_indices[in_index]
                ex_to_in[ex_idx][in_idx] = np.random.uniform(0, 1)
                if (i+1) % 4 == 0:
                    in_index += 1
        
        return ex_to_in
    

    def set_weights(self):
        # (800, 800)
        ex_to_ex = self.set_Ex_to_Ex()
        # (800, 200)
        ex_to_in = self.set_Ex_to_In()
        # (200, 800)
        in_to_ex = np.random.uniform(-1, 0, size=(200, 800))
        # (200, 200)
        in_to_in = np.random.uniform(-1, 0, size=(200, 200))
        # no self-connections
        for i in range(200):
            in_to_in[i][i] = 0
        
        ori_ex_to_all = np.concatenate((ex_to_ex, ex_to_in), axis=1)
        ori_in_to_all = np.concatenate((in_to_ex, in_to_in), axis=1)
        self.origin_W = np.concatenate((ori_ex_to_all, ori_in_to_all), axis=0)

        # concat to (1000, 1000)
        ex_to_all = np.concatenate((ex_to_ex*17, ex_to_in*50), axis=1)
        in_to_all = np.concatenate((in_to_ex*2, in_to_in), axis=1)
        weights = np.concatenate((ex_to_all, in_to_all), axis=0)
        
        self._W = weights
        self.net.setWeights(weights)


    def set_parameters(self):
        # 800 excitatory neurons
        ex_a = 0.02*np.ones(800)
        ex_b = 0.2*np.ones(800)
        ex_c = -65*np.ones(800)
        ex_d = 8*np.ones(800)

        # 200 inhibitory neurons
        in_a = 0.02*np.ones(200)
        in_b = 0.25*np.ones(200)
        in_c = -65*np.ones(200)
        in_d = 2*np.ones(200)

        # concat to (1000,)
        a = np.concatenate((ex_a, in_a))
        b = np.concatenate((ex_b, in_b))
        c = np.concatenate((ex_c, in_c))
        d = np.concatenate((ex_d, in_d))

        self.net.setParameters(a, b, c, d)


    def set_delays(self):
        ex_to_ex = np.random.randint(1, 21, size=(800, 800))
        ex_to_in = np.ones((800, 200), dtype=int)
        in_to_all = np.ones((200, 1000), dtype=int)

        ex_to_all = np.concatenate((ex_to_ex, ex_to_in), axis=1)
        D = np.concatenate((ex_to_all, in_to_all), axis=0)
        
        self.net.setDelays(D)


    def rewire(self, p):
        """ for each excitatory neuron in the network, with probability p, rewire 
        its connections to random neurons from the other module """
        self.rewire_p = p
        for k in range(8):
            # loop inside each module
            for i in range(k*100, (k+1)*100):
                for j in range(k*100, (k+1)*100):
                    # if there is a connection in the current neuron, rewire
                    if self._W [i, j] == 17:
                        if np.random.rand() < p:
                            self._W [i, j] = 0
                            self.origin_W[i, j] = 0
                            # select a random neuron index from other modules
                            if k == 0:
                                remain_idx = random.randint(100, 799)
                            elif k == 7:
                                remain_idx = random.randint(0, 699)
                            else:
                                # pick randomly from ranges outside the current module
                                if random.choice([True, False]):
                                    remain_idx = random.randint(0, k * 100 - 1)
                                else:
                                    remain_idx = random.randint((k + 1) * 100, 799)

                            self._W [i, remain_idx] = 17
                            self.origin_W[i, remain_idx] = 1
        self.net.setWeights(self._W)


    def plot_connectivity(self):
        plt.figure(figsize=(5, 5))
        y, x = np.where(self._W!=0)
        plt.scatter(x, y, s = 0.1)
        plt.gca().invert_yaxis()  # Invert the y-axis

        plt.title("Matrix Connectivity, p=" + str(self.rewire_p))
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.show()


    def get_small_world_index(self):
        A = np.abs(self.origin_W)
        N = A.shape[0]
        k = A.sum(axis=0).mean()
        pl_rand = np.log(N)/np.log(k)
        cc_rand = k/N

        # cc of weighted directed network
        cc = bct.clustering_coef_wd(A).mean()
        # pl = bct.charpath(bct.distance_wei(A)[1])[0]
        distance_matrix, _ = bct.distance_wei(A)  # Distance matrix
        pl = np.mean(distance_matrix[distance_matrix != np.inf])  # Characteristic path length

        self.sigma = (cc/cc_rand)/(pl/pl_rand)


    def set_all(self):
        self.set_weights()
        self.set_delays()
        self.set_parameters()
        self.get_small_world_index()


    def plot_raster(self, T):
        V = np.zeros((T, 1000))
        lambda_rate = 0.01  # Poisson rate per neuron per ms
        injected_current = 15  # Current to inject if Poisson process triggers

        for t in range(T):
            extra_I = np.zeros(1000)
            # For each neuron, generate a random number from a Poisson distribution
            # The Poisson distribution with lambda=0.01 will give a 1% chance of being > 0
            random_events = np.random.poisson(lambda_rate, 1000)

            # Inject current I=15 into neurons where the Poisson value is > 0
            extra_I[random_events > 0] = injected_current
            self.net.setCurrent(extra_I)
            self.net.update()
            V[t,:], _ = self.net.getState()

        V = V[:, :800]
        t, n = np.where(V > 29)
        self._V = V

        plt.figure(figsize=(10, 3))
        plt.scatter(t, n, s=5)
        plt.title("p=" + str(self.rewire_p) + ", small-world index =" + f'{self.sigma:.4f}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')
        plt.show()


    def plot_mean_fire_rate(self, window_size, step_size, num_windows):
        # Initialize an array to hold the mean firing rates (shape: (50, 8))
        mean_firing_rates = np.zeros((num_windows, 8))
        fire = self._V > 29
        # Loop over each time window
        for w in range(num_windows):
            start = w * step_size
            end = start + window_size
            
            # Compute the mean firing rate for each module in this window
            for k in range(8):
                # Select the neurons in the current module (columns of `V`)
                module_neurons = fire[start:end, k*100:(k+1)*100]
                # Calculate the mean firing rate over the selected window
                mean_firing_rates[w, k] = np.mean(module_neurons)

        # Plot the mean firing rate for each module
        time_points = np.arange(0, num_windows * step_size, step_size)  # Time points for x-axis

        plt.figure(figsize=(10, 3))
        for module in range(8):
            plt.plot(time_points, mean_firing_rates[:, module], label=f'Module {module + 1}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Firing Rate')
        plt.title('Mean Firing Rate in Each Module (p =' + str(self.rewire_p) + ')' )
        plt.legend()
        plt.show()

