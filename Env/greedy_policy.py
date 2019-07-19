import random
import numpy as np

class GreedyPolicy():

    def __init__(self, n_outputs,n_steps_annealing=1000000, min_epsilon=0.1, max_epsilon=1,seed=None):
        self.n_outputs=n_outputs
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.n_steps_annealing = n_steps_annealing
        self.seed=seed
        random.seed(a=self.seed)

    def generate(self, q_values, step):
        epsilon = max(self.min_epsilon, self.max_epsilon-(self.max_epsilon-self.min_epsilon)/self.n_steps_annealing * step )
        if random.random() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return np.argmax(q_values)
        
if __name__=='__main__':

    a=GreedyPolicy(2)
    for step in range(10):
        print(a.generate([1,0],step))


#    import matplotlib.pyplot as plt
#
#    plt.plot(b)
#    plt.show()
    print(a.generate(np.array((1,0)),1000000))