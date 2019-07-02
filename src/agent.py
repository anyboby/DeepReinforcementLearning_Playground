import numpy as np
import random
import threading

from master_network import MasterNetwork

import constants as Constants

"""
The Agent is responsible for learning and determining the next action to take
based on the policy the master network has determined. The action is chosen stochastically
"""
class Agent:
    frames = 0
    def __init__ (self, master_network, eps_start=0.4, eps_end=0.15, eps_steps=75000, num_actions=Constants.NUM_ACTIONS):
        self.master_network = master_network
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.num_actions = num_actions
        
        
        self.memory = []  #used for n step return
        self.R = 0.
        
    def getEpsilon(self):
        if (Agent.frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + Agent.frames * (self.eps_end - self.eps_start)/self.eps_steps
        
    """
    act chooses an action based on given state and the policy - probabilty distribution 
    supplied by the master network 
    Additionally e-greedy policy is applied to enhance exploration in early stages
    """
    def act(self, s):
        epsilon = self.getEpsilon()
        #global frames; frames = frames + 1
        Agent.frames = Agent.frames + 1
        
        #save model
        #if Agent.frames%Constants.SAVE_FRAMES==0: self.save_model_weights(path = Constants.SAVE_FILE + "_e" + str(Agent.frames)[:-3])



        if random.random() < epsilon:
            #if Agent.frames%100==0: print("acting randomly: {}, frame nr: {}, eps: {}".format(a, Agent.frames, epsilon)) 
            uni_pi = [1/self.num_actions for i in range(0,self.num_actions)]    
            a = random.randint(0,self.num_actions-1)
            p = 1
            return a, uni_pi, p

        
        else:
            s = np.array([s])
            pi = self.master_network.predict_p(s)[0]

            #choose action with prob distribution of pi
            a = np.random.choice(range(len(pi)), p=pi)
            p = pi[a]

            #a = [i*p for i in Constants.DISC_ACTIONS[a_i]]

            return a, pi, p
        
    """
    train receives a set of samples including state, action, reward and the next state
    in order to send these to the memory, an array denoting the current actions position in 
    the total range of possible actions is created and sent to the memory
    """
    def train(self, s, a, r, s_): 
        GAMMA = Constants.GAMMA
        GAMMA_N = Constants.GAMMA_N

        """
        returns a tupel array containing the state, action, discounted rewards (self.R) and n-th final state after n steps
        """
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]
            
            return s, a, self.R, s_

        a_hot = np.zeros(self.num_actions)
        a_hot[a] = 1          #later needed, to access chosen action by easy multiplication 
        
        self.memory.append ((s, a_hot, r, s_))
        
        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, R, s_ = get_sample(self.memory, n)
                
                self.master_network.train_push(s, a, R, s_)

                #@MO DIEHIER NOCH CHECKEN
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0
        
        if len(self.memory) >= Constants.N_STEP_RETURN:
            s, a, R, s_ = get_sample(self.memory, Constants.N_STEP_RETURN)

            self.master_network.train_push(s, a, R, s_)
        
            self.memory.pop(0)    

            #### Interesting: recalculating R explicitly because the recursive version is numerically instable 
            #### to disurbances, they get propagated and amplified.
            #### e.g. for n = 4 now R becomes
            #### R = r_1*γ + r_2*γ²+r_3*γ³
            self.R = 0
            for i in range(Constants.N_STEP_RETURN-2):
                self.R = self.R + self.memory[i][2]*Constants.GAMMA**(i+1)


	# possible edge case - if an episode ends in <N steps, the computation is incorrect
