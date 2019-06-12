import numpy as np

import gym, threading, time
import constants as Constants

"""
The Environment runs the number of predefined episodes in a loop
Based on a gym environment containing an instance of the agent
"""
class Environment(threading.Thread):
    stop_signal = False
    env_lock = threading.Lock()

    def __init__(self, env, agent, render=False, eps_start=Constants.EPS_START, eps_end = Constants.EPS_STOP, eps_steps = Constants.EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = env
        self.agent = agent
        self.disc_action_space = 4  
        self.actions = [[0,1,0],[0,0,0.8],[-1,0,0],[1,0,0]]
        self.OWN_IMAGE_SIZE = Constants.IMAGE_SIZE
        self.OWN_IMAGE_STACK = Constants.IMAGE_STACK

        """ disabled for multithreading testing
        if isinstance(self.env.action_space, gym.spaces.Discrete): 
            print("Discrete Actionspace, size: {}".format(self.env.action_space.n))
            self.disc_action_space = self.env.action_space.n
            self.actions = [0,1]
        """         
        # hardcoded for Box(3,) in CarRacing-v0
        """ disabled for multithreading testing
        elif isinstance(self.env.action_space, gym.spaces.Box): 
            print("Box Actionspace, size: {}, ".format(self.env.action_space.shape[0]))
            self.disc_action_space = 4  
            self.actions = [[0,1,0],[0,0,0.8],[-1,0,0],[1,0,0]]
        """
        
    """
    executes runEpisode as long as no sigint is received
    """    
    def run(self):
        while not self.stop_signal:
            self.runEpisode()
            
    def stop(self):
        self.stop_signal = True
    
    def runEpisode(self):
        
        #### load and preprocess image ####
        print("test")
        #print(id(self.env))
        print(id(self.env.reset))

        with self.env_lock: 
            img = self.env.reset()
        print("test2")
        img =  self.rgb2gray(img, True)
        print("test3")
        s = np.zeros(self.OWN_IMAGE_SIZE)
        print("test4")
        for i in range(self.OWN_IMAGE_STACK):    
            s[:,:,i] = img

        R = 0
        s_=s
        
        while True:
            time.sleep(Constants.THREAD_DELAY) #yield delay for safety
            
            if self.render: self.env.render()

            a = self.agent.act(s)  #action based on current state
            img_rgb, r, done, info = self.env.step(self.actions[a]) #retrieve the next state and reward for the corresponding action
            
            if not done:
                img =  self.rgb2gray(img_rgb, True)
                for i in range(self.OWN_IMAGE_STACK-1):
                    s_[:,:,i] = s_[:,:,i+1]  # update stacked layers with the older stacked layers from previous step 
                s_[:,:,self.OWN_IMAGE_STACK-1] = img  # update newest picture on top of stack
            else:    #last step of episode is finished, no next state
                s_ = None
            
            self.agent.train(s, a, r, s_) #let agent train with the information from step
    
            s = s_  #assume new state
            R += r  #add reward for the last step to total Rewards
        
            if done or self.stop_signal:
                break
            
        print ("Total R: {}".format(R))

    def rgb2gray(self, rgb, norm):
   
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray.astype('float32') / 128 - 1 

        return gray 