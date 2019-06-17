import numpy as np

from gym import wrappers
import gym, threading, time
import constants as Constants

"""
The Environment runs the number of predefined episodes in a loop
Based on a gym environment containing an instance of the agent
"""
class Environment(threading.Thread):
    stop_signal = False
    env_lock = threading.Lock()
    #### TODO: Gescheiten frame und episodecounter für saving erstellen
    global_episodes = 0

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
        #print(id(self.env))

        with self.env_lock:
            img = self.env.reset()

        Environment.global_episodes += 1    
        img =  self.rgb2gray(img, True)
        s = np.zeros(self.OWN_IMAGE_SIZE)
        for i in range(self.OWN_IMAGE_STACK):    
            s[:,:,i] = img

        R = 0
        s_=s
        
        while True:
            time.sleep(Constants.THREAD_DELAY) #yield delay for safety
            
            if self.render: self.env.render()
            a = self.agent.act(s)  #action based on current state
            with self.env_lock:
                img_rgb, r, done, info = self.env.step(self.actions[a]) #retrieve the next state and reward for the corresponding action

                ####### Experimental: chagne rewards #####
                #if r>0: r=4*r
                #print (info)


            if not done:
                img =  self.rgb2gray(img_rgb, True)
                for i in range(self.OWN_IMAGE_STACK-1):
                    s_[:,:,i] = s_[:,:,i+1]  # update stacked layers with the older stacked layers from previous step 
                s_[:,:,self.OWN_IMAGE_STACK-1] = img  # update newest picture on top of stack
            else:    #last step of episode is finished, no next state
                s_ = None

            #let agent put data in memory and possibly trigger optimization
            self.agent.train(s, a, r, s_) 

            s = s_  #assume new state
            R += r  #add reward for the last step to total Rewards

            if done or self.stop_signal:
                break
        print ("_________________________")
        print ("{} finished an episode".format(threading.currentThread().getName()))
        print ("Total R: {}".format(R))
        ######save model if Episode reward is high enough####
        if (R>Constants.MIN_SAVE_REWARD): 
            print ("JACKPOT: SAVING WEIGHTS")
            self.agent.save_model_weights(path = Constants.SAVE_PATH + "_e_" + str(Environment.global_episodes) + "_jackpot")
            wrappers.Monitor(self.env, Constants.SAVE_PATH)
        print ("_________________________")




    def rgb2gray(self, rgb, norm):
   
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray.astype('float32') / 128 - 1 

        return gray 