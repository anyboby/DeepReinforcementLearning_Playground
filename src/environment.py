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

    def __init__(self, env, agent, summary, saver, render=False, eps_start=Constants.EPS_START, eps_end = Constants.EPS_STOP, eps_steps = Constants.EPS_STEPS):
        threading.Thread.__init__(self)
        self.render = render
        self.env = env
        self.agent = agent

        self.disc_action_space = 4
        self.actions = [[0,1,0],[0,0,0.8],[-1,0,0],[1,0,0]]
        self.OWN_IMAGE_SIZE = Constants.IMAGE_SIZE
        self.OWN_IMAGE_STACK = Constants.IMAGE_STACK

        self.saver = saver
        self.saveData = saver.data
        self.local_t = 0
        self.maxEpReward = 0

        self.summary_writer = summary[0]
        self.summary_op = summary[1]
        self.score_input = summary[2]

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
        
    
    def run(self):
        """
        executes runEpisode as long as no sigint is received
        """
        while not self.stop_signal:
            
            diff_global_t = self.runEpisode()
            self.saveData.global_t += diff_global_t
            self.saver.saveIfRequested(self.agent.master_network.session)


            
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

        self.maxEpReward = 0
        R = 0
        s_=s
        start_local_t = self.local_t
        
        while True:
            time.sleep(Constants.THREAD_DELAY) #yield delay for safety
            
            if self.render: self.env.render()
            a, pi = self.agent.act(s)  #action based on current state
            with self.env_lock:
                img_rgb, r, done, info = self.env.step(self.actions[a]) #retrieve the next state and reward for the corresponding action

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

            self.local_t += 1

                    #adding in early termination
            if R > self.maxEpReward:
                self.maxEpReward  = R

            if self.maxEpReward - R > 10:
               done = True

            if done or self.stop_signal:

                print("score={}".format(R))
                self._record_score(self.agent.master_network.session, self.summary_writer, self.summary_op, self.score_input, R, self.saveData.global_t, pi)

                break

        print ("_________________________")
        print ("{} finished an episode".format(threading.currentThread().getName()))
        print ("Total R: {}".format(R))
        print ("_________________________")

        diff_local_t = self.local_t - start_local_t
        return diff_local_t

        
    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t, pi):
        summary_str = sess.run(summary_op, feed_dict={ score_input: score })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

        print ("****** ADDING NEW SCORE ******")
        self.saveData.append(score, pi)
        if score > Constants.MIN_SAVE_REWARD:
            self.saveData.requestSave()

    def rgb2gray(self, rgb, norm):
   
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray.astype("float32") / 128 - 1 

        return gray 