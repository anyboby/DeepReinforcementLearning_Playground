import numpy as np
import cv2

from gym import wrappers
import gym, threading, time
import constants as Constants
import tensorflow as tf

"""
The Environment runs the number of predefined episodes in a loop
Based on a gym environment containing an instance of the agent
"""
class Environment(threading.Thread):
    stop_signal = False
    env_lock = threading.Lock()
    global_episodes = 0
    maxScore = 0

    def __init__(self, env, agent, summary, saver, render=False, eps_start=Constants.EPS_START, eps_end = Constants.EPS_STOP, eps_steps = Constants.EPS_STEPS, cvshow = False):
        threading.Thread.__init__(self)
        self.render = render
        self.cvshow = cvshow
        self.env = env
        self.agent = agent

        self.OWN_IMAGE_SIZE = Constants.IMAGE_SIZE
        self.OWN_IMAGE_STACK = Constants.IMAGE_STACK

        self.saver = saver
        self.data = saver.data

        #local_t are frames, global_t are overall frames done from all envs etc.
        self.local_t = 0
        self.maxEpReward = 0

        self.summary_writer = summary[0]
        self.summary_op = summary[1]
        self.score_input = summary[2]
        #self.state_input = summary[3]
        self.weight_phs = summary[3]

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
            self.data.global_t += diff_global_t
            self.saver.saveIfRequested(self.agent.master_network.session)


            
    def stop(self):
        self.stop_signal = True
    
    def runEpisode(self):
        
        #### load and preprocess image ####
        #print(id(self.env))

        with self.env_lock:
            img = self.env.reset()

        img =  self._process_img(img)
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
            a, pi, p = self.agent.act(s)  #action based on current state
            action = [i*p for i in Constants.DISC_ACTIONS[a]]

            with self.env_lock:
                img_rgb, r, done, info = self.env.step(action) #retrieve the next state and reward for the corresponding action

            if not done:
                img =  self._process_img(img_rgb)

                for i in range(self.OWN_IMAGE_STACK-1):
                    s_[:,:,i] = s_[:,:,i+1]  # update stacked layers with the older stacked layers from previous step 
                s_[:,:,self.OWN_IMAGE_STACK-1] = img  # update newest picture on top of stack
            else:    #last step of episode is finished, no next state
                s_ = None

            #######   adding in early termination  ############
            R += r  #add reward for the last step to total Rewards
            if R > self.maxEpReward:
                self.maxEpReward  = R

            if self.maxEpReward - R > Constants.EARLY_TERMINATION:
                done = True
                s_ = None
                #r = -100
            ####################################################

            #clip rewards and send to agent
            #let agent put data in memory and possibly trigger optimization
            #r_cl = np.clip(r, -1, 1)

            self.agent.train(s, a, r, s_,) 

            #skip frames for speed
            if self.cvshow:
                cv2.imshow("image", s)
                cv2.waitKey(Constants.WAITKEY)

            if not done:
                #tensorboard epxects batchsize in first dimension, so add additional dim
                singleState = np.expand_dims(s, axis=0)

            s = s_  #assume new state
            self.local_t += 1

            if done or self.stop_signal:
                self._record_score(self.agent.master_network.session, self.summary_writer, self.summary_op, self.score_input, R, None, singleState, self.weight_phs, self.data.global_t, pi)
                Environment.global_episodes += 1
                break

        print ("_________________________")
        print ("{} finished an episode".format(threading.currentThread().getName()))
        print ("Total R: {}".format(R))
        if Environment.global_episodes%100==0:
            print (str(Environment.global_episodes) + " episodes haven been played so far!")
        print ("_________________________")

        diff_local_t = self.local_t - start_local_t
        return diff_local_t

        
    def _record_score(self, sess, summary_writer, summary_op, score_input, score, state_input,  state, weight_phs,  global_t, pi):
        master_network = self.agent.master_network
        # only traverse layers containing weights or biases 
        trainable_layers = [layer for layer in master_network.model.layers if len(layer.weights)!=0]
        weights = []
        for layer in trainable_layers:
            layer_weights, layer_biases = layer.get_weights()
            weights.append(layer_weights)
            weights.append(layer_biases)
        
        #create dictionary to feed to session with tf placeholders as keys and arrays of weights/biases as value
        #feed_dict = { score_input: score, state_input: state}
        feed_dict = { score_input: score}
        for i in range(0,len(weight_phs)):
            feed_dict.update({weight_phs[i].name:weights[i]})

        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        for summary in master_network.summary_strs:
            summary_writer.add_summary(summary, global_t)
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

        #print ("****** ADDING NEW SCORE ******")
        self.data.append(score, pi)

        if score > Environment.maxScore:
            Environment.maxScore = score
            self.saver.save(master_network.session)
        #elif score > Constants.MIN_SAVE_REWARD:
        #    self.data.requestSave()

    #####################
    ### Preprocessing ###
    #####################

    def _process_img(self, img):
        """
        preprocess a state which is given as : (96, 96, 3)
        """
        img = self._rgb2gray(img, True) # squash to 96 x 96
        #img = self._zero_center(img)
        img = self._crop(img)
        #img = cv2.resize(img, (40,40))
        #ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
        return img


    def _rgb2gray(self, rgb, norm):
   
        gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
        if norm:
            # normalize
            gray = gray.astype("float32") / 128 - 1 

        return gray 

    def _zero_center(self, img):
        #return np.divide(frame - 127, 127.0)
        return img - 127.0

    def _crop(self, img, length=12):
        """
        crop 96 by 96 to 84 by 84
        """
        h = len(img)
        w = len(img[0])
        return img[0:h - length, (int)(length/2):w - (int)(length/2)]
