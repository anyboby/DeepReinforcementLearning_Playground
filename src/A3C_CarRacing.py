import tensorflow as tf
import numpy as np
import Helper_Lib as helper

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as Keras

##########################
#Constants and Parameters#
##########################

#CartPole-v0
#CarRacing-v0
ENV = "CarRacing-v0"

RUN_TIME = 30
# THREADS = 8
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001 # thread delay is needed to enable more parallel threads than cpu cores

#discount rate
GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

DISCRETIZATION_RATIO = 3

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 4
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

MIN_BATCH = 32
LEARNING_RATE = 5e-3

# these values are basically weights in the overall sum of losses
LOSS_V = .5             # v loss coefficient
LOSS_ENTROPY = .01      # entropy coefficient  

############################

"""
The Environment runs the number of predefined episodes in a loop
Based on a gym environment containing an instance of the agent
"""
class Environment(threading.Thread):
    stop_signal = False
    
    def __init__(self, render=False, eps_start=EPS_START, eps_end = EPS_STOP, eps_steps = EPS_STEPS):
        threading.Thread.__init__(self)
        
        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)

        if isinstance(self.env.action_space, gym.spaces.Discrete): 
            print("Discrete Actionspace, size: {}".format(self.env.action_space.n))
            self.disc_action_space = self.env.action_space.n
            self.actions = [0,1]
            
        # hardcoded for Box(3,) in CarRacing-v0
        elif isinstance(self.env.action_space, gym.spaces.Box): 
            print("Box Actionspace, size: {}, ".format(self.env.action_space.shape[0]))
            self.disc_action_space = 4
            self.actions = [[0,1,0],[0,0,0.8],[-1,0,0],[1,0,0]]
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
        img = self.env.reset()
        img =  helper.rgb2gray(img, True)
        s = np.zeros(IMAGE_SIZE)
        for i in range(IMAGE_STACK):
            s[:,:,i] = img


        R = 0
        s_=s
        
        while True:
            time.sleep(THREAD_DELAY) #yield delay for safety
            
            if self.render: self.env.render()

            a = self.agent.act(s)  #action based on current state
            img_rgb, r, done, info = self.env.step(self.actions[a]) #retrieve the next state and reward for the corresponding action
            
            
            if not done:
                img =  helper.rgb2gray(img_rgb, True)
                for i in range(IMAGE_STACK-1):
                    s_[:,:,i] = s_[:,:,i+1]
                s_[:,:,IMAGE_STACK-1] = img
            
            else:    #last step of episode is finished, no next state
                s_ = None
            
            self.agent.train(s, a, r, s_) #let agent train with the information from step

    
            s = s_  #assume new state
            R += r  #add reward for the last step to total Rewards
        
            if done or self.stop_signal:
                break
            
        print ("Total R: {}".format(R))

            
        

"""
The Master Network delivers the policy and value function as output of the
neural network it contains
"""
class MasterNetwork:
    
    train_queue = [[],[],[],[],[]]  #s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    
    def __init__(self):
        self.session = tf.Session()
        Keras.set_session(self.session)
        Keras.manual_variable_initialization(True)
        
        #build model first
        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
    
        self.session.run (tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        
        self.default_graph.finalize()  #avoid modifications
    
    def _build_model(self):
        l_input = Input(shape = IMAGE_SIZE)
        x = l_input
        x = Convolution2D(16, (16,16), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (8,8), strides=(2,2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense (256, activation='relu')(x)
        l_dense = Dense (16, activation='relu')(x)
        
        #l_dense = Dense(16, activation='relu')(l_input)
        
        #actions need to have a correct probability distribution, hence the softmax activation
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_values = Dense(1, activation='linear')(l_dense)
        
        model = Model(inputs = [l_input], outputs=[out_actions, out_values])
        model._make_predict_function() #have to initialize before threading
        
        return model
            
    """
    builds a tf graph to define the loss functions so tf can solve them
    
    the policy loss function is the negation of the Objective function J:
    L_π = - (1/n) * ∑ [A(s_i,a_i) * log π(a_i|s_i)]
    
    the value loss used in this graph is the summed error square of our estimated Value V(s0) 
    towards the real Value V = r0 + γr1+γ2r2+...+γ(n−1)r(n−1)
    LV= (1/n) * ∑ [e_i²]
    """
    def _build_graph(self, model):
        # 2D array placeholders that hold a whole batch later when called in minimize()
        # first dimension is unlimited and represents training batches
        # second dimension is number of variables
        s_t = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  #discounted n-step reward
        
        # retrieve policy and value functions from Master Model
        # @MO determine dimensions of p and v
        p,v = model(s_t)
        
        # we need the probability of a certain action a, given state s 
        # therefore the probabilities p are multiplied with the hot_encoded vector a_t and sum-reduced
        # (axis 1 representing the dimension of different actions)which will leave us the 
        # exact probability of taking the action a (a-th index in p) given state s
        # the small constant added is to prevent NaN errors, if a probability was zero 
        # (possible through eps-greedy)
        log_prob = tf.log(tf.reduce_sum(p* a_t, axis=1, keep_dims = True) + 1e-10)
        
        # advantage for n-step reward, r_t holds the n-stepo return reward and approximates the 
        # action-state function Q(s,a) 
        advantage = r_t-v
        
        # policy loss according to above def. the advantage is regarded as constant 
        # and should not be included in tf gradient building. Averaging over the sum 
        # is done later in the code
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        
        # since Q(s,a) is approximated by n-step return reward r_t, the value error equals
        # the advantage function now!
        loss_value = LOSS_V * tf.square(advantage) 
                
        # maximize (@MO: minimize ?) entropy
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p+1e-10), axis=1, keep_dims = True)  
        
        # The previously skipped average-over-sum's in one step now
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        #@MO: what does RMSProp Optimizer do ? it allows manual learning rates but otherwise ?
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)
        
        return s_t, a_t, r_t, minimize
    
    
    """
    optimize preprocesses data and runs minimize() of MasterNetwork. optimize is called by 
    an optimizer, possibly multiple isntances of optimizer to handle incoming samples fast enough
    """
    def optimize(self):
        #make sure enough training samples are in queue, yield to other threads otherwise
        if len(self.train_queue[0])<MIN_BATCH:
            time.sleep(0) #yield
            return
        
        #@MO: WHY LOCK QUEUE, how is 'with' used in python ?
        # extract all samples from training queue with lock (for multithrading security)
        # 
        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:
                return
            
            # s_mask indicates whether s_ is a dummy inserted due to terminal state being reached,
            # contains 0 (isDummy) or 1 (isNotDummy)
            s,a,r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], []]
            
        # transform into blocks of numpy arrays
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)
        

        if len(s) > 5*MIN_BATCH:
            print ("Opimizer alert! Minimizing batch of {}".format(len(s)))
        
        # the reward received from the training queue is so far only immediate up to n-th step 
        # reward and missed (n * V(s_n) ). v is therefore first calculated starting from 
        # the latest state s_, discounted and added. 
        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask #set v to 0 where s_ is terminal state           

        # retrieve placeholders
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t:a, r_t:r})
        
        
    def train_push(self, s, a, r, s_):
        #@MO: what does this lock do ?
        with self.lock_queue:
            #queue s, a, and r into the training queue
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)
            
            # if the next state s_ is after last possible state, insert
            # dummy state for parallelism and flag it in queue[4]
            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
                
            else: 
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
            
            
    def predict(self, s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return p, v
        
    def predict_p(self, s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return p
    
    def predict_v(self, s):
        with self.default_graph.as_default():
            p,v = self.model.predict(s)
            return v      
    
"""
The optimizer calls MasterNetwork.optimize() endlessly in a loop, possibly from multiple threads
"""
class Optimizer(threading.Thread):
    
    stop_signal = False
    
    def __init__(self):
        threading.Thread.__init__(self)
    def run (self):
        while not self.stop_signal:
            masterNetwork.optimize()
    def stop(self):
        self.stop_signal = True

        
frames = 0
"""
The Agent is responsible for learning and determining the next action to take
based on the policy the master network has determined. The action is chosen stochastically
"""
class Agent:
    def __init__ (self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        
        
        self.memory = []  #used for n step return
        self.R = 0.
        
    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start)/self.eps_steps
        
    """
    act chooses an action based on given state and the policy - probabilty distribution 
    supplied by the master network 
    Additionally e-greedy policy is applied to enhance exploration in early stages
    """
    def act(self, s):
        epsilon = self.getEpsilon()
        global frames; frames = frames + 1
        
        if random.random() < epsilon:
            return random.randint(0, NUM_ACTIONS-1)
        
        else:
            s = np.array([s])
            p = masterNetwork.predict_p(s)[0]
            
            #a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)
            
            return a
        
    """
    train receives a set of samples including state, action, reward and the next state
    in order to send these to the memory, an array denoting the current actions position in 
    the total range of possible actions is created and sent to the memory
    """
    def train(self, s, a, r, s_): 
        
        """
        returns a tupel array containing the state, action reward and n-th final state
        depends on the Reward self.R to be calculated when called
        """
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n-1]
            
            return s, a, self.R, s_
            
        a_vec = np.zeros(NUM_ACTIONS)
        a_vec[a] = 1          #later needed, to access chosen action by easy multiplication
                              #Tensorflow does not allow indexed inputs
        
        self.memory.append ((s, a_vec, r, s_))
        
        self.R = (self.R + r * GAMMA_N) / GAMMA
        
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                masterNetwork.train_push(s, a, r, s_)
                
                #@MO DIEHIER NOCH CHECKEN
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0
        
        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            masterNetwork.train_push(s, a, r, s_)
            
            #@MO DIEHIER NOCH CHECKEN
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

	# possible edge case - if an episode ends in <N steps, the computation is incorrect


    # main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]

# check action space
print ("Env: {}, Action Space: ".format(ENV))
#env_test.disc_action_space()
#print(env_test.env.action_space.low)
NUM_ACTIONS = env_test.disc_action_space
NONE_STATE = np.zeros(NUM_STATE)

#create networks and threads
masterNetwork = MasterNetwork()

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

# start threads
for o in opts:
    o.start()
for e in envs:
    e.start()
time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()
    
for o in opts:
    o.stop()
for o in opts:
    o.join()
    
print("Training finished")
env_test.run()
