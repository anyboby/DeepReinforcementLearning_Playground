import threading, time

import tensorflow as tf
from tensorflow import ConfigProto as cp

import constants as Constants

from keras.models import *
from keras.layers import *
from keras import backend as Keras

"""
The Master Network delivers the policy and value function as output of the
neural network it contains
"""
class MasterNetwork:
    
    train_queue = [[],[],[],[],[]]  #s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    lock_model = threading.Lock()

    def __init__(self, replay_mode = False):

        ## these lines are based on stackoverflow to avoid gpu memory overusage error (https://github.com/tensorflow/tensorflow/issues/24828)
        self.replay_mode = replay_mode
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)
        Keras.set_session(self.session)
        Keras.manual_variable_initialization(True)
        
        #build model first
        if not replay_mode: self.model = self._build_model()
        else:  self.model = load_model(Constants.SAVE_PATH) #+ "_jackpot" oder + "_<frameNumer>"

        self.graph = self._build_graph(self.model)
    
        self.session.run (tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        
        if not replay_mode: self.default_graph.finalize()  #avoid modifications
    
    def _build_model(self):
        l_input = Input(shape = Constants.IMAGE_SIZE)   # input layer, shape:(?,96,96,4)
        x = l_input
        x = Convolution2D(16, (8,8), strides=(4,4), activation='relu')(x)               
                                                                                           
        x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)      
                                                                                           
        x = Flatten()(x)                      
        x = Dense (256, activation='relu')(x) 
                                        
        l_dense = Dense (16, activation='relu')(x)
                                                      
        
        #l_dense = Dense(16, activation='relu')(l_input)
        
        #actions need to have a correct probability distribution, hence the softmax activation
        out_actions = Dense(Constants.NUM_ACTIONS, activation='softmax')(l_dense)  #output dense layer for actions: 
                                                                            # kernel shape: (16,NUM_ACTIONS = 4 as of now)
                                                                            # outputshape: (?, 4)

        out_values = Dense(1, activation='linear')(l_dense)              #output dense layer for values: 
                                                                            # kernel shape: (16,1)
                                                                            # outputshape: (?, 1)
        
        model = Model(inputs = [l_input], outputs=[out_actions, out_values])
        model._make_predict_function() #have to initialize before threading
        print("Model Built, Summary:")
        model.summary()

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
        s_t = tf.placeholder(tf.float32, shape=(None, Constants.IMAGE_SIZE[0], Constants.IMAGE_SIZE[1], Constants.IMAGE_SIZE[2]))
        a_t = tf.placeholder(tf.float32, shape=(None, Constants.NUM_ACTIONS))
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
        log_prob = tf.log(tf.reduce_sum(p* a_t, axis=1, keepdims = True) + 1e-10)
        
        # advantage for n-step reward, r_t holds the n-stepo return reward and approximates the 
        # action-state function Q(s,a) 
        advantage = r_t-v
        
        # policy loss according to above def. the advantage is regarded as constant 
        # and should not be included in tf gradient building. Averaging over the sum 
        # is done later in the code
        loss_policy = - log_prob * tf.stop_gradient(advantage)
        
        # since Q(s,a) is approximated by n-step return reward r_t, the value error equals
        # the advantage function now!
        #loss_value = Constants.LOSS_V * tf.square(advantage) 
        loss_value = Constants.LOSS_V * tf.nn.l2_loss(advantage)
        
        # maximize (@MO: minimize ?) entropy
        # It’s useful to know that entropy for fully deterministic policy (e.g. [1, 0, 0, 0] 
        # for four actions) is 0 and it is maximized for totally uniform policy 
        # (e.g. [0.25, 0.25, 0.25, 0.25]).
        entropy = Constants.LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p+1e-10), axis=1, keepdims = True)  
        
        # The previously skipped average-over-sum's in one step now
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        
        #@MO: what does RMSProp Optimizer do ? it allows manual learning rates but otherwise ?
        optimizer = tf.train.RMSPropOptimizer(Constants.LEARNING_RATE,
                                                 epsilon=Constants.RMSP.EPSILON,
                                                 decay=Constants.RMSP.ALPHA)
        minimize = optimizer.minimize(loss_total)
        
        return s_t, a_t, r_t, minimize

    """
    optimize preprocesses data and runs minimize() of MasterNetwork. optimize is called by 
    an optimizer, possibly multiple isntances of optimizer to handle incoming samples fast enough
    """
    def optimize(self):
        #make sure enough training samples are in queue, yield to other threads otherwise
        if len(self.train_queue[0])<Constants.MIN_BATCH:
            time.sleep(0) #yield
            return
        
        #@MO: WHY LOCK QUEUE, how is 'with' used in python ?
        # extract all samples from training queue with lock (for multithrading security)
        # 
        with self.lock_queue:
            if len(self.train_queue[0]) < Constants.MIN_BATCH:
                return
            
            # s_mask indicates whether s_ is a dummy inserted due to terminal state being reached,
            # contains 0 (isDummy) or 1 (isNotDummy)
            s,a,r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], []]
            
        # transform into blocks of numpy arrays
        #print("shape of s[0] : {}".format(s[0].shape))
        #print("shape of np.array(s) : {}".format(np.array(s).shape))
        s = np.array(s)   # new shape of a: (32,96,96,4)
        a = np.vstack(a)  # new shape of a: (32,4)
        r = np.vstack(r)  # new shape of r: (32,1)
        #s_ = np.vstack(s_) # new shape of s_: (32,96,96,4)
        #print("shape of s_[0] : {}".format(s_[0].shape))
        #print("shape of np.array(s_) : {}".format(np.array(s_).shape))
        s_ = np.array(s_)
        s_mask = np.vstack(s_mask) # new shape of s_mask: (32,1)
        

        if len(s) > 5*Constants.MIN_BATCH:
            print ("Opimizer alert! Minimizing batch of {}".format(len(s)))
        
        # the reward received from the training queue is so far only immediate up to n-th step 
        # reward and missed (n * V(s_n) ). v is therefore first calculated starting from 
        # the latest state s_, discounted and added. 
        v = self.predict_v(s_)
        r = r + Constants.GAMMA_N * v * s_mask #set v to 0 where s_ is terminal state           

        # retrieve placeholders
        s_t, a_t, r_t, minimize = self.graph
        #print("Learning them weightz")
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
                self.train_queue[3].append(Constants.NONE_STATE)
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

    def save_weights(self, path):
        with self.lock_model:
            print("saving...")
            self.model.save(path)


    def load_weights(self):
        with self.lock_model:
            print("saving...")
            self.model = load_model(Constants.SAVE_PATH)

