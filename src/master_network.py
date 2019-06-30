import threading, time

import tensorflow as tf
from tensorflow import ConfigProto as cp

import constants as Constants
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras import backend as Keras

from savedata import SaveData




"""
The Master Network delivers the policy and value function as output of the
neural network it contains
"""
class MasterNetwork:
    
    train_queue = [[],[],[],[],[]]  #s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()
    lock_model = threading.Lock()

    def __init__(self, savedata):

        ## these lines are based on stackoverflow to avoid gpu memory overusage error (https://github.com/tensorflow/tensorflow/issues/24828)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=config)
        Keras.set_session(self.session)
        Keras.manual_variable_initialization(True)
        
        self.data = savedata

        #build model first
        self.model = self._build_model()

        self.graph = self._build_graph(self.model)
    
        self.session.run (tf.global_variables_initializer())

        self.default_graph = tf.get_default_graph()
        
        self.summary_strs = []


        #Keras.function(inputs=input_tensors, outputs=gradients)

        #if not replay_mode: self.default_graph.finalize()  #avoid modifications
    
    def _build_model(self):
        l_input = Input(shape = Constants.IMAGE_SIZE)   # input layer, shape:(?,96,96,4)
        x = l_input

        x = Convolution2D(16, (8,8), strides=(4,4), activation="relu")(x)      

        x = Convolution2D(32, (3,3), strides=(2,2), activation="relu")(x)      

        x = Flatten()(x)                      
    
        x = Dense (256, activation="relu")(x) 
                                    
        #l_dense = Dense (16, activation="relu")(x)
                                                    
            
        #actions need to have a correct probability distribution, hence the softmax activation
        out_actions = Dense(Constants.NUM_ACTIONS, activation="softmax")(x)  

        out_values = Dense(1, activation="linear")(x)              
                                                                            
                                                                        
        
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
        with tf.name_scope("loss_preparation"):      

            # array placeholders that hold a whole batch later when called in minimize()
            # first dimension is unlimited and represents training batches
            # second dimension is number of variables (e.g. image width, second dim is image height, fourth width is stack size)
            s_t = tf.placeholder(tf.float32, shape=(None, Constants.IMAGE_SIZE[0], Constants.IMAGE_SIZE[1], Constants.IMAGE_SIZE[2]), name = "state")
            a_t = tf.placeholder(tf.float32, shape=(None, Constants.NUM_ACTIONS), name="actions")
            r_t = tf.placeholder(tf.float32, shape=(None, 1), name = "rewards")  #discounted n-step reward
            
            # retrieve policy and value functions from Master Model
            pi,v = model(s_t)
            
            summaries = []

            #mainly for tensorboard recordings
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        

            # we need the probability of a certain action a, given state s 
            # therefore the probabilities p are multiplied with the hot_encoded vector a_t and sum-reduced
            # (axis 1 representing the dimension of different actions)which will leave us the 
            # exact probability of taking the action a (a-th index in p) given state s
            # the small constant added is to prevent NaN errors, if a probability was zero 
            # (possible through eps-greedy)
            #avoid NaN for zero probabilities
            pi_clipped = tf.clip_by_value(pi, 1e-20, 1.0)   
            log_prob = tf.log(tf.reduce_sum(pi_clipped* a_t, axis=1, keepdims = True) + 1e-10)
        
            # advantage for n-step reward, r_t holds the n-step return reward and approximates the 
            # action-state function Q(s,a) 
            advantage = r_t-v
            
            #summaries for tb            
            tf_v_summary = tf.summary.scalar("values", l2_norm(v))
            summaries.append(tf_v_summary)

            tf_r_summary = tf.summary.scalar("rewards", l2_norm(r_t))
            summaries.append(tf_r_summary)
            
            tf_adv_summary = tf.summary.scalar("advantage", l2_norm(advantage))
            summaries.append(tf_adv_summary)

            # policy loss according to above def. the advantage is regarded as constant 
            # and should not be included in tf gradient building. Averaging over the sum 
            # is done later in the code
            loss_policy = - log_prob * tf.stop_gradient(advantage)
            
            tf_ploss_summary = tf.summary.scalar("policy_loss", l2_norm(loss_policy))
            summaries.append(tf_ploss_summary)

            
            # since Q(s,a) is approximated by n-step return reward r_t, the value error equals
            # the advantage function now!
            #loss_value = Constants.LOSS_V * tf.square(advantage) 
            loss_value = Constants.LOSS_V * tf.square(advantage)

            tf_vloss_summary = tf.summary.scalar("value_loss", l2_norm(loss_value))
            summaries.append(tf_vloss_summary)

            # It’s useful to know that entropy for fully deterministic policy (e.g. [1, 0, 0, 0] 
            # for four actions) is 0 and it is maximized for totally uniform policy 
            # (e.g. [0.25, 0.25, 0.25, 0.25]).
            entropy = Constants.LOSS_ENTROPY * tf.reduce_sum(pi_clipped * tf.log(pi_clipped+1e-10), axis=1, keepdims = True)  
            


            # The previously skipped average-over-sum's in one step now
            loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
            tf_oloss_summary = tf.summary.scalar("overall_loss", loss_total)
            summaries.append(tf_oloss_summary)
            
        with tf.name_scope("train"):


            self.lr = tf.train.polynomial_decay(Constants.LEARNING_RATE, self.data.global_t, Constants.RUN_TIME, 0.000001, 1.0)
            tf_lr_summary = tf.summary.scalar("learnrate", self.lr)
            summaries.append(tf_lr_summary)

            #optimizer = tf.train.RMSPropOptimizer(self.lr,
            #                                        epsilon=Constants.RMSP.EPSILON,
            #                                        decay=Constants.RMSP.ALPHA)

            optimizer = tf.train.RMSPropOptimizer(self.lr,
                                                    epsilon=Constants.RMSP.EPSILON,
                                                    decay=0)
            #split training op def in two steps to get gradients for tensorboard
            grads_and_vars = optimizer.compute_gradients(loss_total)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, Constants.RMSP.GRADIENT_NORM_CLIP), var) for grad, var in grads_and_vars]
            minimize = optimizer.apply_gradients(clipped_grads_and_vars)
        
            #add l2 norm of grads and variables histogramms to tf summary
            for gradient, variable in clipped_grads_and_vars:
                if "dense_3" in variable.name or "dense_2" in variable.name:
                    tf_gradnorm_summary = tf.summary.scalar("grad_l2" + variable.name, l2_norm(gradient))
                    tf_weightnorm_summary = tf.summary.scalar(variable.name + "_l2", l2_norm(variable))
                    summaries = summaries + [tf_gradnorm_summary, tf_weightnorm_summary]

        return s_t, a_t, r_t, minimize, summaries

    """
    optimize preprocesses data and runs minimize() of MasterNetwork. optimize is called by 
    an optimizer, possibly multiple isntances of optimizer to handle incoming samples fast enough
    returns true, if it actually trained
    """
    def optimize(self, writesummaries = False):
        #make sure enough training samples are in queue, yield to other threads otherwise
        if len(self.train_queue[0])<Constants.MIN_BATCH:
            time.sleep(0) #yield
            return False
        
        # extract all samples from training queue with lock (for multithrading security)
        # 
        with self.lock_queue:
            if len(self.train_queue[0]) < Constants.MIN_BATCH:
                return False
            
            # s_mask indicates whether s_ is a dummy inserted due to terminal state being reached,
            # contains 0 (isDummy) or 1 (isNotDummy)
            s,a,r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], []]
            
        # transform into blocks of numpy arrays
        # reshape arrays from row vectors to colums (vertical) vectors
        s = np.array(s)
        a = np.vstack(a)
        r = np.vstack(r)

        #print (str(r))
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


        # retrieve placeholders including summary ops
        s_t, a_t, r_t, minimize, summaries = self.graph

        if writesummaries:
            
            try:
                #run minimization + tb summaries
                results = self.session.run([minimize] + summaries, feed_dict={s_t: s, a_t:a, r_t:r})

                #leave out result from minimize run
                self.summary_strs = self.summary_strs + results[1:]
            except:
                print ("s_t: " + str(s))
                print ("a_t: " + str(a))
                print ("r_t: " + str(r))

        else: 
            #print (r)
            #run minimization only
            try:
                self.session.run(minimize, feed_dict={s_t: s, a_t:a, r_t:r})
            except:
                print ("s_t: " + str(s))
                print ("a_t: " + str(a))
                print ("r_t: " + str(r))
        return True






    def train_push(self, s, a, r, s_):
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
            p,_ = self.model.predict(s)
            return p
    
    def predict_v(self, s):
        with self.default_graph.as_default():
            _,v = self.model.predict(s)
            return v      

    def init_tf_summary(self):
        
        #score scalars
        score_input = tf.placeholder(tf.int32)
        tf_score_summary = tf.summary.scalar("score", score_input)
        
        #state images
        #state_input = tf.placeholder(tf.float32, shape=(None, Constants.IMAGE_SIZE[0], Constants.IMAGE_SIZE[1], Constants.IMAGE_SIZE[2]), name = "state")
        #tf_image_summary = tf.summary.image("state", state_input)

        #add placeholder-histogramms to all trainable weights in the model
        weight_phs = ()
        tf_weight_summaries = []
        for trainable_weight in self.model.trainable_weights:
            weight_ph = tf.placeholder(tf.float32, shape=trainable_weight.shape)
            tf_weight_summary = tf.summary.histogram(trainable_weight.name, weight_ph)
            weight_phs = weight_phs + (weight_ph,)
            tf_weight_summaries.append(tf_weight_summary)


        #summary_op      =  tf.summary.merge([tf_score_summary, tf_image_summary]+tf_weight_summaries)
        summary_op      =  tf.summary.merge([tf_score_summary]+tf_weight_summaries)
        summary_writer  =  tf.summary.FileWriter(Constants.LOG_FILE, self.session.graph)

        #return summary_writer, summary_op, score_input, state_input, weight_phs
        return summary_writer, summary_op, score_input, weight_phs



