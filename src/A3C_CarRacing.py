import tensorflow as tf

import numpy as np

import constants as Constants
from master_network import MasterNetwork
from optimizer import Optimizer
from agent import Agent
from environment import Environment

from keras.models import load_model

import gym, time, random, threading
           
####################
#  A3C CarRacing 
####################

# check action space
print ("Env: {}, Action Space: ".format(Constants.ENV))

master_network = MasterNetwork(replay_mode=Constants.REPLAY_MODE)
env_test = Environment(gym.make(Constants.ENV), Agent(master_network, Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS), render=True, eps_start=0., eps_end=0.)

info_file = open(Constants.INFO_PATH, "w")


#dont train if replaymode is set
if not Constants.REPLAY_MODE:

    #initialize threads
    envs = [Environment(gym.make(Constants.ENV), Agent(master_network, info_file , Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS)) for i in range(Constants.THREADS)]
    opts = [Optimizer(master_network) for i in range(Constants.OPTIMIZERS)]

    # start threads
    for o in opts:
        o.start()
    for e in envs:
        e.start()
    time.sleep(Constants.RUN_TIME)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()
        
    for o in opts:
        o.stop()
    for o in opts:
        o.join()
    
info_file.close()

print("Training finished")
env_test.run()
