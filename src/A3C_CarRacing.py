import tensorflow as tf

import numpy as np

import constants as Constants
from master_network import MasterNetwork
from optimizer import Optimizer
from agent import Agent
from environment import Environment

import gym, time, random, threading
           
#frames = 0

#create networks and threads
master_network = MasterNetwork()
    
# main
env_test = Environment(gym.make(Constants.ENV),master_network , render=False, eps_start=0., eps_end=0.)
STATE_SHAPE = env_test.env.observation_space.shape

# check action space
print ("Env: {}, Action Space: ".format(Constants.ENV))
#env_test.disc_action_space()
#print(env_test.env.action_space.low)

#initialize threads
envs = [Environment(gym.make(Constants.ENV), Agent(master_network, Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS)) for i in range(Constants.THREADS)]
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
    
print("Training finished")
env_test.run()
