
import numpy as np

##########################
#Constants and Parameters#
##########################

#CartPole-v0
#CarRacing-v0
ENV = "CarRacing-v0"

RUN_TIME = 30000
# THREADS = 8
THREADS = 4
OPTIMIZERS = 2
THREAD_DELAY = 0.001 # thread delay is needed to enable more parallel threads than cpu cores

#discount rate
GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 4
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

NUM_ACTIONS = 4
NONE_STATE = np.zeros(IMAGE_SIZE) #create Nullstate to append when s_ is None


MIN_BATCH = 32
LEARNING_RATE = 5e-3

# these values are basically weights in the overall sum of losses
LOSS_V = .5             # v loss coefficient
LOSS_ENTROPY = .01      # entropy coefficient  

############################
