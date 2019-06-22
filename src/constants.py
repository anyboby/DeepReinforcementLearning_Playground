
import numpy as np

##########################
#Constants and Parameters#
##########################

#CartPole-v0
#CarRacing-v0
ENV = "CarRacing-v0"

#150000 seconds for 1.7 days (currently about 10000 episodes)
#60000 seconds for 16.7 hours (about 4000 episodes)
#30000 seconds for 8.3 hours (currently about 2000 episodes)
RUN_TIME = 300
# THREADS = 8
THREADS = 2
OPTIMIZERS = 1
THREAD_DELAY = 0.0001 # thread delay is needed to enable more parallel threads than cpu cores

#discount rate
GAMMA = 0.99
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.7
EPS_STOP = .1
EPS_STEPS = 2000000
# eps_steps should be approx. steps*number of episodes (in this case 1000 steps)

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
IMAGE_STACK = 4
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

NUM_ACTIONS = 4
NONE_STATE = np.zeros(IMAGE_SIZE) #create Nullstate to append when s_ is None

########################
# Log & saving #
########################

DATA_FOLDER = "data_20_06_19"

LOG_FILE        =  DATA_FOLDER + "/tmp/a3c_log"
CHECKPOINT_DIR  =  DATA_FOLDER + "/checkpoints"

LOAD_PATH = DATA_FOLDER + "/carRacing_exp_e3700"
SAVE_FILE = DATA_FOLDER + "/carRacing_exp_savedata"

MIN_SAVE_REWARD = 60
SAVE_FRAMES = 50000
REPLAY_MODE = False

MIN_BATCH = 32
LEARNING_RATE = 5e-4

#RMSP Parameters
class RMSP:
    ALPHA       =  0.99      # decay parameter for RMSProp
    EPSILON     =  0.1       # epsilon parameter for RMSProp
    GRADIENT_NORM_CLIP  =  40.0      # Gradient clipping norm


# these values are basically weights in the overall sum of losses
LOSS_V = .5             # v loss coefficient
LOSS_ENTROPY = .01      # entropy coefficient  

############################
