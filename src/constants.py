
import numpy as np

##########################
#Constants and Parameters#
##########################

#CartPole-v0
#CarRacing-v0
ENV = "CarRacing-v0"


#approximate speed: 1000frames/15seconds = 66frames/second
# 2851200 frames / 12 hours
# 1426100 frames / 6 hours
# 713050 frames / 3 hours
# 4000 frames / minute

# run_time in global frames
RUN_TIME = 2851200
# THREADS = 8
THREADS =1
OPTIMIZERS = 1
THREAD_DELAY = 0.0001 # thread delay is needed to enable more parallel threads than cpu cores

#discount rate
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

#### leave this out
EPS_START = .4
EPS_STOP = .05
EPS_STEPS = 2851200
# eps_steps should be approx. steps*number of episodes (in this case 1000 steps)

IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
IMAGE_STACK = 4
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_STACK)

NUM_ACTIONS = 5
DISC_ACTIONS = [[0,1,0],[0,0,0.8],[-1,0,0],[1,0,0],[0,0,0]]
NONE_STATE = np.zeros(IMAGE_SIZE) #create Nullstate to append when s_ is None
EARLY_TERMINATION = 15 # score difference between epMax and current score for termination
SUMMARY_STEPS = 100

########################
# Log & saving #
########################

DATA_FOLDER = "data_2_07_19_noRclip_et1000_feature256"

LOG_FILE        =  DATA_FOLDER + "/tmp/a3c_log_transfer"
CHECKPOINT_DIR  =  DATA_FOLDER + "/checkpoints_copy"
SAVE_FILE = DATA_FOLDER + "/carRacing_savedata"

MIN_SAVE_REWARD = 100
SAVE_FRAMES = 50000
REPLAY_MODE = False
WAITKEY = 1 #1 for 1ms and 0 for indefinite wait til key stroke

MIN_BATCH = 32
LEARNING_RATE = 1e-4
FEATURE_LAYER = 256

#RMSP Parameters
class RMSP:
    ALPHA       =  0.9      # decay parameter for RMSProp
    EPSILON     =  1e-10      # epsilon parameter for RMSProp
    GRADIENT_NORM_CLIP  =  40.0      # Gradient clipping norm


class ADAM:
    USE_LOCKING = True

# these values are basically weights in the overall sum of losses
LOSS_V = .5           # v loss coefficient
LOSS_ENTROPY = .01      # entropy coefficient  

############################
