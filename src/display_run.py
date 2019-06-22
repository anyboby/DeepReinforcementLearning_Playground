import tensorflow as tf
import numpy as np
import sys, random
import gym

from master_network import MasterNetwork
from environment import Environment
from optimizer import Optimizer
from rmsprop_applier import RMSPropApplier

import constants as Constants

def choose_action(pi_values):
  #return np.random.choice(range(len(pi_values)), p=pi_values)
  return np.argmax(pi_values)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print ("Usage %s <checkpoint-name>" % sys.argv[0])

  else:
    # use CPU for display tool
    device = "/cpu:0"

    global_network = MasterNetwork(Constants.NUM_ACTIONS, device)

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = Constants.RMSP.ALPHA,
                                  momentum = 0.0,
                                  epsilon = Constants.RMSP.EPSILON,
                                  clip_norm = Constants.RMSP.GRADIENT_NORM_CLIP,
                                  device = device)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(sys.argv[1])
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
      print("Could not find old checkpoint")

    env = gym.make(Constants.ENV)
    env.monitor.start('/data/CarRacingEval')
    game_state = GameState(env, display=True, no_op_max=0)

    while True:
      pi_values = global_network.run_policy(sess, game_state.S)

      action = choose_action(pi_values)
      game_state.process(action)

      if game_state.terminal:
        env.monitor.close()
        break
        #game_state.reset()
      else:
        game_state.update()

