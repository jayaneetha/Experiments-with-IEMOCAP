import argparse

import os
import tensorflow as tf
from keras import Input
from keras.optimizers import Adam

import models
from constants import NUM_MFCC, NO_features
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, WandbLogger
from rl.memory import SequentialMemory
from rl_custom_policy import CustomPolicyBasedOnMaxBoltzmann
from rl_iemocapEnv import IEMOCAPEnv

WINDOW_LENGTH = 1

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='iemocap-rl_v4-CustomPolicyBasedOnMaxBoltzmann')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    env = IEMOCAPEnv()
    nb_actions = env.action_space.n

    input_layer = Input(shape=(1, NUM_MFCC, NO_features))
    model = models.get_model_9_rl(input_layer, model_name_prefix='mfcc')

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
    #                               nb_steps=1000000)
    policy = CustomPolicyBasedOnMaxBoltzmann()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'rl-files/dqn_{}_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'rl-files/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'rl-files/dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        callbacks += [WandbLogger(project='iemocap-rl-v4')]
        dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)
    elif args.mode == 'test':
        weights_filename = 'rl-files/dqn_{}_weights.h5f'.format(args.env_name)
        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    run()
