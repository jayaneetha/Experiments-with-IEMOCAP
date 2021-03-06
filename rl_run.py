import argparse
import os

import tensorflow as tf
from keras import Input
from keras.optimizers import Adam

import models
from constants import NUM_MFCC, NO_features, SAVEE_NO_features, IMPROV_NO_features
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, WandbLogger
from rl.memory import SequentialMemory
from rl.policy import MaxBoltzmannQPolicy, Policy, LinearAnnealedPolicy, EpsGreedyQPolicy, SoftmaxPolicy, GreedyQPolicy, \
    BoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl_MSImprovEnv import ImprovEnv
from rl_custom_policy import CustomPolicy, CustomPolicyBasedOnMaxBoltzmann
from rl_iemocapEnv import IEMOCAPEnv, DataVersions
from rl_saveeEnv import SAVEEEnv

WINDOW_LENGTH = 1


def parse_args(args):
    dv: DataVersions = DataVersions.V4
    pol: Policy = EpsGreedyQPolicy()

    if args.data_version == 'v3':
        dv = DataVersions.V3
    if args.data_version == 'v4':
        dv = DataVersions.V4
    if args.data_version == 'savee':
        dv = DataVersions.Vsavee
    if args.data_version == 'improv':
        dv = DataVersions.Vimprov

    if args.policy == 'LinearAnnealedPolicy':
        pol = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05,
                                   nb_steps=args.zeta_nb_steps)
    if args.policy == 'SoftmaxPolicy':
        pol = SoftmaxPolicy()
    if args.policy == 'EpsGreedyQPolicy':
        pol = EpsGreedyQPolicy()
    if args.policy == 'GreedyQPolicy':
        pol = GreedyQPolicy()
    if args.policy == 'BoltzmannQPolicy':
        pol = BoltzmannQPolicy()
    if args.policy == 'MaxBoltzmannQPolicy':
        pol = MaxBoltzmannQPolicy()
    if args.policy == 'BoltzmannGumbelQPolicy':
        pol = BoltzmannGumbelQPolicy()
    if args.policy == 'CustomPolicy':
        pol = CustomPolicy()
    if args.policy == 'CustomPolicyBasedOnMaxBoltzmann' or args.policy == 'zetapolicy':
        pol = CustomPolicyBasedOnMaxBoltzmann(zeta_nb_steps=args.zeta_nb_steps, eps=args.eps)

    return dv, pol


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2dataset(v):
    ds = v.lower()
    if ds == 'v3':
        return DataVersions.V3
    if ds == 'v4':
        return DataVersions.V4
    if ds == 'savee':
        return DataVersions.Vsavee
    if ds == 'improv':
        return DataVersions.Vimprov


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='iemocap-rl-v3.1')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--policy', type=str, default='EpsGreedyQPolicy')
    parser.add_argument('--data-version', choices=['v4', 'v3', 'savee', 'improv'], type=str, default='v4')
    parser.add_argument('--disable-wandb', type=str2bool, default=False)
    parser.add_argument('--zeta-nb-steps', type=int, default=1000000)
    parser.add_argument('--nb-steps', type=int, default=500000)
    parser.add_argument('--max-train-steps', type=int, default=440000)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pre-train', type=str2bool, default=False)
    parser.add_argument('--pre-train-dataset',
                        choices=[DataVersions.V4, DataVersions.V3, DataVersions.Vsavee, DataVersions.Vimprov],
                        type=str2dataset, default=DataVersions.V4)
    parser.add_argument('--warmup-steps', type=int, default=50000)
    parser.add_argument('--pretrain-epochs', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    data_version, policy = parse_args(args)

    print("Starting ...\n\tPolicy: {}\n\tData Version: {}\n\tEnvironment: {}".format(args.policy, args.data_version,
                                                                                     args.env_name))

    env = None

    if data_version == DataVersions.V4 or data_version == DataVersions.V3:
        env = IEMOCAPEnv(data_version)

    if data_version == DataVersions.Vsavee:
        env = SAVEEEnv(data_version)

    if data_version == DataVersions.Vimprov:
        env = ImprovEnv(data_version)

    for k in args.__dict__.keys():
        print("\t{} :\t{}".format(k, args.__dict__[k]))
        env.__setattr__("_" + k, args.__dict__[k])

    exp_name = "P-{}-S-{}-e-{}-pt-{}".format(args.policy, args.zeta_nb_steps, args.eps, args.pre_train)
    if args.pre_train:
        exp_name = "P-{}-S-{}-e-{}-pt-{}-pt-w-{}".format(args.policy, args.zeta_nb_steps, args.eps, args.pre_train,
                                                         args.pre_train_dataset.name)
    env.__setattr__("_experiment", exp_name)

    nb_actions = env.action_space.n

    input_layer = Input(shape=(1, NUM_MFCC, NO_features))
    if data_version == DataVersions.Vsavee:
        input_layer = Input(shape=(1, NUM_MFCC, SAVEE_NO_features))
    if data_version == DataVersions.Vimprov:
        input_layer = Input(shape=(1, NUM_MFCC, IMPROV_NO_features))

    model = models.get_model_9_rl(input_layer, model_name_prefix='mfcc')

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05,
    #                               nb_steps=1000000)

    # policy = MaxBoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   nb_steps_warmup=args.warmup_steps, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1., train_max_steps=args.max_train_steps)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.pre_train:
        from data import FeatureType
        from Datastore import Datastore
        from IMPROVDataset import ImprovDataset

        datastore: Datastore = None
        no_features: int = 0

        if args.pre_train_dataset == DataVersions.V4:
            from V4Dataset import V4Datastore
            datastore = V4Datastore(FeatureType.MFCC)
            no_features = NO_features

        if args.pre_train_dataset == DataVersions.Vimprov:
            datastore = ImprovDataset(22)
            no_features = IMPROV_NO_features

        assert datastore is not None
        assert no_features != 0

        x_train, y_train, y_gen_train = datastore.get_pre_train_data()

        dqn.pre_train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, no_features)), y=y_train,
                      EPOCHS=args.pretrain_epochs, batch_size=128)

    if args.mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'rl-files/models/dqn_{}_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'rl-files/models/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'rl-files/logs/dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]

        if not args.disable_wandb:
            project_name = 'iemocap-rl-' + args.data_version

            if data_version == DataVersions.Vsavee:
                project_name = 'iemocap-rl-v4'

            callbacks += [WandbLogger(project=project_name, name=args.env_name)]

        dqn.fit(env, callbacks=callbacks, nb_steps=args.nb_steps, log_interval=10000)

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
