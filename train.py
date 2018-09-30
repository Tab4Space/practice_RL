import argparse
import tensorflow as tf
from dqn_cartpole import DQN


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, choices=['DQN'])
    parser.add_argument('--episode', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)

    # parser.add_argument('--game', type=str, default='CartPole-v1')

    return parser.parse_args()


def main():
    args = read_args()
    models = [DQN]

    sess = tf.Session()

    for model in models:
        if args.model == model.NAME:
            m = model()

    # m.build_model()
    m.train_model()


    # for model in models:
    #     if args.model == net

if __name__ == '__main__':
    main()