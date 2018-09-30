import tensorflow as tf
import numpy as np
import gym, random

from collections import deque


class DQN(object):
    def __init__(self, **kwargs):
        self.N_EPISODE = 2000
        self.N_BATCH = 64
        self.LR = 0.01
        self.DISCOUNT = 0.99

        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n


    ## 네트워크 구조와 리턴으로 예측값
    def make_model(self, inputs, scope):
        with tf.variable_scope(scope):
            w1 = tf.get_variable('w1', dtype=tf.float32, shape=[self.state_size, 64])
            w2 = tf.get_variable('w2', dtype=tf.float32, shape=[64, self.action_size])

            layer1 = tf.nn.relu(tf.matmul(inputs, w1))
            pred = tf.nn.relu(tf.matmul(layer1, w2))

            return pred

    # 손실값 구하고 train 구조
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='aa')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='bb')

        self.predDQN = self.make_model(self.x, 'predDQN')
        self.targetDQN = self.make_model(self.x, 'targetDQN')
        
        self.loss = tf.losses.mean_squared_error(self.targetDQN, self.predDQN)
        self.train = tf.train.AdamOptimizer(self.LR).minimize(self.loss)


    def train_model(self):
        replay_memory = deque(maxlen=50000)
        op_holder = []
        
        with tf.Session() as sess:
            self.build_model()
        
            src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='predDQN')
            dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='targetDQN')

            sess.run(tf.global_variables_initializer())

            for src_var, dst_var in zip(src_vars, dst_vars):
                op_holder.append(dst_var.assign(src_var.value()))

            sess.run(op_holder)

            for ep in range(self.N_EPISODE):
                epsilon = 1. / ((ep / 10) + 1)
                done = False
                step_count = 0
                state = self.env.reset()
                state = np.reshape(state, [-1, self.state_size])

                while not done:
                    self.env.render()

                    if state.ndim == 1:
                        state = np.reshape(state, [-1, self.state_size])

                    if np.random.rand(1) < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(sess.run(self.predDQN, feed_dict={self.x:state}))

                    next_state, reward, done, _, = self.env.step(action)

                    if done:
                        reward = -1

                    replay_memory.append((state, action, reward, next_state, done))

                    if len(replay_memory) > self.N_BATCH:
                        minibatch = random.sample(replay_memory, self.N_BATCH)
                        state_batch = np.vstack([x[0] for x in minibatch])
                        action_batch = np.array([x[1] for x in minibatch])
                        # print(action_batch)
                        reward_batch = np.array([x[2] for x in minibatch])
                        next_state_batch = np.vstack([x[3] for x in minibatch])
                        done_array = np.array([x[4] for x in minibatch])

                        inputs = state_batch

                        Q_target = reward_batch + self.DISCOUNT*np.max(sess.run(self.targetDQN, feed_dict={self.x:next_state_batch}), axis=1) * ~done_array
                        label = sess.run(self.predDQN, feed_dict={self.x:state_batch})
                        #print(action_batch)
                        label[np.arange(len(inputs)), action_batch] = Q_target

                        loss, _ = sess.run([self.loss, self.train], feed_dict={self.x:inputs, self.y:label})

                    if step_count % 5 == 0:
                        sess.run(op_holder)

                    state = next_state
                    step_count += 1

                print("Episode: {}  steps: {}".format(ep, step_count))


myDQN = DQN()
myDQN.train_model()