import numpy as np
import tensorflow as tf
import ipdb
import gym
import os

class DeepCartPole:

    def __init__(self):
        self.EPISODE = 1000
        self.TRANSITION = 3000
        self.restore = False
        self.sess = tf.Session()
        self.env = gym.make("CartPole-v0")
        if self.isRestore():
            #ipdb.set_trace()
            self.saver = tf.train.import_meta_graph('dcp_savefile/deep_cartpole_save.meta')
            self.saver.restore(self.sess, './dcp_savefile/deep_cartpole_save')
            self._load_network()
        else:
            self._build_network()
            self._gen_dataset()
            self._train()
        #self.test()

    def _gen_dataset(self):
        self.EPISODE = 1000
        self.TRANSITION = 3000

        state = self.env.reset()

        states = []
        newY = []
        for i in range(self.EPISODE):
            state = self.env.reset()
            for j in range(self.TRANSITION):
                pole_angle = state[2]
                pole_velocity = state[3]
                if pole_angle < 0 and pole_velocity < 0:     # bad
                    newY.append(1)                           # action : left
                elif pole_angle > 0 and pole_velocity > 0:   # bad
                    newY.append(2)                           # action : right
                else:                                        # safe
                    newY.append(0)
                states.append(state)
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break

        self.X = np.asarray(states)
        self.Y = np.asarray(newY)
        self.y_one_hot = tf.one_hot(self.Y, self.n_h2)
        self.Y_one_hot = self.sess.run(self.y_one_hot)

    def _build_network(self):
        n_input = 4
        n_h1_output = 2
        self.n_h2 = 3

        self.input = tf.placeholder(tf.float32, [None, n_input], name="input")
        self.label = tf.placeholder(tf.float32, [None, self.n_h2], name="label")  # one hot vector

        W1 = tf.Variable(tf.random_normal([n_input, n_h1_output]), name="W1")
        b1 = tf.Variable(tf.random_normal([n_h1_output]), name="b1")
        fc1 = tf.nn.relu(tf.add(tf.matmul(self.input, W1), b1), name="fc1")

        W2 = tf.Variable(tf.random_normal([n_h1_output, self.n_h2]), name="W2")
        b2 = tf.Variable(tf.random_normal([self.n_h2]), name="b2")
        self.fc2 = tf.nn.softmax(tf.add(tf.matmul(fc1, W2), b2), name="fc2")

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.fc2, name="loss")
        self.train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss, name="train")
        self.loss_print = tf.reduce_sum(self.loss, name="loss_print")

        self.predict_action = tf.argmax(self.fc2, axis=1, name="predict_action")
        predict_action_onehot = tf.one_hot(self.predict_action, self.n_h2, name="predict_action_onehot")
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_action_onehot, self.label), dtype=tf.float32),
                                    name="accuracy")
            
    def _load_network(self):
        print("Load complete!")
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.label = graph.get_tensor_by_name("label:0")
    
        self.fc2 = graph.get_tensor_by_name("fc2:0")
        self.loss = graph.get_tensor_by_name("loss:0")
        self.train = graph.get_operation_by_name("train")
        self.loss_print = graph.get_tensor_by_name("loss_print:0")

        self.predict_action = graph.get_tensor_by_name("predict_action:0")
        self.accuracy = graph.get_tensor_by_name("accuracy:0")

    def isRestore(self):
        if os.path.exists('./dcp_savefile/deep_cartpole_save.meta'):
            return True
        else:
            return False

    
    def _train(self):

        if os.path.exists('./dcp_savefile/deep_cartpole_save.meta'):
            self.saver = tf.train.import_meta_graph('dcp_savefile/deep_cartpole_save.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
            self._load_network()

        else:
            self.sess.run(tf.global_variables_initializer())
            batch_size = 100
            batch_end = self.X.shape[0]
            batch_iter = batch_end  # for first shuffle
            for step in range(30000):

                if batch_iter + batch_size > batch_end:
                    index_list = list(range(self.X.shape[0]))
                    np.random.shuffle(index_list)
                    self.X = self.X[index_list]
                    self.Y = self.Y[index_list]
                    self.y_one_hot = tf.one_hot(self.Y, self.n_h2)
                    self.Y_one_hot = self.sess.run(self.y_one_hot)
                    batch_iter = 0

                batchX = self.X[batch_iter:batch_iter+batch_size]
                batchY_onehot = self.Y_one_hot[batch_iter:batch_iter+batch_size]
                batch_iter += batch_size

                self.sess.run(self.train, feed_dict={self.input: batchX, self.label: batchY_onehot})

                if step % 1000 == 0:

                    #print(step, ", loss :", self.sess.run(self.loss_print, feed_dict={self.input: self.X, self.label: self.Y_one_hot})/batch_size)
                    #pred_prob = self.sess.run(self.fc2, feed_dict={self.input: self.X})
                    pred_action = self.sess.run(self.predict_action, feed_dict={self.input: self.X})

                    true_num = 0
                    false_num = 0
                    for iter, y in enumerate(self.Y):
                        if y != 0:
                            if y == pred_action[iter]:
                                true_num += 1
                            else:
                                false_num += 1

                    onetwo_accu = true_num / (false_num + true_num)

                    #print("onetwo_accu", onetwo_accu)
                    print("deep cartpole accuracy ({}):".format(step), self.sess.run(self.accuracy,
                                            feed_dict={self.input: self.X, self.label: self.Y_one_hot}))

            self.saver = tf.train.Saver()
            self.saver.save(self.sess, "./dcp_savefile/deep_cartpole_save")
            self.restore = True

    def test(self):
        for i in range(self.EPISODE):

            ipdb.set_trace()
            state = self.env.reset()
            rewards = 0
            for j in range(self.TRANSITION):
                self.env.render()
                pred_action = self.sess.run(self.predict_action, feed_dict={self.input: [state]})
                if pred_action[0] == 0:
                    action = self.env.action_space.sample()
                elif pred_action[0] == 1:
                    action = 0
                elif pred_action[0] == 2:
                    action = 1
                next_state, reward, done, info = self.env.step(action)
                rewards += reward
                state = next_state
                if done:
                    print("total rewards : {}".format(rewards))
                    break

    def get_features(self, state):
        return self.sess.run(self.fc2, feed_dict={self.input: [state]})[0]

if __name__ == "__main__":
    dcp = DeepCartPole()
    dcp.test()
