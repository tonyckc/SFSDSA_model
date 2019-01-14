import tensorflow as tf


class Autoencoder(object):

    def __init__(self, n_layers, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), ae_para=[0, 0], is_first=True, regularzation_rate=0.1):
        # prepare
        self.n_layers = n_layers
        self.transfer = transfer_function
        self.in_keep_prob = 1 - ae_para[0]

        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.sparsity_level = 0.1  # np.repeat([0.05], self.n_hidden).astype(np.float32)
        self.sparse_reg = ae_para[1]
        self.epsilon = 1e-06

        # model

        self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
        self.theory = tf.placeholder(tf.float32, [None, self.n_layers[0]])


        #
        #self.keep_prob = tf.placeholder(tf.float32)
        #print(self.keep_prob)
        #
        self.hidden_encode = []
        #
        h = self.x
        # congression layer y = w*x + b
        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)
        #
        self.hidden_recon = []
        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                       self.weights['recon'][layer]['b']))
            self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]

        # cost
        if is_first:
            if self.sparse_reg == 0:

                    regularizer = tf.contrib.layers.l2_regularizer(regularzation_rate)
                    regularaztion = regularizer(self.weights['encode'][layer]['w'])+regularizer(self.weights['recon'][layer]['w'])
                    self.cost = tf.reduce_mean(tf.square(tf.subtract(self.reconstruction, self.theory))) + regularaztion
            else:

                   self.cost = tf.reduce_mean(tf.square(tf.subtract(self.reconstruction, self.theory)))
        else:
            if self.sparse_reg == 0:
                     regularizer = tf.contrib.layers.l2_regularizer(regularzation_rate)
                     regularaztion = regularizer(self.weights['encode'][layer]['w']) + regularizer( self.weights['recon'][layer]['w'])
                     self.cost = tf.reduce_mean(tf.square(tf.subtract(self.reconstruction, self.x))) + regularaztion

            else:

                     self.cost = tf.reduce_mean(tf.square(tf.subtract(self.reconstruction, self.x)))
        self.optimizer = optimizer.minimize(self.cost)

    def _initialize_weights(self):
        # the type of all_weights of dict, the first partial 'encode'=>list=>dict 、the second partial 'decode'
        all_weights = dict()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                tf.random_normal([self.n_layers[layer], self.n_layers[layer+1]],stddev=0.002, dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                tf.random_normal([self.n_layers[layer-1], self.n_layers[layer]], stddev=0.002, dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights


    def kl_divergence(self,  p, p_hat):
        return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                              - p * tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                              + (1 - p) * tf.log(tf.clip_by_value(1-p, 1e-8, tf.reduce_max(1-p)))
                              - (1 - p) * tf.log(tf.clip_by_value(1-p_hat, 1e-8, tf.reduce_max(1-p_hat))))

    def partial_fit(self):
        return (self.cost, self.optimizer)

    def calc_total_cost(self):
        return self.cost

    def transform(self):
        return self.hidden_encode[-1]

    def reconstruct(self):
        return self.reconstruction

    def setNewX(self,x):
        self.hidden_encode = []
        h = tf.nn.dropout(x, self.keep_prob)
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)



