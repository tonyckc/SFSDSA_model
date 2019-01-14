# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:02:36 2018

@author: ckc
"""
import tensorflow as tf
from tensorflow import logging
from function_file.Autoencoder import Autoencoder
import scipy.io as sci
import numpy  as np
import random
import os

"""Read data from  *.mat including two kinds"""
data_tarin = sci.loadmat('train_data_new.mat')
data_valid = sci.loadmat('valid_data.mat')
data_test = sci.loadmat('test_data.mat')
#
data_train_noisy = data_tarin['noiseSignal']
data_train_theory = data_tarin['theorySignal']
#
data_valid_theory = data_valid['theorySignal']
data_valid_noisy = data_valid['noiseSignal']
#
data_test_theory = data_test['theorySignal']
data_test_noisy = data_test['noiseSignal']

logging.info("--------------------------------Data Reading Done!--------------------------------")

""" Set hyper-parameters """
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001]
LEARNING_RATE_DECAY = 0.99
REGULARZATION_RATE = [0.05, 0.1, 0.15, 0.20]
DISPLAY_STEP = 1
CORRUPTION_LEVEL = 0.3
SPARSE_REG = 0
MODEL_SIZE = [434, 217, 108, 434]
# MODEL_SIZE = [434, 217, 108, 54, 434]
# MODEL_SIZE = [434, 217, 108, 54, 27,434]
# MODEL_SIZE = [434, 217, 108,54,27,13 434]
NUM_BATCH = len(data_train_noisy) // BATCH_SIZE
logging.info("--------------------------------Hyper-parameters Setting Done!--------------------------------")
"""Start grid search """
for re_num in REGULARZATION_RATE:
    for lr_num in LEARNING_RATE:
        """Data Saving Dir"""
        TEST_NAME = "MSE_test_re=" + str(re_num) + '_lr=' + str(lr_num)
        curdir = os.getcwd()
        save_dir = curdir + '/result/' + TEST_NAME
        if os.path.exists(save_dir):
            print(save_dir + 'is exists')
        else:
            os.makedirs(save_dir)

        """Definite Autoencoder"""
        AE = Autoencoder(n_layers=[MODEL_SIZE[0], MODEL_SIZE[1]],
                         transfer_function=tf.nn.selu,
                         optimizer=tf.train.AdamOptimizer(learning_rate=lr_num),
                         ae_para=[CORRUPTION_LEVEL, SPARSE_REG], is_first=True, regularzation_rate=re_num)
        AE_2 = Autoencoder(n_layers=[MODEL_SIZE[1], MODEL_SIZE[2]],
                           transfer_function=tf.nn.selu,
                           optimizer=tf.train.AdamOptimizer(learning_rate=lr_num),
                           ae_para=[CORRUPTION_LEVEL, SPARSE_REG], is_first=False, regularzation_rate=re_num)
        '''
             AE_3 = Autoencoder(n_layers=[MODEL_SIZE[1], MODEL_SIZE[2]],
                                transfer_function=tf.nn.selu,
                                optimizer=tf.train.AdamOptimizer(learning_rate=lr_num),
                                ae_para=[CORRUPTION_LEVEL, SPARSE_REG], is_first=False, regularzation_rate=re_num)
             AE_4 = Autoencoder(n_layers=[MODEL_SIZE[1], MODEL_SIZE[2]],
                                transfer_function=tf.nn.selu,
                                optimizer=tf.train.AdamOptimizer(learning_rate=lr_num),
                                ae_para=[CORRUPTION_LEVEL, SPARSE_REG], is_first=False, regularzation_rate=re_num)
             AE_5 = Autoencoder(n_layers=[MODEL_SIZE[1], MODEL_SIZE[2]],
                                transfer_function=tf.nn.selu,
                                optimizer=tf.train.AdamOptimizer(learning_rate=lr_num),
                                ae_para=[CORRUPTION_LEVEL, SPARSE_REG], is_first=False, regularzation_rate=re_num)
        '''
        logging.info("--------------------------------Initialize Auto-encoders Done!--------------------------------")

        """Create the placeholder of input and compare data  """
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, MODEL_SIZE[0]])
        compare_data = tf.placeholder(dtype=tf.float32, shape=[None, MODEL_SIZE[0]])
        logging.info("Create the placeholder of input and compare data Done")

        """Create the reconstruction layer"""
        x = tf.placeholder(dtype=tf.float32, shape={None, MODEL_SIZE[2]})
        W = tf.Variable(tf.random_normal(shape=[MODEL_SIZE[2], MODEL_SIZE[3]], stddev=0.002, dtype=tf.float32))
        b = tf.Variable(tf.zeros(shape=[MODEL_SIZE[3]], dtype=tf.float32))
        y = tf.add(tf.matmul(x, W), b)
        logging.info("--------------------------------Create the reconstruction layer Done!---------------------------"
                     "-----")

        """Create the optimizer"""

        regularizer = tf.contrib.layers.l2_regularizer(re_num)
        regularaztion = regularizer(W)
        cost = tf.reduce_mean(tf.square(tf.subtract(y, compare_data))) + regularaztion
        cost_ = tf.reduce_mean(tf.abs(tf.subtract(y, compare_data)))
        train = tf.train.AdamOptimizer(learning_rate=lr_num).minimize(cost)
        logging.info("Create the optimizer Done")
        """Create Fine Tuning partial"""
        data_ft = input_data

        for layer in range(len(AE.n_layers) - 1):
            data_ft = AE.transfer(tf.add(
                tf.matmul(data_ft, AE.weights['encode'][layer]['w']), AE.weights['encode'][layer]['b']
            ))
        for layer in range(len(AE_2.n_layers) - 1):
            data_ft = AE_2.transfer(tf.add(
                tf.matmul(data_ft, AE_2.weights['encode'][layer]['w']), AE_2.weights['encode'][layer]['b'])
            )

        ft_result = tf.add(tf.matmul(data_ft, W), b)
        regularizer_ft = tf.contrib.layers.l2_regularizer(re_num)
        regularaztion_ft = regularizer_ft(W) + regularizer_ft(AE.weights['encode'][layer]['w']) + regularizer_ft(
            AE_2.weights['encode'][layer]['w'])
        cost_ft = tf.reduce_mean(tf.square(tf.subtract(ft_result, compare_data))) + regularaztion_ft
        cost_ft_ = tf.reduce_mean(tf.abs(tf.subtract(ft_result, compare_data)))
        train_ft = tf.train.AdamOptimizer().minimize(cost_ft)
        logging.info("Fine Tuning partialDone!")

        with tf.Session() as sess:
            "Global Variables initialization "
            init_variable = tf.global_variables_initializer()
            sess.run(init_variable)
            logging.info(
                "Global Variables initialization Done!"
            )
            """Training the first Auto-encoder"""
            cost_log = []
            for epoch in range(EPOCHS):
                batch_point = 0
                rand_batch_point = random.randint(0, (NUM_BATCH - 1))
                for num in range(NUM_BATCH):
                    batch_point = (rand_batch_point * BATCH_SIZE)
                    batch_data_noisy = np.array(data_train_noisy[batch_point:(batch_point + BATCH_SIZE)]).astype(
                        'float32')
                    batch_data_theory = np.array(data_train_theory[batch_point:(batch_point + BATCH_SIZE)]).astype(
                        'float32')
                    temp = AE.partial_fit()
                    cost_AE, opt = sess.run(temp, feed_dict={AE.x: batch_data_noisy,
                                                             AE.theory: batch_data_theory})  # ,AE.keep_prob: AE.in_keep_prob

                    avg_cost = cost_AE
                    # print("{}".format(avg_cost))

                    if num % 10 == 0:
                        print(
                            "Epoch: %d Iteration: %d" % (epoch + 1, num),
                            "Cost: {}".format(avg_cost)
                        )
                    if num % 1 == 0:
                        cost_log.append(avg_cost)

            sci.savemat(save_dir + os.path.sep + 'AE1_Cost' + TEST_NAME + '.mat', {'cost_log': cost_log})  # 写入mat文件
            # print(cost_log)

            print("************************First AE training finished******************************")

            """Training the Second Auto-encoder"""
            cost_log = []
            for epoch in range(EPOCHS):
                batch_point = 0
                rand_batch_point = random.randint(0, (NUM_BATCH - 1))
                for num in range(NUM_BATCH):
                    batch_point = rand_batch_point * BATCH_SIZE
                    batch_data_noisy = data_train_noisy[batch_point:(batch_point + BATCH_SIZE)]

                    input_data_AE_2 = sess.run(AE.transform(),
                                               feed_dict={AE.x: batch_data_noisy})
                    # , AE.keep_prob: AE.in_keep_prob
                    temp = AE_2.partial_fit()
                    cost_AE2, opt = sess.run(temp, feed_dict={AE_2.x: input_data_AE_2})
                    # , AE_2.keep_prob: AE_2.in_keep_prob
                    avg_cost = cost_AE2

                    if num % 10 == 0:
                        print(
                            "Epoch: %d Iteration: %d" % (epoch + 1, num),
                            "Cost: {}".format(avg_cost)
                        )
                    if num % 1 == 0:
                        cost_log.append(avg_cost)
            sci.savemat(save_dir + os.path.sep + 'AE2_Cost' + TEST_NAME + '.mat', {'cost_log': cost_log})  #
            print("************************Second AE training finished******************************")

            """ Training the reconstruction layer  """
            cost_log = []
            cost_log_valid = []
            for epoch in range(EPOCHS):
                batch_point = 0
                rand_batch_point = random.randint(0, (NUM_BATCH - 1))
                for num in range(NUM_BATCH):
                    batch_point = (rand_batch_point * BATCH_SIZE)
                    batch_data_noisy = data_train_noisy[batch_point:(batch_point + BATCH_SIZE)]
                    batch_data_theory = data_train_theory[batch_point:(batch_point + BATCH_SIZE)]
                    # train
                    AE_out = sess.run(AE.transform(), feed_dict={AE.x: batch_data_noisy})
                    AE2_out = sess.run(AE_2.transform(), feed_dict={AE_2.x: AE_out})
                    train_cost, train_step = sess.run([cost, train],
                                                      feed_dict={x: AE2_out, compare_data: batch_data_theory})
                    avg_cost = train_cost

                    if num % 10 == 0:
                        print(
                            "Epoch: %d Iteration: %d" % (epoch + 1, num),
                            "Cost: {}".format(avg_cost)
                        )
                        # valid
                        AE_out = sess.run(AE.transform(), feed_dict={AE.x: data_valid_noisy})
                        AE2_out = sess.run(AE_2.transform(), feed_dict={AE_2.x: AE_out})
                        valid_cost = sess.run(cost_, feed_dict={x: AE2_out, compare_data: data_valid_theory})
                        avg_cost_valid = valid_cost
                        cost_log_valid.append(avg_cost_valid)
                    if num % 1 == 0:
                        cost_log.append(avg_cost)

            sci.savemat(save_dir + os.path.sep + 'Recon_Cost_' + TEST_NAME + '.mat', {'cost_log': cost_log})  #
            sci.savemat(save_dir + os.path.sep + 'Recon_Cost_Valid_' + TEST_NAME + '.mat',
                        {'cost_log_valid': cost_log_valid})  #
            print("************************reconstruction layer training finished******************************")

            """Training of fine tune"""
            cost_log = []
            test_log = np.zeros([1000, 434])  # (NUM_BATCH // 50 +1) * EPOCHS * 3
            test_cost_log = []
            batch_data_test = np.zeros([3, 434])
            batch_data_test_theory = np.zeros([3, 434])
            count = [0, 801, 1601]
            "prepare test data-set"
            for i in range(3):
                batch_data_test[i] = data_train_noisy[count[i]]
            for i in range(3):
                batch_data_test_theory[i] = data_train_theory[count[i]]
            j = 0
            for epoch in range(EPOCHS):
                batch_point = 0
                rand_batch_point = random.randint(0, (NUM_BATCH - 1))
                for num in range(NUM_BATCH):
                    batch_point = (rand_batch_point * BATCH_SIZE)
                    batch_data_noisy = data_train_noisy[batch_point:(batch_point + BATCH_SIZE)]
                    batch_data_theory = data_train_theory[batch_point:(batch_point + BATCH_SIZE)]
                    # train
                    cost, opt = sess.run([cost_ft, train_ft],
                                         feed_dict={input_data: batch_data_noisy, compare_data: batch_data_theory})
                    avg_cost = cost

                    if num % 10 == 0:
                        print(
                            "Epoch: %d Iteration: %d" % (epoch + 1, num),
                            "Cost: {}".format(avg_cost)
                        )
                        out_result, avg_out_result_cost = sess.run([ft_result, cost_ft_], feed_dict=
                        {input_data: data_valid_noisy, compare_data: data_valid_theory})

                        for i in range(3):
                            j += 1
                            test_log[(NUM_BATCH // 50) * epoch + (num // 50) + j] = out_result[i]

                        test_cost_log.append(avg_out_result_cost)

                    if num % 1 == 0:
                        cost_log.append(avg_cost)

            sci.savemat(save_dir + os.path.sep + 'FineTune_Cost_' + TEST_NAME + '.mat',
                        {'cost_log': cost_log})  #
            sci.savemat(save_dir + os.path.sep + 'FineTune_valid_' + TEST_NAME + '.mat',
                        {'test_log': test_log})  #
            sci.savemat(save_dir + os.path.sep + 'FineTune_valid_cost_' + TEST_NAME + '.mat',
                        {'test_cost_log': test_cost_log})  #

            print("************************fine tune layer training finished******************************")
            """Test"""
            '''
            test_data = np.array(data_train_noisy[0]).reshape([1, 434])
            test_data_theory = np.array(data_train_theory[0]).reshape([1, 434])
            out_result = sess.run(ft_result, feed_dict={input_data: test_data})
            test_file = open('test_file.txt', 'w')
            error_out = 0
            error_input = 0
            for i in range(len(out_result[0])):
                test_file.write(str(out_result[0][i]))
                test_file.write('\n')
                error_out += (abs(out_result[0][i] - test_data_theory[0][i]))
                error_input += (abs(test_data[0][i] - test_data_theory[0][i]))
            print("out-theory{}\n".format(error_out / 434))
            print("input-theory{}\n".format(error_input / 434))

            test_file.close()
            '''
            out_result = sess.run(ft_result, feed_dict=
            {input_data: data_test_noisy, compare_data: data_test_theory})
            MAE_50 = []
            f = open(save_dir + os.path.sep + 'MAE50.txt', 'w')

            for i in range(len(data_test_noisy)):
                temp = 0
                for j in range(50):
                    temp += abs(out_result[i][j] - data_test_theory[i][j])
                    if j == 49:
                        MAE_50.append(temp / 50)
            MAE50 = sum(MAE_50) / len(data_test_noisy)
            print(MAE50)
            f.write(str(MAE50))
            f.close()
            print("************************Test Finished******************************")
