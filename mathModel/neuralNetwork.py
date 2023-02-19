# coding=gbk

import numpy as np
import pickle
import scipy.special
from key_management import Keyword_and_Parameter as kp
from document import Document_process as dop

'''
    功能内容：多隐层结构的神经网络模型算法，包括网络参数初始化、网络结果运算、网络误差反向传递、网络训练过程和网络训练准确率计算
    创建时间：2022/3/22
    完成时间：2022/3/25
'''


class neuralNetwork:
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置神经网络的3种层，每层的神经元的数量
        self.inodes = inputnodes  # 输入层神经元数量
        self.hnodes = hiddennodes  # 隐层神经元数量
        self.onodes = outputnodes  # 输出层神经元数量
        # 输入层与隐藏层之间的权重矩阵，首次进行随机生成，范围为 到下一层连接数的开根号的倒数的正负数
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 隐藏层与输出层之间的权重矩阵，首次进行随机生成，范围为 到下一层连接数的开根号的倒数的正负数
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)  # 初始化激活函数
        self.inverse_activation_function = lambda x: scipy.special.logit(x)  # 初始化逆向激活函数
        # 学习因子
        self.lr = learningrate
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 1.针对给定的训练样本进行输出
        # 将输入参数和目标参数分别转换成2维竖排形式
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 得到隐藏层的输入数据
        hidden_inputs = np.dot(self.wih, inputs)
        # 得到隐藏层的输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 得到输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 得到输出层的输出
        final_outputs = self.activation_function(final_inputs)

        # 2.计算误差，并向上优化权重
        # 计算结果误差
        output_errors = targets - final_outputs
        # 计算隐藏层各权重的误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 得到各隐藏层到输出层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        # 得到各输入层到隐藏层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        pass

    # 查询神经网络的输出
    def query(self, inputs_list):
        # 将输入的数据数组，转换成为2维形式
        inputs = np.array(inputs_list, ndmin=2).T
        # 隐藏层的输入数据，为权重矩阵点乘输入层数据
        hidden_inputs = np.dot(self.wih, inputs)
        # 隐藏层的输出数据，为隐藏层的输入数据进行sigmoid计算
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层的输入数据，为权重矩阵点乘隐藏层的输出数据
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层的输出数据，为输出层的输入数据进行sigmoid计算
        final_outputs = self.activation_function(final_inputs)

        return final_inputs

    # 根据输入测试集和目标集，输出准确率
    def score(self, inputs_list, targets_list):
        scorecard = []
        for i in range(len(inputs_list)):
            output_values = neuralNetwork.query(inputs_list[i])
            output_value = np.argmax(output_values)
            if targets_list[i] == output_value:
                scorecard.append(1)
            else:
                scorecard.append(0)
        scorecard = np.asarray(scorecard)
        return scorecard.sum() / scorecard.size

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


class neuralNetwork_2:

    # 初始化神经网络
    def __init__(self, inputnodes, outputnodes, hnodes_num, hnodes_num_list, learningrate, activation_type):
        # 设置神经网络的3种层，每层的神经元的数量
        self.inodes = inputnodes  # 输入层神经元数量
        self.onodes = outputnodes  # 输出层神经元数量
        self.hnodes_num = hnodes_num

        # 判断进行几层隐层生成
        self.hnodes = hnodes_num_list  # 隐层神经元数量
        # 输入层与隐藏层之间的权重矩阵，首次进行随机生成，范围为 到下一层连接数的开根号的倒数的正负数
        self.wih = np.random.normal(0.0, pow(self.hnodes[0], -0.5), (self.hnodes[0], self.inodes))
        # 隐藏层与输出层之间的权重矩阵，首次进行随机生成，范围为 到下一层连接数的开根号的倒数的正负数
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes[-1]))
        # 输入层与隐藏层之间的偏置矩阵，首次进行随机生成，范围为 -1到+1之间
        self.bih = np.random.normal(0.0, 0.5, (self.hnodes[0], 1))
        # 隐藏层与输出层之间的偏置矩阵，首次进行随机生成，范围为 -1到+1之间
        self.bho = np.random.normal(0.0, 0.5, (self.onodes, 1))

        # 隐藏层之间的权重矩阵
        self.whh = []
        self.bhh = []
        for i in range(hnodes_num - 1):
            # 隐藏层与隐藏层之间的偏置矩阵，首次进行随机生成，范围为 -1到+1之间
            self.whh.append(np.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes[i + 1], self.hnodes[i])))
            # 隐藏层与隐藏层之间的偏置矩阵，首次进行随机生成，范围为 -1到+1之间
            self.bhh.append(np.random.normal(0.0, 0.5, (self.hnodes[i + 1], 1)))

        # 设定激活函数的类型
        self.activation_type = activation_type

        if activation_type == 'sigmoid':
            self.activation_function = lambda x: scipy.special.expit(x)  # 初始化激活函数,这里采用的是sigmoid函数
        elif activation_type == 'relu':
            self.activation_function = lambda x: np.maximum(x, 0)  # 初始化激活函数,这里采用的是Relu函数

        self.inverse_activation_function = lambda x: scipy.special.logit(x)  # 初始化逆向激活函数

        # 学习因子
        self.lr = learningrate
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 1.针对给定的训练样本进行输出
        # 将输入参数和目标参数分别转换成2维竖排形式
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 中间隐层之间的输出与输入
        hiddens_inputs = []
        hiddens_outputs = []
        for i in range(len(self.whh) + 1):
            if i == 0:
                # 得到隐藏层的输入数据
                hidden_inputs = np.dot(self.wih, inputs) + self.bih
            else:
                # 进行后一层隐层的输入数据计算
                hidden_inputs = np.dot(self.whh[i - 1], hidden_outputs) + self.bhh[i - 1]
            # 进行后一层隐层的输出数据计算，后期可以在这里控制每一个隐层的激活函数
            hidden_outputs = self.activation_function(hidden_inputs)
            # 将每次的隐层输入输出数据进行分别保存
            hiddens_inputs.append(hidden_inputs)
            hiddens_outputs.append(hidden_outputs)

        # 得到输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho
        # 得到输出层的输出
        final_outputs = self.activation_function(final_inputs)

        # 2.计算误差，并向上优化权重
        # 计算结果误差
        output_errors = targets - final_outputs

        # 中间隐层之间的输出与输入
        hiddens_errors = []
        for i in range(len(self.whh) + 1):
            if i == 0:
                hidden_errors = np.dot(self.who.T, output_errors)
            else:
                hidden_errors = np.dot(self.whh[len(self.whh) - i].T, hidden_errors)
            hiddens_errors.append(hidden_errors)

        # 误差反向传导函数
        self.back_propagation(output_errors, hiddens_errors, final_outputs, hiddens_outputs, inputs, final_inputs,
                              hiddens_inputs)

        pass

    # 查询神经网络的输出
    def query(self, inputs_list):
        # 将输入的数据数组，转换成为2维形式
        inputs = np.array(inputs_list, ndmin=2).T

        for i in range(len(self.whh) + 1):
            if i == 0:
                # 得到隐藏层的输入数据
                hidden_inputs = np.dot(self.wih, inputs) + self.bih
            else:
                # 进行后一层隐层的输入数据计算
                hidden_inputs = np.dot(self.whh[i - 1], hidden_outputs) + self.bhh[i - 1]
            # 进行后一层隐层的输出数据计算
            hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层的输入数据，为权重矩阵点乘隐藏层的输出数据
        final_inputs = np.dot(self.who, hidden_outputs) + self.bho
        # 输出层的输出数据，为输出层的输入数据进行sigmoid计算
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # 根据输入测试集和目标集，输出准确率
    def score(self, inputs_list, targets_list, result_type):
        scorecard = []
        recall = []
        for i in range(len(inputs_list)):
            if result_type == 'list':
                output_values = neuralNetwork.query(inputs_list[i])
                output_value = np.argmax(output_values)
            else:
                if output_values[0][0] >= 0.5:
                    output_value = 1
                else:
                    output_value = 0
            if targets_list[i] == output_value:
                scorecard.append(1)
            else:
                scorecard.append(0)
            if targets_list[i] == 1 and output_value == 1:
                recall.append(1)
            elif targets_list[i] == 1 and output_value != 1:
                recall.append(0)
        scorecard = np.asarray(scorecard)
        recall = np.asarray(recall)
        score = (2 * (scorecard.sum() / scorecard.size) * (recall.sum() / recall.size)) / ((scorecard.sum() / scorecard.size) + (recall.sum() / recall.size))
        return score

    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        for i in range(len(self.whh) + 1):
            if i == 0:
                # 得到隐藏层的输入数据
                hidden_outputs = np.dot(self.who.T, final_inputs - self.bho)
                hidden_outputs -= np.min(hidden_outputs)
                hidden_outputs /= np.max(hidden_outputs)
                hidden_outputs *= 0.98
                hidden_outputs += 0.01
            else:
                # 进行后一层隐层的输入数据计算
                hidden_outputs = np.dot(self.whh[len(self.whh) - i].T, hidden_inputs - self.bhh[len(self.whh) - i])
                hidden_outputs -= np.min(hidden_outputs)
                hidden_outputs /= np.max(hidden_outputs)
                hidden_outputs *= 0.98
                hidden_outputs += 0.01
            # 进行后一层隐层的输出数据计算，后期可以在这里控制每一个隐层的激活函数
            hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    # Relu函数导数
    def ReluPrime(self, x):
        x[x > 0] = 1
        x[x <= 0] = self.lr
        return x

    # change by 2022/6/5
    def save_result(self, network_save_path=''):
        para_save = {'inputnodes': self.inodes, 'outputnodes': self.onodes, 'hnodes_num': self.hnodes_num,
                     'hnodes_num_list': self.hnodes, 'learningrate': self.lr, 'activation_type': self.activation_type}
        if network_save_path == '':
            network_save_path = kp.get_parameter_with_keyword('netword_save_path')
        save_name = ['initial', 'who', 'whh', 'wih', 'bho', 'bhh', 'bih']
        save_data = [para_save, self.who, self.whh, self.wih, self.bho, self.bhh, self.bih]
        for i in range(len(save_name)):
            output_hal = open(network_save_path + save_name[i] + '.pkl', 'wb')
            str = pickle.dumps(save_data[i])
            output_hal.write(str)
            output_hal.close()

    # change by 2022/6/5
    def load_paremeter(self, path=''):
        if path == '':
            network_save_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('network_save_path'))
        else:
            network_save_path = dop.path_of_holder_document(path)
        for i in range(len(network_save_path)):
            if network_save_path[i][-3:] == 'csv':
                continue
            with open(network_save_path[i], 'rb') as file:
                data = pickle.loads(file.read())
            # if network_save_path[i].rsplit('/', 1)[1][:-4] == 'initial':
            #     self.inodes = data['inputnodes']  # 输入层神经元数量
            #     self.onodes = data['outputnodes']  # 输出层神经元数量
            #     self.hnodes_num = data['hnodes_num']
            #     # 判断进行几层隐层生成
            #     self.hnodes = data['hnodes_num_list']
            #     self.activation_type = data['activation_type']
            #     self.lr = data['learningrate']

            if network_save_path[i].rsplit('/', 1)[1][:-4] == 'who':
                self.who = data

            elif network_save_path[i].rsplit('/', 1)[1][:-4] == 'whh':
                self.whh = data

            elif network_save_path[i].rsplit('/', 1)[1][:-4] == 'wih':
                self.wih = data

            elif network_save_path[i].rsplit('/', 1)[1][:-4] == 'bho':
                self.bho = data

            elif network_save_path[i].rsplit('/', 1)[1][:-4] == 'bhh':
                self.bhh = data

            elif network_save_path[i].rsplit('/', 1)[1][:-4] == 'bih':
                self.bih = data

    # 误差反向传导函数
    def back_propagation(self, output_errors, hiddens_errors, final_outputs, hiddens_outputs, inputs, final_inputs,
                         hiddens_inputs):
        if self.activation_type == 'sigmoid':
            for i in range(len(hiddens_errors)):
                if i == 0:
                    # 得到各隐藏层到输出层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
                    self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs),
                                                 np.transpose(hiddens_outputs[len(hiddens_errors) - i - 1]))

                    # 得到各输入层到隐藏层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
                    self.wih += self.lr * np.dot(hiddens_errors[-1] * hiddens_outputs[i] * (1 - hiddens_outputs[i]),
                                                 np.transpose(inputs))

                    # 得到各隐藏层到输出层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
                    self.bho += self.lr * output_errors * final_outputs * (1 - final_outputs)

                    # 得到各输入层到隐藏层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
                    self.bih += self.lr * hiddens_errors[-1] * hiddens_outputs[i] * (1 - hiddens_outputs[i])
                else:
                    self.whh[len(hiddens_errors) - i - 1] += self.lr * np.dot(
                        hiddens_errors[i - 1] * hiddens_outputs[len(hiddens_errors) - i] * (1 - hiddens_outputs[len(hiddens_errors) - i]),
                        np.transpose(hiddens_outputs[len(hiddens_errors) - i - 1]))

                    self.bhh[len(hiddens_errors) - i - 1] += self.lr * hiddens_errors[i - 1] * hiddens_outputs[
                        len(hiddens_errors) - i] * (1 - hiddens_outputs[len(hiddens_errors) - i])

        elif self.activation_type == 'relu':
            # 得到各隐藏层到输出层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
            # self.who += self.lr * np.dot(output_errors * self.ReluPrime(final_inputs), np.transpose(hidden_outputs))
            #
            # # 得到各输入层到隐藏层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
            # self.wih += self.lr * np.dot(hidden_errors * self.ReluPrime(hidden_inputs), np.transpose(inputs))
            #
            # # 得到各隐藏层到输出层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
            self.bho += self.lr * output_errors * self.ReluPrime(final_inputs)
            #
            # # 得到各输入层到隐藏层之间各权重的变化值，并修正上一次的权重，其中transpose是将2维数组转换成1维
            # self.bih += self.lr * hidden_errors * self.ReluPrime(hidden_inputs)
