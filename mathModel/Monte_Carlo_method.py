# coding=gbk
"""
蒙特卡罗算法的底层实现函数
文档创建时间：2022/9/23
文档修改时间：
"""

# 导入算法包
import csv
import datetime
from data import Data_Basic_Function as dbf
from key_management import Keyword_and_Parameter as kp

'''
    创建时间: 2022/9/23
    完成时间: 2022/9/26
    功能: 首次生成样本库函数
    关键词：首次、样本库
    修改时间：
'''


def create_sample_first(key_word, sample_num, sample_features, sample_rate=None):
    """
    首次生成样本库函数
    :param list key_word: 设计生成的各类型样本的关键特征
    :param list sample_num: 设定各关键特征样本要生成的个数
    :param list sample_features: 生成特征的参数值值域
    :param list sample_rate: 选择值域内各参数的概率，当为空时，表示概率为平均分
    :return:
    """
    # 循环生成所有的样本
    sample_data = []  # 用于保存样本数据
    for k, key in enumerate(key_word):
        if sample_num[k] != 0:
            for i in range(int(sample_num[k])):
                # 判断是否各参数的选中概率为同样的
                if len(sample_rate) == 0:
                    rate = [1.0 / len(sample_features) for j in range(len(sample_features))]
                    sample_data.append([key, dbf.get_feature_random(sample_features, rate)])
                else:
                    # 如果概率不相同
                    sample_data.append([key, dbf.get_feature_random(sample_features, sample_rate)])
        else:
            pass
    return sample_data


'''
    创建时间: 2022/9/26
    完成时间: 2022/9/26
    功能: 根据特征，生成样本库新特征
    关键词：样本库，新特征
    修改时间：No.1 2022/9/27,增加样本输出特征为数字的可能性
'''


def create_sample_feature(sample_data, sample_features, sample_rate=None, random_type='', feature_type='str'):
    """
    根据特征，生成样本库新特征
    :param str random_type: 随机数生成类型
    :param str feature_type: 判断输出的结果的类型，str为字符串，int为整型，float为浮点型
    :param list sample_data: 已经生成的样本集数据
    :param list sample_features: 生成特征的参数值值域，当feature_type不是str时，为字典类型，存放各关键字类型对应的随机参数
    :param list sample_rate: 选择值域内各参数的概率，当为空时，表示概率为平均分
    :return:
    """
    # 遍历所有的已有样本数据，并添加上新的特征值
    for i in range(len(sample_data)):
        if feature_type == 'str':
            if len(sample_rate) == 0:
                rate = [1.0 / len(sample_features) for i in range(len(sample_features))]
                sample_data[i].append(dbf.get_feature_random(sample_features, rate, random_type, feature_type))
            else:
                # 如果概率不相同
                sample_data[i].append(dbf.get_feature_random(sample_features, sample_rate, random_type, feature_type))
        else:
            # 根据该条记录下关键字内容，输入对应的随机参数
            sample_data[i].append(
                dbf.get_feature_random(sample_features[int(sample_data[i][0])], sample_rate, random_type, feature_type))
    return sample_data


'''
    创建时间: 2022/9/26
    完成时间: 2022/9/27
    功能: 对模拟生成的所有特征进行计算，输出目标值
    关键词：特征计算，目标值
    修改时间：
'''


def compute_sample_feature(sample_data, index, filter_num, treat_type='add'):
    """
    对模拟生成的所有特征进行计算，输出目标值
    :param str treat_type: 对各特征的计算方式
    :param list sample_data: 对象数据
    :param int index: 数据过滤的字段
    :param filter_num: 数据过滤判断数值
    :return:
    """
    # 遍历所有的数据
    sum_num = 0  # 用于保存数据的总和
    count_num = 0  # 用于保存数据的总个数
    for i in range(len(sample_data)):
        # 进行数据过滤,满足条件才进行计算
        if sample_data[i][index] == filter_num:
            if treat_type == 'add':
                for j in range(2, len(sample_data[i])):
                    sum_num += sample_data[i][j]

            elif treat_type == 'avg':
                for j in range(2, len(sample_data[i])):
                    sum_num += sample_data[i][j]
                    count_num += 1
    # 输出结果
    if treat_type == 'add':
        return sum_num
    elif treat_type == 'avg':
        return sum_num / count_num


'''
    创建时间: 2022/9/26
    完成时间: 2022/9/27
    功能: 蒙特卡罗算法single函数
    关键词：single函数
    修改时间：
'''


def monte_carlo_single(flow_data, cut_rate):
    """
    对模拟生成的所有特征进行计算，输出目标值
    :param dict flow_data: 各车型对应的流量数据
    :param list cut_rate: 根据事故点位置计算出来的车辆分布概率
    :return:
    """
    # 各车型对应的车身长范围
    vehicle_length = kp.get_parameter_with_keyword('vehicle_length')
    # 各车型对应的车身长范围
    gap_length = kp.get_parameter_with_keyword('gap_length')

    # 各车型特征赋值
    vehicle_type = ['1', '2', '3', '4', '11', '12', '13', '14', '15', '16']

    # 生成各车型对应数量的数组
    vehicle_num = [0 if flow_data[v_type] < 0 else flow_data[v_type] for v_type in vehicle_type]

    # 创建样本数据
    sample_data = create_sample_first(vehicle_type, vehicle_num, [1, 0], cut_rate)

    # 补充样本的车长度特征
    sample_data = create_sample_feature(sample_data, vehicle_length, random_type='normality', feature_type='float')

    # 补充样本的车间距长度特征
    sample_data = create_sample_feature(sample_data, gap_length, random_type='normality', feature_type='float')

    # 进行所有特征的计算
    total_length = compute_sample_feature(sample_data, 1, 1, 'add')

    # 对特征计算结果进行处理，并输出最终结果
    return round(total_length / 4, 1)


'''
    创建时间: 2022/10/12
    完成时间: 2022/10/12
    功能: 蒙特卡罗算法总函数
    关键词：总函数
    修改时间：
'''


def monte_carlo_main(flow_data, cut_rate, simple_num=500):
    """
    蒙特卡罗算法总函数
    :param dict flow_data: 各车型对应的流量数据
    :param list cut_rate: 根据事故点位置计算出来的车辆分布概率
    :param int simple_num:
    :return:
    """
    simple_data = []
    for i in range(simple_num):
        simple_data.append(monte_carlo_single(flow_data, cut_rate))
    # return round(sum(simple_data) / simple_num, 1)
    return max(simple_data)


if __name__ == '__main__':
    vehicle_type = ['1', '2', '3', '4', '11', '12', '13', '14', '15', '16']
    have_data = dbf.get_dict_from_document('./4.statistic_data/basic_data/820_have_type.csv',
                                           [0, 1, 2, 5], encoding='utf-8', ifIndex=False, key_length=3, key_for_N=False)
    data_result = []
    start_time = '2022-08-07 00:00:00'
    end_time = '2022-08-07 23:55:00'
    now_time = '2022-08-07 00:00:00'
    for i in range(288):
        if now_time > end_time:
            break
        print(now_time)
        type_dict = {}
        for vt in vehicle_type:
            try:
                type_dict[vt] = float(have_data['G003061003000820-' + now_time + '-' + vt])
            except:
                type_dict[vt] = 0

        # vehicle_flow = {'1': 2120, '2': 5, '3': 40, '4': 20, '11': 25, '12': 20, '13': 41, '14': 5, '15': -3, '16': 0, '21': 0, '22': 0, '23': 0,
        #                 '24': 0, '25': 0, '26': 0}
        data_result.append(['G003061003000820', now_time, monte_carlo_main(type_dict, [1, 0])])
        now_time = datetime.datetime.strftime(datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S') +
                                              datetime.timedelta(minutes=5), '%Y-%m-%d %H:%M:%S')

    with open('./4.statistic_data/basic_data/820_length_ls.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_result)
