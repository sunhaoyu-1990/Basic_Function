import csv
import math

import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import time
import GetData as gd
import Data_Basic_Function as dbf
import Feature_of_Portray as fp
from scipy.stats import *
import Keyword_and_Parameter as kp
import Document_process as dop
import scipy_function as sf
import Data_process as dp
import time as ti
import datetime
import warnings
import sys

sys.setrecursionlimit(3000)

warnings.filterwarnings("ignore")

'''
    文档创建时间：2021/09/26
    用于车辆稽查中间表的特征数据转换
'''

'''
    创建时间：2021/10/25
    完成时间：2021/10/25
    修改时间：No.1 2021/10/26，增加循环，循环输出所有路径数据
            No.2 2021/11/13，加入最小路径数据表拆分
            No.3 2021/11/15，入口ID和出口ID有空值的出现，现加入当有空值出现时，就将最短路径相应特征设置为nan
'''


def gantryNum_of_shortPath(data, data_ls):
    """
    找出从Enid出发到Exid的最短路径和经过门架个数
    :param data:DataFrame数据
    :param Enids:起始收费站ID列表
    :param Exids:终点收费站ID列表
    :param Vehicle:
    :return:
    """
    data_ls['combine_short'] = data_ls['入口ID'] + data_ls['出口ID'] + data_ls['出口车型'].map(
        lambda x: str(int(x)))

    data_short = data[['combine', 'FEE', 'PM', 'SGROUP', 'GAN_NUM']]
    data_short = data_short.drop_duplicates(['combine'])
    data_short = data_short.set_index('combine')
    data_whole = data_ls.set_index('combine_short')
    data_whole_short = pd.merge(left=data_whole, right=data_short, how='left', left_index=True, right_index=True)

    return data_whole_short


'''
    创建时间：2021/10/25
    完成时间：2021/10/25
    修改时间：No.1 2022/2/23，增加了对车牌的处理，将车牌定为出口车牌（不带颜色），只对有出口信息的数据增加，分别分离出车牌的颜色
            No.2 2022/2/24，增加了data_enter, data_exit的输入数据，修改了入口数据字段、出口数据字段和门架数据字段
                            
'''


def basic_information_of_vehicle_new(data_gantry, data_enter, data_exit):
    """
    将一天内的所有行驶记录（每个PASSID为一条行驶记录）及其基础信息提取出来
    :param data_gantry: 门架数据
    :param data_enter: 入口收费数据
    :param data_exit: 出口收费数据
    :return:
    """
    # 缺少过程数据
    data_in = data_enter[['PASSID', 'VEHICLEPLATE', 'IDENTIFYVEHICLEID', 'ENSTATIONFX', 'ENSTATIONHEX',
                          'ENTIME', 'MEDIATYPE', 'VEHICLETYPE', 'VEHICLECLASS', 'ENWEIGHT']]

    # 2022/2/23更新，增加’入口HEX码‘
    data_in.columns = ['PASSID', '入口车牌(全)', '入口识别车牌', '入口ID', '入口HEX码', '入口时间', '入口通行介质', '入口车型',
                       '入口车种', '入口重量']
    data_in = data_in.set_index('PASSID')

    # 2022/2/24修改，增加了从出口数据中提取的入口信息
    data_out = data_exit[['PASSID', 'ENVEHICLEPLATE', 'ENIDENTIFYVEHICLEID', 'ENSTATIONFX', 'ENSTATIONHEX', 'ENTIME',
                          'ENVEHICLETYPE', 'ENVEHICLECLASS', 'ENWEIGHT', 'ENAXLECOUNT', 'EXVEHICLEPLATE',
                          'EXIDENTIFYVEHICLEID', 'EXSTATIONFX', 'EXTIME', 'MEDIATYPE', 'EXVEHICLETYPE',
                          'EXVEHICLECLASS', 'EXWEIGHT', 'EXAXLECOUNT', 'OBUVEHICLETYPE', 'OBUSN', 'ETCCARDID',
                          'EXITFEETYPE', 'OBUVEHICLEPLATE', 'CPUVEHICLEPLATE', 'FEE']]
    data_out.columns = ['PASSID', '入口车牌(全)(出口)', '入口识别车牌(出口)', '入口ID(出口)', '入口HEX码(出口)', '入口时间(出口)',
                        '入口车型(出口)', '入口车种(出口)', '入口重量(出口)', '入口轴数', '出口车牌(全)', '出口识别车牌', '出口ID',
                        '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号',
                        '出口计费方式', '出口OBU车牌', '出口CPU车牌', 'pay_fee']
    data_out = data_out.set_index('PASSID')

    # 2022/2/24增加，增加了从出口数据中提取的入口信息

    # 2022/2/24修改，增加了对车牌的处理，增加出入口车牌（不带颜色）、车牌颜色和车牌（全）
    data_out[['出口车牌(全)', '入口车牌(全)(出口)']] = data_out[['出口车牌(全)', '入口车牌(全)(出口)']].fillna('_')
    data_in['入口车牌(全)'] = data_in['入口车牌(全)'].fillna('_')

    data_out['出口车牌'] = data_out['出口车牌(全)'].map(lambda x: x.split('_')[0])
    data_out['出口车牌颜色'] = data_out['出口车牌(全)'].map(lambda x: x.split('_')[1])

    data_out['入口车牌(出口)'] = data_out['入口车牌(全)(出口)'].map(lambda x: x.split('_')[0])
    data_out['入口车牌颜色(出口)'] = data_out['入口车牌(全)(出口)'].map(lambda x: x.split('_')[1])
    data_in['入口车牌'] = data_in['入口车牌(全)'].map(lambda x: x.split('_')[0])
    data_in['入口车牌颜色'] = data_in['入口车牌(全)'].map(lambda x: x.split('_')[1])

    data_noinout = data_gantry.groupby(['PASSID', 'VEHICLEPLATE', 'IDENTIFYVEHICLEID', 'ENTOLLSTATIONHEX', 'ENTIME',
                                        'MEDIATYPE', 'VEHICLETYPE', 'VEHICLECLASS', 'ENWEIGHT'])[['GANTRYID']].count()
    data_noinout = data_noinout.sort_values(['GANTRYID'], ascending=False)
    data_noinout = data_noinout.reset_index()
    data_noinout = data_noinout.groupby(['PASSID', 'VEHICLEPLATE', 'IDENTIFYVEHICLEID', 'ENTOLLSTATIONHEX', 'ENTIME',
                                         'MEDIATYPE', 'VEHICLETYPE', 'VEHICLECLASS', 'ENWEIGHT']).head(1)
    data_noinout = data_noinout[['PASSID', 'VEHICLEPLATE', 'IDENTIFYVEHICLEID', 'ENTOLLSTATIONHEX', 'ENTIME',
                                 'MEDIATYPE', 'VEHICLETYPE', 'VEHICLECLASS', 'ENWEIGHT']]
    # 2022/2/24增加，增加了对车牌的处理，增加门架的入口车牌（不带颜色）、车牌颜色和车牌（全）
    data_noinout.columns = ['PASSID', '入口车牌(全)', '入口识别车牌', '入口HEX码', '入口时间',
                            '入口通行介质', '入口车型', '入口车种', '入口重量']
    data_noinout['入口车牌(全)'] = data_noinout['入口车牌(全)'].fillna('')

    # 由于门架入口车牌（全）有特殊异常情况，对其进行遍历处理
    out_vehicle_all = data_noinout['入口车牌(全)'].values
    out_vehicle = []
    out_color = []
    for vehicle in out_vehicle_all:
        if vehicle == '':
            out_vehicle.append('')
            out_color.append('')
        else:
            vehicle_list = vehicle.split('_')
            try:
                out_color.append(vehicle_list[1])
                out_vehicle.append(vehicle_list[0])
            except:
                out_vehicle.append(vehicle)
                out_color.append('')
    data_noinout['入口车牌'] = out_vehicle
    data_noinout['入口车牌颜色'] = out_color

    data_noinout['middle_type'] = 'none'

    data_all = pd.merge(left=data_in, right=data_out, how='outer', left_index=True, right_index=True)

    # 2022/2/24增加，把合并后的数据分为，有入有出、有入无出和无入有出3种,并分别取值和改字段名成
    data_all_whole = data_all[(data_all['入口HEX码'].notnull()) & (data_all['出口ID'].notnull())]
    data_all_in = data_all[(data_all['入口HEX码'].notnull()) & (data_all['出口ID'].isnull())]
    data_all_out = data_all[(data_all['入口HEX码'].isnull()) & (data_all['出口ID'].notnull())]

    data_all_whole = data_all_whole[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                     '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                     '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种',
                                     '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
                                     '出口CPU车牌', 'pay_fee']]
    data_all_whole['middle_type'] = 'whole'
    data_all_in = data_all_in[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                               '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                               '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                               '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                               'pay_fee']]
    data_all_in['middle_type'] = 'in'
    data_all_out = data_all_out[['入口车牌(全)(出口)', '入口识别车牌(出口)', '入口车牌(出口)', '入口车牌颜色(出口)', '入口ID(出口)',
                                 '入口HEX码(出口)', '入口时间(出口)', '出口通行介质', '入口车型(出口)', '入口车种(出口)',
                                 '入口重量(出口)', '入口轴数', '出口车牌(全)', '出口识别车牌', '出口车牌', '出口车牌颜色', '出口ID',
                                 '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量', '出口轴数', 'OBU车型', 'OBU设备号',
                                 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌', 'pay_fee']]
    data_all_out.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                            '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                            '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                            '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                            'pay_fee']
    data_all_out['middle_type'] = 'out'
    # 将3种类型的数据进行合并
    data_all = pd.concat((data_all_whole, data_all_in, data_all_out), axis=0)

    data_all['pay_fee'] = data_all['pay_fee'].fillna(0)
    data_all = data_all.reset_index()
    data_all = pd.concat((data_all, data_noinout), axis=0)
    data_all = data_all.drop_duplicates(['PASSID'])

    return data_all


'''
    创建时间：2021/11/4
    完成时间：2021/11/4
    修改时间：
'''


def compare_of_two_rows(data, features, threshold, new_feature, threshole_scope=[1, 0], type=0):
    """
    针对data的features的两列值，进行每个值的相似度计算，并根据阈值进行赋值，高于阈值赋值0，低于阈值赋值1，结果保存在新的列名下
    :param type:
    :param threshole_scope:
    :param data:
    :param features:
    :param threshold:
    :param new_feature:
    :return:
    """
    # 将空值转变为空字符串，方便后面字符串对比计算
    data = data.fillna('')
    firts_list = data[features[0]].values
    second_list = data[features[1]].values
    if type == 0:  # 如果type为0，则进行相似度计算
        result_list = dbf.get_similarity_of_two_list(firts_list, second_list)
    elif type == 1:  # 如果type为1，则进行大小的比较
        result_list = judge_list_bigger_or_smaller(firts_list, second_list, type=1)

    data[new_feature] = dbf.data_transform_by_threshold(result_list, threshold, threshole_scope)

    return data


'''
    创建时间：2021/10/25
    完成时间：2021/10/25
    修改时间：No.1 2021/11/11，1.将for循环计算改为直接DataFrame赋值
                             2.添加了是否计算相似度的情况，similarity=0为只对比是否相同，similarity=1为不相同的在进行相似度对比
'''


def judge_rows_value(data, features, new_features, similarity=0):
    """
    循环对照数据中各特征组的数值是否相同
    :param similarity:如果similarity为0，则只进行相同对比，不相同即判断为不一致, 如果similarity为1，则进行相同对比后，不相同的再进行相似度的判断
    :param data:原始数据
    :param features:需要对比的特征组
    :param new_features:对应的每次对比结果，添加的新列名称
    :return: 结果新增到新列中，相同添加0，不同添加1，有空值添加2
    """

    for i in range(len(features)):
        # 输入数据和要对比的列名，得到对比后的结果，并赋值给新的一列
        # 将数据分为入口车牌和出口车牌一致和不一致两类，一致的直接赋值0，不一致分为有空值和无空值情况，有空值的赋0，无空值的进行相似度比较
        # 获取入口车牌和出口车牌一致数据
        data_same = data[data[features[i][0]] == data[features[i][1]]]
        data_same[new_features[i]] = 0
        # 获取入口车牌和出口车牌不一致但有空值的数据
        data_notsame_havenone = data[
            (data[features[i][0]] != data[features[i][1]]) & (
                    (data[features[i][0]] == '') | (data[features[i][1]] == ''))]
        data_notsame_havenone[new_features[i]] = 0
        # 获取入口车牌和出口车牌不一致数据
        data_notsame = data[
            (data[features[i][0]] != data[features[i][1]]) & (data[features[i][0]] != '') & (
                    data[features[i][1]] != '')]
        if similarity == 0:  # 如果similarity为0，则只进行相同对比，不相同即判断为不一致
            data_notsame[new_features[i]] = 1
        elif similarity == 1:  # 如果similarity为1，则进行相同对比后，不相同的再进行相似度的判断
            # 2022/2/23修改，将threshold值从0.7改为0.8
            data_notsame = compare_of_two_rows(data_notsame, [features[i][0], features[i][1]], 0.8, new_features[i])
        else:
            # 如果similarity为2，则进行大小比较，如果前一个小于后一个返回1，反之返回0
            data_notsame = compare_of_two_rows(data_notsame, [features[i][0], features[i][1]], 0.5, new_features[i],
                                               threshole_scope=[0, 1], type=1)
            # 进行判断值是否在有效范围内
            threshold = [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]
            data_notsame[new_features[i]] = dbf.data_transform_by_threshold(data_notsame[features[i][1]].values,
                                                                            threshold,
                                                                            data_notsame[new_features[i]].values)

            data_notsame_havenone[new_features[i]] = 0

        data = pd.concat((data_same, data_notsame_havenone, data_notsame), axis=0)

    return data


'''
    创建时间：2022/2/25
    完成时间：2022/2/25
    功能：对输入的数组进行批量判断大小
    修改时间：
'''


def judge_list_bigger_or_smaller(list1, list2, type=0):
    """
    对输入的数组进行批量判断大小
    :param list1: 数组1
    :param list2: 数组2
    :return:
    """
    result = []
    for i in range(len(list2)):
        if list1[i] < list2[i]:
            if type == 0:
                result.append(0)
            else:
                result.append(1)
        else:
            if type == 0:
                result.append(1)
            else:
                result.append(0)
    return result


'''
    创建时间：2021/10/26
    完成时间：2021/10/26
    修改时间：No.1 2021/10/29，更改获取相应passid的所有数据部分语句，将data['BIZID'] != '0'更改
            No.2 2021/11/2，添加门架经过时间串的输出
            No.3 2021/11/3，增加将门架行驶时长转为分钟代码
'''


def path_of_gantry_data(data, passid):
    """
    获取每个passID的门架路径、门架个数、门架总费用和最大门架行驶时间和门架时间串
    :param data:原始数据
    :param passid:passid列表
    :return:
    """
    # 给所有空值赋值为0，用于后续判断是否为空，用是否为0来替换
    data = data.fillna(0)
    # 建立空数组，装载每个passid的门架路径、门架个数、门架总费用和最大门架行驶时间
    data_all = []
    # 循环所有的passid
    for i, pid in enumerate(passid):
        # 获取相应passid的所有数据，并且去除掉干扰数据
        data_passid = data[
            (data['PASSID'] == pid) & ((data['FEE'] > 0) | ((data['FEE'] == 0) & (data['BIZID'] != 0)))].sort_values(
            ['TRANSTIME'], ascending=True)

        # 如果获取到的数据为空，不进行下面的计算，直接跳到下一个
        if data_passid.empty:
            continue

        # 通过过门架时间 减去 上一门架时间，得到过程行驶时间。如果是入口门架，则没有上一门架时间，则减去入口时间
        try:
            data_passid['门架行驶时长'] = data_passid['TRANSTIME'] - data_passid['LASTGANTRYTIME']
        except:
            data_passid['门架行驶时长'] = data_passid['TRANSTIME'] - data_passid['ENTIME']

        # 获取同一个passid所有门架ID的列表
        gantry_id = data_passid['GANTRYID'].values

        # 将门架行驶时长转为分钟，2021/11/3
        data_passid['门架行驶时长(分钟)'] = data_passid['门架行驶时长'].map(lambda x: x.total_seconds() / 60.0)

        # 获取同一个passid所有门架经过时间，修改时间2021/11/2
        gantry_time = data_passid['TRANSTIME'].values
        # 创建空字符串，用于将门架经过时间进行串联
        time = ''
        # 将所有的门架ID用 | 相隔，进行组合
        for i, gt in enumerate(gantry_time):
            ts = pd.to_datetime(str(gt))
            d = ts.strftime('%Y-%m-%d %H:%M:%S')
            if i == 0:
                time = d
            else:
                time = time + '|' + d

        # 创建空字符串，用于将门架ID进行串联
        path = ''

        # 计算经过的门架个数
        gantry_num = len(gantry_id)

        # 将所有的门架ID用 | 相隔，进行组合
        for i, gantry in enumerate(gantry_id):
            if i == 0:
                path = gantry[:-3]
            else:
                path = path + '|' + gantry[:-3]
        # 计算行程总费用
        fee = data_passid['FEE'].sum()
        # 获取所有门架行驶时长中最长的时间
        max_time = data_passid['门架行驶时长'].max()
        # 获取所有门架行驶时长中最长的时间（分钟），2021/11/3
        max_time_minites = data_passid['门架行驶时长(分钟)'].max()
        # 将所有计算出的数据组合成为数组，添加到data_all中
        data_all.append([pid, path, gantry_num, fee, max_time, max_time_minites, time])

    data_all = pd.DataFrame(data_all, columns=['PASSID', '门架路径', '门架数', '门架费用', '最大门架行驶时间', '最大门架行驶时间(分钟)', '门架时间串'])
    data_all = data_all.set_index('PASSID')

    return data_all


'''
    创建时间：2021/11/8
    完成时间：2021/11/8
    修改时间：No.1 2021/11/10，门架数据中添加新字段，入口时间、出口时间，用于后续的牌识路径
            No.2 2022/1/6, add the vehicle type and class from gantry data
            No.3 2022/2/24，删除了门架车型和门架车种的字段，将type修改为GANTRYTYPE
'''


def path_of_gantry_data_new(data):
    """
    获取到每个行驶记录的路径相关信息
    :param data: DataFrame类型数据
    :return:
    """
    # 给所有空值赋值为0，用于后续判断是否为空，用是否为0来替换
    data[['FEE', 'BIZID']] = data[['FEE', 'BIZID']].fillna(0)
    # 获取相应passid的所有数据，并且去除掉干扰数据
    # data = data[((data['FEE'] > 0) | ((data['FEE'] == 0) & (data['BIZID'] != 0)))].sort_values(['TRANSTIME'],
    #                                                                                            ascending=True)
    data = data.sort_values(['TRANSTIME'], ascending=True).drop(['BIZID'], axis=1)

    # get the list of DataFrame
    data = data.values

    data_dict = {}
    for i in range(len(data)):
        try:
            data_dict[data[i][0]].append(list(data[i]))
        except:
            data_dict[data[i][0]] = [list(data[i])]

    data_total = []
    for i, key in enumerate(data_dict.keys()):
        data_key = data_dict[key]
        transtime_string = ''
        gantry_string = ''
        interval_string = ''
        gantryType_string = ''
        starttime = data_key[0][3].strftime('%Y-%m-%d %H:%M:%S')
        endtime = data_key[-1][3].strftime('%Y-%m-%d %H:%M:%S')
        fee_sum = 0
        fee_path = ''
        length_sum = 0
        for j in range(len(data_key)):
            if j != len(data_key) - 1:
                if len(data_key[j][2]) < 16:
                    continue
                else:
                    gantry_string_new = data_key[j][2][:16]
                if (data_key[j + 1][3] - data_key[j][3]).total_seconds() <= 15 and data_key[j + 1][2][:14] == data_key[j][2][:14]:
                    if data_key[j][1] == 0:
                        continue
                    elif data_key[j + 1][1] == 0:
                        data_key[j], data_key[j + 1] = data_key[j + 1], data_key[j]
                        continue
                    else:
                        if j > 0:
                            if data_key[j - 1][2][14:16] == data_key[j][2][14:16]:
                                data_key[j], data_key[j + 1] = data_key[j + 1], data_key[j]
                                continue
                            else:
                                continue
                        elif j < len(data_key) - 2:
                            if data_key[j][2][14:16] == data_key[j + 2][2][14:16]:
                                data_key[j], data_key[j + 1] = data_key[j + 1], data_key[j]
                                continue
                            else:
                                continue
                        else:
                            continue

                else:
                    transtime_string = transtime_string + data_key[j][3].strftime('%Y-%m-%d %H:%M:%S') + '|'
                    gantry_string = gantry_string + gantry_string_new + '|'
                    interval_string = interval_string + data_key[j][5] + '|'
                    if '2' in data_key[j][7]:
                        gantryType_string = gantryType_string + '2|'
                    elif '3' in data_key[j][7]:
                        gantryType_string = gantryType_string + '3|'
                    else:
                        gantryType_string = gantryType_string + data_key[j][7] + '|'
                    fee_sum += data_key[j][1]
                    fee_path = fee_path + str(data_key[j][1]) + '|'
                    length_sum += data_key[j][6]
            else:
                if len(data_key[j][2]) < 16:
                    if len(gantry_string) != 0:
                        gantry_string = gantry_string[:-1]
                        interval_string = interval_string[:-1]
                        gantryType_string = gantryType_string[:-1]
                        fee_path = fee_path[:-1]
                    continue
                else:
                    gantry_string_new = data_key[j][2][:16]
                transtime_string += data_key[j][3].strftime('%Y-%m-%d %H:%M:%S')
                gantry_string += gantry_string_new
                interval_string += data_key[j][5]
                if '2' in data_key[j][7]:
                    gantryType_string += '2'
                elif '3' in data_key[j][7]:
                    gantryType_string += '3'
                else:
                    gantryType_string += data_key[j][7]
                fee_sum += data_key[j][1]
                fee_path += str(data_key[j][1])
                length_sum += data_key[j][6]
        gan_num = len(interval_string.split('|'))
        data_total.append([key, gantry_string, interval_string, gan_num, transtime_string, gantryType_string, fee_sum,
                           fee_path, length_sum, starttime, endtime])

    data_all = pd.DataFrame(data_total, columns=['PASSID', '门架路径', '收费单元路径', '门架数', '门架时间串', '门架类型串',
                                                 '门架费用', '门架费用串', '总里程', '入口门架时间', '出口门架时间'])
    data_all = data_all.set_index('PASSID')

    return data_all


'''
    创建时间：2021/10/26
    完成时间：2021/10/26
    内容:检查每辆车的第一天和最后一条通行记录，如果有缺失，即删除
    修改时间：No.1 2021/11/19,增加 是否端口为省界的判断,如果缺失出入口,但只要端口为省界,也算完整数据,已测试
            No.2 2022/2/24，根据新的字段进行修改
            No.3 2022/5/10, delete some code
'''


def del_first_end_of_vehicle_new(data):
    """
    检查每辆车的第一天和最后一条通行记录，如果有缺失，即删除
    :param data:
    :return:
    """
    # problem应用 '出口车牌' 和 '入口车牌' 会造成数据缺失
    ls_noout = data[(data['middle_type'] == 'in')]
    ls_out = data[(data['middle_type'] == 'out') | (data['middle_type'] == 'whole')]
    # 2021/11/14，加上无入无出的车辆分类
    ls_none = data[(data['middle_type'] == 'none')]

    ls_out['车牌'] = ls_out['出口车牌']
    ls_out['车牌(全)'] = ls_out['出口车牌(全)']
    ls_out['结尾时间'] = ls_out['出口门架时间']
    ls_noout['车牌'] = ls_noout['入口车牌']
    ls_noout['车牌(全)'] = ls_noout['入口车牌(全)']
    ls_noout['结尾时间'] = ls_noout['出口门架时间']
    # 2021/11/14，无入无出的车辆的车牌和结尾时间赋值
    ls_none['车牌'] = ls_none['入口车牌']
    ls_none['车牌(全)'] = ls_none['入口车牌(全)']
    ls_none['结尾时间'] = ls_none['出口门架时间']
    basic_of_gantry = pd.concat((ls_noout, ls_out, ls_none), axis=0)
    basic_of_gantry = basic_of_gantry.reset_index()
    # 将车牌和结尾时间进行排序
    basic_of_gantry = basic_of_gantry.sort_values(['车牌', '结尾时间'], ascending=True)

    # 获取每个车牌的第一条数据
    data_first = basic_of_gantry.groupby(['车牌']).head(1)
    # 获取每个车牌的最后一条数据
    data_end = basic_of_gantry.groupby(['车牌']).tail(1)

    # 区分出单独一条的数据
    data_dup = pd.concat((data_first, data_end), axis=0)
    data_dup = data_dup[data_dup.duplicated()]
    data_dup = data_dup[(data_dup['middle_type'] == 'none') & (data_dup['是否端口为省界'] == 0)]
    dup_index = list(data_dup.index.values)

    # 获取第一条数据中入口信息缺失和数据的index
    data_first_noin = data_first[(data_first['middle_type'] == 'out') & (data_first['是否端口为省界'] == 0)]
    data_first_pro_noin = data_first[(data_first['middle_type'] == 'none') & (data_first['是否端口为省界'] == 3)]
    pro_noin_index = list(data_first_pro_noin.index.values)
    noin_index = list(data_first_noin.index.values)

    # 获取最后一条数据中出口信息缺失和数据的index
    data_end_noout = data_end[(data_end['middle_type'] == 'in') & (data_end['是否端口为省界'] == 0)]
    data_end_pro_noout = data_end[(data_end['middle_type'] == 'none') & (data_end['是否端口为省界'] == 2)]
    pro_noout_index = list(data_end_pro_noout.index.values)
    noout_index = list(data_end_noout.index.values)

    # 合并两缺失index，并去重
    del_index = list(set(noin_index + noout_index + dup_index + pro_noin_index + pro_noout_index))

    # 获取删除后的数据
    print(len(basic_of_gantry))
    data_normal = basic_of_gantry.loc[del_index, :]
    print(len(data_normal))
    try:
        data_normal = data_normal.drop(['index'], axis=1)
    except:
        print('no_index')
    data_abnormal = basic_of_gantry.drop(del_index, axis=0)
    print(len(data_abnormal))
    try:
        data_abnormal = data_abnormal.drop(['index'], axis=1)
    except:
        print('no_index')

    return data_normal, data_abnormal


'''
    创建时间：2021/10/26
    完成时间：2021/10/31
    修改时间：No.1 2021/11/1：1.之前是分别计算出入口的牌识完整度和匹配度，目前将出入口的牌识路径进行了合并，即比较出入口牌识路径，一样则用入口牌识路径，
            不一样，用长度较长的牌识路径。对比取出后的牌识路径命名为“牌识路径”，并以这个为基础进行牌识完整度和匹配度的计算。
                            2.增加了牌识路径不一的字段，即将出入口牌识路径都有且不相同的行驶记录，赋值为1，怀疑有套牌风险。
            No.2 2021/11/2：1.修改牌识路径不一的判断逻辑，之前是只要出入口牌识路径都有且不相同就记录为1，现在是出入口牌识路径都有且没有包含关系，就赋值为1。
            No.3 2021/11/11：对输入的数据进行分类，有异常的在进行特征判断，无异常的直接分配合格的标识
            No.4 2021/11/18，将门架占比的计算进行拆分计算在合并，已测试
            No.5 2021/11/21，加入出入口为省界的车辆稽查特征计算,只
            No.6 2022/1/4, add the compute process of province data
            No.7 2022/2/10，增加对ETC车辆的收费单元路径和最短路径的虚拟门架ID去除，用于后续判断路径是否缺失
            No.8 2022/2/18，增加对车辆收费单元路径的路径类型判断，判断是否为U型、J型、往复还是正常路径
            No.9 2022/2/25，针对更新的基础数据字段进行新的特征判断和修改就字段名称
            No.10 2022/3/6, add the charge result of if path is whole
            No.11 2022/3/20, add the different treat for whole data and province data
            N0.12 2022/4/20, gantry out time don't have the head and tail gantry
'''


def contrast_imformation_of_vehicle_noiden(data, data_type='whole'):
    """
    将数据进行二次处理，输出门架路径完整度、车牌、出入口车牌是否一致、行驶时间等
    :param data: 原始数据
    :param data_type: type of data,whole or province or province_in
    :return:
    problem：加入入口各识别车牌和出口各识别车牌对比
    """
    # 得到出入口的车牌、入口识别车牌、出口识别车牌、出口OBU车牌和出口ETC卡车牌的对比匹配情况，2022/2/25更新
    # 临时加入，后续出入口车牌不去掉后两位
    if data_type == 'whole':  # 判断是否是对省内数据处理
        # 将所有可能为空的字符串字段进行，空值填充为‘’
        data[['入口ID', '入口车牌', '入口车牌颜色', '出口ID', '门架路径', '门架时间串', '门架类型串', 'SGROUP', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
              '出口CPU车牌', '收费单元路径']] = data[['入口ID', '入口车牌', '入口车牌颜色', '出口ID', '门架路径', '门架时间串', '门架类型串', 'SGROUP', 'OBU车型', 'OBU设备号', 'ETC卡号',
                                  '出口计费方式', '出口OBU车牌', '出口CPU车牌', '收费单元路径']].fillna('')
        # 将所有可能为空的数值字段，空值填充为0
        data[['门架数', '入口轴数', '出口轴数', '入口重量', '出口重量']] = data[['门架数', '入口轴数', '出口轴数', '入口重量', '出口重量']].fillna(0)

    else:  # 判断是否是对跨省数据处理
        data[['入口ID', '出口ID', '门架路径', '门架时间串', '门架类型串', 'SGROUP', '收费单元路径']] = data[['入口ID', '出口ID', '门架路径', '门架时间串', '门架类型串', 'SGROUP', '收费单元路径']].fillna('')
        data['门架数'] = data['门架数'].fillna(0)

    if data_type == 'whole':  # 判断是否是对省内数据处理
        print('开始进行各识别车牌的匹配----------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))

        # 判断出入车牌的相似性，2022/2/25修改，去掉了’入口车牌识别一致性‘、’出口车牌识别一致性‘、’入口OBU一致性‘和’入口CPU一致性‘
        data = judge_rows_value(data, [['入口车牌', '出口车牌']], ['出入车牌是否匹配'], similarity=1)
        # 判断出入车牌、出入车牌颜色、OBU车牌、ETC卡车牌、出入通行介质和出入车型的一致性，即是否完全相同，2022/2/25添加
        data = judge_rows_value(data, [['入口车牌', '出口车牌'], ['入口车牌颜色', '出口车牌颜色'], ['出口车牌(全)', '出口OBU车牌'],
                                       ['出口车牌(全)', '出口CPU车牌'], ['入口车型', '出口车型']],
                                ['出入车牌是否相同', '出入车牌颜色是否相同', 'OBU车牌是否相同', 'ETC卡车牌是否相同', '出入车型是否相同'],
                                similarity=0)

        # 2022/3/13 add,for ETC
        # 2022/4/12 change
        data_etc = data[data['出口通行介质'] == 1]
        data_noetc = data[data['出口通行介质'] != 1]
        # 判断OBU车型是否高于计费车型, 2022/2/25添加
        data_etc = judge_rows_value(data_etc, [['出口车型', 'OBU车型']], ['obu车型是否高于计费车型'], similarity=2)
        data_noetc['obu车型是否高于计费车型'] = 0
        data = pd.concat((data_etc, data_noetc), axis=0)

        # 判断是否采用最小费额计费, 2022/2/25添加
        data['是否采用最小费额计费'] = data['出口计费方式'].map(lambda x: 0 if x == 6.0 else 1)

        # 判断出口车型是否小于出口轴数, 2022/2/25添加
        axle_disc = {11: 2, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 21: 2, 22: 2, 23: 3, 24: 4, 25: 5, 26: 6}
        data_cargo = data[(data['出口车型'] >= 11) & (data['出口车型'] <= 25) & (data['出口车种'] != 25) & (data['出口车型'] != 16)]
        data_cargo['出口车型轴数'] = data_cargo['出口车型'].map(lambda x: axle_disc[int(x)])
        data_other = data[(data['出口车型'] < 11) | (data['出口车型'] >= 26) | (data['出口车种'] == 25) | (data['出口车型'] == 16)]
        data_cargo = compare_of_two_rows(data_cargo, ['出口车型轴数', '出口轴数'], 0.5, '是否出口轴数大于出口车型',
                                         [0, 1], type=1)
        data_other['是否出口轴数大于出口车型'] = 0
        data_cargo = data_cargo.drop(['出口车型轴数'], axis=1)
        data = pd.concat((data_cargo, data_other), axis=0)

    else:
        print('开始进行出入口信息的匹配----------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
        data[['出入车牌是否匹配', '出入车牌是否相同', '出入车牌颜色是否相同', 'OBU车牌是否相同', 'ETC卡车牌是否相同', '出入车型是否相同',
              'obu车型是否高于计费车型', '是否采用最小费额计费', '是否出口轴数大于出口车型']] = 5
        data['GAN_NUM'] = data['GAN_NUM'].fillna(0)
        data['GAN_NUM'] = data['GAN_NUM'].map(lambda x: 0 if x == '' else int(x))
        data['总里程'] = data['总里程'].fillna(0)
        data['总里程'] = data['总里程'].map(lambda x: 0 if x == '' else int(x))
        data['PM'] = data['PM'].fillna(0)
        data['PM'] = data['PM'].map(lambda x: 0 if x == '' else int(x))
        data['FEE'] = data['FEE'].fillna(0)
        # 对最小费额费用进行四舍五入
        data['FEE'] = data['FEE'].map(lambda x: 0 if x == '' else round(float(x)/100) * 100)

    # 中间数据表各别字段的处理
    if data_type == 'whole':
        print('开始进行个别中间字段的处理---------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
        data['pay_fee'] = data['pay_fee'].fillna(0)
        # 对门架费用进行四舍五入
        data['门架费用'] = data['门架费用'].map(lambda x: 0 if x == '' else round(x) * 100)
        # 对实收费用进行四舍五入
        data['pay_fee'] = data['pay_fee'].map(lambda x: 0 if x == '' else round(x) * 100)
        data['GAN_NUM'] = data['GAN_NUM'].fillna(0)
        data['GAN_NUM'] = data['GAN_NUM'].map(lambda x: 0 if x == '' else int(x))
        data['总里程'] = data['总里程'].fillna(0)
        data['总里程'] = data['总里程'].map(lambda x: 0 if x == '' else int(x))
        data['PM'] = data['PM'].fillna(0)
        data['PM'] = data['PM'].map(lambda x: 0 if x == '' else int(x))
        data['FEE'] = data['FEE'].fillna(0)
        # 对最小费额费用进行四舍五入
        data['FEE'] = data['FEE'].map(lambda x: 0 if x == '' else round(float(x)/100) * 100)

    # 进行门架和牌识的完整度和匹配度计算-----------------------
    # 分离出门架路径和SGROUP一样的和不一样的数据

    print('开始进行门架的完整度和匹配度计算----', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
    # 2022/3/6 add, get every unit's next unit
    gantry_relation_list = dbf.get_dict_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True)

    # 2022/3/13 add, get the fee unit with its' visual or not
    gantry_service_list = dbf.get_dict_from_document('../Data_Origin/gantry_service.csv',
                                                     ['BEFOREGANTRYID', 'GANTRYID', 'DISTANCE', 'SERVICEAREISNOT'],
                                                     encoding='gbk', length=16, key_length=2)

    # 进行牌识和门架数占比计算,2021/11/18修改，将门架占比的计算进行拆分计算在合并
    data_big = data[data['GAN_NUM'] > 2]
    data_little = data[(data['GAN_NUM'] <= 2) & (data['GAN_NUM'] > 0)]
    data_none = data[data['GAN_NUM'] == 0]
    data_little['门架数占比'] = data_little['门架数'] / data_little['GAN_NUM']
    data_big['门架数占比'] = data_big['门架数'].map(lambda x: x - 2 if x >= 2 else 0) / data_big['GAN_NUM'].map(
        lambda x: x - 2)
    data_none['门架数占比'] = 1
    data = pd.concat((data_big, data_little, data_none), axis=0)

    # 2022/3/8添加，用于判断路径是否完整
    # 获取数据上路径完整的数据, 2022/3/27 change
    if data_type == 'whole':
        data_ab = data[((data['类型'] == 'province_in') | (data['类型'] == 'whole')) & (data['本省入口站id'] == '')]
        data = data[((data['类型'] != 'province_in') & (data['类型'] != 'whole')) | (data['本省入口站id'] != '')]
    else:
        data_ab = data[((data['类型'] == 'province_out') | (data['类型'] == 'province_pass')) & (data['本省入口站id'] == '')]
        data = data[((data['类型'] != 'province_out') & (data['类型'] != 'province_pass')) | (data['本省入口站id'] != '')]
    data_whole = data[((data['总里程'] == data['PM']))]  # 2022/4/15修改，将(data['门架数占比'] != 1.0)的条件删除
    data_noWhole = data[((data['总里程'] != data['PM']))]  # 2022/4/15修改，将(data['门架数占比'] != 1.0)的条件删除
    data_noWhole_0 = data_noWhole[data_noWhole['门架数占比'] == 0.0]
    data_noWhole = data_noWhole[data_noWhole['门架数占比'] != 0.0]
    data_noWhole['id_in'] = data_noWhole['入口ID'].map(lambda x: x[:11] if len(x) > 11 else x)
    data_noWhole['id_out'] = data_noWhole['出口ID'].map(lambda x: x[:11] if len(x) > 11 else x)
    data_whole_2 = data_noWhole[(data_noWhole['id_in'] == 'G0065610080') | (data_noWhole['id_out'] == 'G0065610080')].drop(['id_in', 'id_out'], axis=1)
    # 获取路径不完整的数据
    data_noWhole = data_noWhole[(data_noWhole['id_in'] != 'G0065610080') & (data_noWhole['id_out'] != 'G0065610080')].drop(['id_in', 'id_out'], axis=1)

    # 2022/3/28 add, get the out and no province gantry data

    # 获取收费单元路径的数组
    interval_string_list = data_noWhole['收费单元路径'].values
    # 2022/3/6 add, 对疑似不完整的路径进行完整情况判断
    if_path_whole_list = []
    for i in range(len(interval_string_list)):
        # 将门架数组转换为字典类型
        interval_disc = dbf.basic_list_to_disc(interval_string_list[i].split('|'), addType='disc')
        # 将门架路径字典与标准路径字典进行比较
        if_path_whole_list.append(dbf.basic_compare_disc_match_other(gantry_relation_list, interval_disc,
                                                                     value_list=interval_string_list[i].split('|')))

    # 路径完整的赋值为0
    data_noWhole_0['路径是否完整'] = 2
    data_ab['路径是否完整'] = 1
    data_whole['路径是否完整'] = 0
    data_whole_2['路径是否完整'] = 0

    # 疑似缺失的数据赋值为判定的结果，如果完整为0，不完整为1
    data_noWhole['路径是否完整'] = if_path_whole_list
    data_isWhole = data_noWhole[data_noWhole['路径是否完整'] == 0]
    # 将判定后不完整的数据分离出来，进行完整度和匹配度的计算
    data_noWhole = data_noWhole[data_noWhole['路径是否完整'] == 1]
    data_noWhole_noshort = data_noWhole[data_noWhole['GAN_NUM'] == 0]
    data_noWhole = data_noWhole[data_noWhole['GAN_NUM'] != 0]

    # 门架路径和SGROUP不一样的数据进行完整度和匹配度计算
    match_gan, integrity_gan = match_of_path(data_noWhole, ['收费单元路径', 'SGROUP'])
    # ------------------------------------------------
    # 给门架路径和SGROUP不一样的数据进行赋值
    data_noWhole['门架完整度'] = integrity_gan
    data_noWhole['门架匹配度'] = match_gan

    # 2022/3/8添加，将路径完整的数据的完整度和匹配度直接赋值为1
    data_whole['门架完整度'] = 1
    data_whole['门架匹配度'] = 1
    data_whole_2['门架完整度'] = 1
    data_whole_2['门架匹配度'] = 1
    data_noWhole_0['门架完整度'] = 0
    data_noWhole_0['门架匹配度'] = 0
    data_noWhole_noshort['门架完整度'] = 1
    data_noWhole_noshort['门架匹配度'] = 1
    data_isWhole['门架完整度'] = 1
    data_isWhole['门架匹配度'] = 1
    data_ab['门架完整度'] = 0.1
    data_ab['门架匹配度'] = 0.1

    data_4 = pd.concat((data_whole, data_isWhole, data_noWhole_0, data_noWhole, data_ab, data_noWhole_noshort, data_whole_2), axis=0)

    # -----------------------------------------------------------------------------------------------
    if data_type == 'whole':
        # 得到费用是否匹配
        # 计算门架费用和实际收费的上下限差值
        print('开始进行门架和实收费用的匹配计算-----', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
        data_4['上限差值'] = data_4['门架费用'] * 1.1 - data_4['pay_fee']
        data_4['下限差值'] = data_4['门架费用'] * 0.8 - data_4['pay_fee']
        # 分别得到在上下限范围内 和 范围外的数据
        # 2022/3/8, 增加判断是否为免费车辆[8.0, 10.0, 14.0, 21.0, 23.0, 26.0]，如果是免费车同时实收费用为0，则判断为正常
        # 2022/5/16, add the 15 16 17
        data_4_free = data_4[((data_4['出口车种'] == 8.0) | (data_4['出口车种'] == 10.0) | (data_4['出口车种'] == 14.0) | (data_4['出口车种'] == 15.0) | (data_4['出口车种'] == 16.0) | (data_4['出口车种'] == 17.0) |
                              (data_4['出口车种'] == 21.0) | (data_4['出口车种'] == 22.0) | (data_4['出口车种'] == 23.0) | (data_4['出口车种'] == 26.0)) & (
                                         data_4['pay_fee'] == 0)]
        # 2022/5/16 add
        data_4_free['是否采用最小费额计费'] == 1
        # 获取到不是免费车辆的数据
        # 2022/5/16, add the 15 16 17
        data_4 = data_4[((data_4['出口车种'] != 8.0) & (data_4['出口车种'] != 10.0) & (data_4['出口车种'] != 14.0) & (data_4['出口车种'] != 15.0) & (data_4['出口车种'] != 16.0) & (data_4['出口车种'] != 17.0) &
                         (data_4['出口车种'] != 21.0) & (data_4['出口车种'] != 22.0) & (data_4['出口车种'] != 23.0) & (data_4['出口车种'] != 26.0)) | (
                                    data_4['pay_fee'] != 0)]

        # 如果实收费用大于门架总费用的80%，同时费用不为0
        data_4_normal = data_4[
            (data_4['下限差值'] <= 0) & (data_4['pay_fee'] != 0) & (data_4['门架费用'] != 0) & (data_4['是否采用最小费额计费'] == 1)]
        # 非免费车辆中费用为0的车辆数据
        data_4_abnormal_0 = data_4[((data_4['pay_fee'] == 0) | (data_4['门架费用'] == 0)) & (data_4['是否采用最小费额计费'] == 1)]
        # 非免费车辆中实收费用小于门架费用的90%，算作异常数据
        data_4_abnormal = data_4[
            (data_4['下限差值'] > 0) & (data_4['pay_fee'] != 0) & (data_4['门架费用'] != 0) & (data_4['是否采用最小费额计费'] == 1)]
        # 采用最小费额计费的在后面进行判断
        data_4_abnormal_nothing = data_4[data_4['是否采用最小费额计费'] == 0]
        # 分别进行'费用是否匹配'的赋值
        data_4_free['门架费用是否匹配'] = 0
        data_4_normal['门架费用是否匹配'] = 0
        data_4_abnormal['门架费用是否匹配'] = 1
        data_4_abnormal_0['门架费用是否匹配'] = 2
        data_4_abnormal_nothing['门架费用是否匹配'] = 3
        data_4 = pd.concat((data_4_normal, data_4_abnormal, data_4_abnormal_0, data_4_abnormal_nothing), axis=0).drop(
            ['上限差值', '下限差值'], axis=1)

        # 计算门架费用和short费的上下限差值
        print('开始进行最小费额费用和实收费用的匹配计算-----', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
        data_4['上限'] = data_4['FEE'] * 1.1
        data_4['下限差值'] = data_4['FEE'] * 0.8 - data_4['pay_fee']
        # 如果实收费用大于最小费用的90%，同时费用不为0，同时门架总费用小于等于最小费用的110%
        data_4_normal = data_4[(data_4['门架费用'] <= data_4['上限']) & (data_4['下限差值'] <= 0) & (data_4['pay_fee'] != 0) & (
                    data_4['FEE'] != 0) & (data_4['是否采用最小费额计费'] == 0)]  # 2022/2/25更改

        data_4_abnormal_0 = data_4[((data_4['pay_fee'] == 0) | (data_4['FEE'] == 0)) & (data_4['是否采用最小费额计费'] == 0)]
        # 非免费车辆中实收费用小于最小费用的90%，算作异常数据
        data_4_abnormal = data_4[
            ((data_4['门架费用'] > data_4['上限']) | (data_4['下限差值'] > 0)) & (data_4['pay_fee'] != 0) & (
                        data_4['FEE'] != 0) & (data_4['是否采用最小费额计费'] == 0)]
        data_4_abnormal_nothing = data_4[data_4['是否采用最小费额计费'] == 1]
        # 分别进行'费用是否匹配'的赋值
        data_4_free['最小费额费用是否匹配'] = 0
        data_4_normal['最小费额费用是否匹配'] = 0
        data_4_abnormal['最小费额费用是否匹配'] = 1
        data_4_abnormal_0['最小费额费用是否匹配'] = 2
        data_4_abnormal_nothing['最小费额费用是否匹配'] = 3
        data_4 = pd.concat((data_4_free, data_4_normal, data_4_abnormal, data_4_abnormal_0, data_4_abnormal_nothing),
                           axis=0).drop(['上限差值', '上限', '下限差值'], axis=1)

    else:
        data_4[['门架费用是否匹配', '最小费额费用是否匹配']] = 5

    # 2022/3/13 add, compute the max time of every gantry for each vehicle
    data_4_nogantry = data_4[(data_4['门架数'] <= 2) & (data_4['GAN_NUM'] > 2)]
    data_4_gantry = data_4[(data_4['门架数'] > 2) | (data_4['GAN_NUM'] <= 2)]
    outRange_time = []
    outRange_time_path = []
    ifOutRange = []
    outRange_gantry = []
    outRange_gantry_path = []
    outRange_speed_path = []
    time_list = data_4_gantry['门架时间串'].values
    gantry_list = data_4_gantry['门架路径'].values
    data_type_list = data_4_gantry['类型'].values  # 2022/4/20 add

    for i, gantry_time in enumerate(time_list):
        outRange_outtime_single = []
        outRange_time_single = []
        outRange_gantry_single = []
        outRange_speed_single = []
        gantry_time_list = gantry_time.split('|')
        gantry_ID_list = gantry_list[i].split('|')
        gantry_time_list = [datetime.datetime.strptime(gan, '%Y-%m-%d %H:%M:%S') if gan != '' else '' for gan in gantry_time_list]  # 2022/5/18 change
        # 2022/4/20 add, delete the station gantry
        if data_type_list[i] == 'whole':
            gantry_time_list = gantry_time_list[1:-1]
            gantry_ID_list = gantry_ID_list[1:-1]
        elif data_type_list[i] == 'province_in':
            gantry_time_list = gantry_time_list[:-1]
            gantry_ID_list = gantry_ID_list[:-1]
        elif data_type_list[i] == 'province_out':
            gantry_time_list = gantry_time_list[1:]
            gantry_ID_list = gantry_ID_list[1:]
        if len(gantry_time_list) >= 2:
        # 2022/4/20-------------------------------------
            for j in range(len(gantry_time_list) - 1):
                # 2022/5/18 add, charge the time is ''
                if gantry_time_list[j] == '' or gantry_time_list[j + 1] == '':
                    continue
                try:
                    next_gantry = gantry_ID_list[j + 1][:11]
                    this_gantry = gantry_ID_list[j][:11]
                except:
                    continue
                if next_gantry == 'G3001610010' or this_gantry == 'G3001610010':
                    k = 20
                elif next_gantry == 'G0005610040' or this_gantry == 'G0005610040' or \
                        next_gantry == 'G0005610050' or this_gantry == 'G0005610050' or \
                        next_gantry == 'G0005610060' or this_gantry == 'G0005610060' or \
                        next_gantry == 'G0005610070' or this_gantry == 'G0005610070':
                    k = 40
                else:
                    k = 50
                try:
                    gantry_service_data = gantry_service_list[gantry_ID_list[j] + '-' + gantry_ID_list[j + 1]]
                    if gantry_service_data[1] == '1':
                        if ((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600) > (
                                (float(gantry_service_data[0]) / k) + 0.5):
                            # 2022/5/16 add
                            outRange_outtime_single.append(
                                round((((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600) - (
                                            (float(gantry_service_data[0]) / k) + 0.5)) * 60, 1))
                            # 2022/4/24 change, add round function and 3600 -> 60
                            outRange_time_single.append(round((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 60, 1))
                            outRange_gantry_single.append(gantry_ID_list[j] + '-' + gantry_ID_list[j + 1])
                            # 2022/4/24 add
                            outRange_speed_single.append(round(float(gantry_service_data[0])/((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600), 2))
                        else:
                            continue
                    else:
                        if ((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600) > (
                                float(gantry_service_data[0]) / k):
                            # 2022/5/16 add
                            outRange_outtime_single.append(
                                round((((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600) - (
                                            float(gantry_service_data[0]) / k)) * 60, 1))
                            # 2022/4/24 change, add round function and 3600 -> 60
                            outRange_time_single.append(
                                round((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 60, 1))
                            outRange_gantry_single.append(gantry_ID_list[j] + '-' + gantry_ID_list[j + 1])
                            # 2022/4/24 add
                            outRange_speed_single.append(round(float(gantry_service_data[0]) / (
                                        (gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600), 2))
                        else:
                            continue
                except:
                    if ((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 60) > 120:
                        # 2022/5/16 add
                        outRange_outtime_single.append(
                            round((((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600) - 2), 1))
                        # 2022/4/24 change, add round function and 3600 -> 60
                        outRange_time_single.append(
                            round((gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 60, 1))
                        outRange_gantry_single.append(gantry_ID_list[j] + '-' + gantry_ID_list[j + 1])
                        # 2022/4/24 add
                        outRange_speed_single.append(round(float(gantry_service_data[0]) / (
                                    (gantry_time_list[j + 1] - gantry_time_list[j]).total_seconds() / 3600), 2))
                    else:
                        continue

        if len(outRange_outtime_single) > 0:
            ifOutRange.append(1)
            max_time = max(outRange_outtime_single)
            outRange_time.append(max_time)
            outRange_gantry.append(outRange_gantry_single[outRange_outtime_single.index(max_time)])
            outRange_time_single = [str(x) for x in outRange_time_single]
            outRange_speed_single = [str(x) for x in outRange_speed_single]
            outRange_time_path.append(dbf.list_to_string_with_sign(outRange_time_single, '|'))
            outRange_gantry_path.append(dbf.list_to_string_with_sign(outRange_gantry_single, '|'))
            outRange_speed_path.append(dbf.list_to_string_with_sign(outRange_speed_single, '|'))  # 2022/4/24 add
        else:
            ifOutRange.append(0)
            outRange_time.append(0)
            outRange_gantry.append('')
            outRange_time_path.append('')
            outRange_gantry_path.append('')
            outRange_speed_path.append('')  # 2022/4/24 add

    data_4_gantry['最大门架超时时长'] = outRange_time
    data_4_gantry['是否门架行驶超时'] = ifOutRange
    data_4_gantry['最大超时门架区间'] = outRange_gantry
    data_4_gantry['超时门架时长串'] = outRange_time_path
    data_4_gantry['超时门架区间串'] = outRange_gantry_path
    data_4_gantry['超时门架速度串'] = outRange_speed_path  # 2022/4/24 add
    data_4_nogantry['最大门架超时时长'] = 0
    data_4_nogantry['是否门架行驶超时'] = 0
    data_4_nogantry['最大超时门架区间'] = ''
    data_4_nogantry['超时门架时长串'] = ''
    data_4_nogantry['超时门架区间串'] = ''
    data_4_nogantry['超时门架速度串'] = ''  # 2022/4/24 add
    data_4 = pd.concat((data_4_gantry, data_4_nogantry), axis=0)

    # 得到行驶时间,以小时计
    print('开始进行时速是的异常处理-----------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
    if data_type == 'whole':
        data_ab = data_4[((data_4['类型'] == 'province_in') | (data_4['类型'] == 'whole')) & (data_4['本省入口站id'] == '')]
        data_4 = data_4[((data_4['类型'] != 'province_in') & (data_4['类型'] != 'whole')) | (data_4['本省入口站id'] != '')]
    else:
        data_ab = data_4[((data_4['类型'] == 'province_out') | (data_4['类型'] == 'province_pass')) & (data_4['本省入口站id'] == '')]
        data_4 = data_4[((data_4['类型'] != 'province_out') & (data_4['类型'] != 'province_pass')) | (data_4['本省入口站id'] != '')]
    data_4['行驶时间'] = data_4['出口门架时间'] - data_4['入口门架时间']
    data_4['行驶时间'] = data_4['行驶时间'].map(lambda x: round(x.total_seconds() / 60.0, 0))  # 2022/4/24 change, add round function
    data_4['行驶时间'] = data_4['行驶时间'].fillna(0)
    # 得到short行驶时间,以小时计
    # 得到行驶时间和里程的匹配情况
    # 2022/3/13 add ，总里程在4以下的，自动判断为正常，其他数据进行时速的判断
    data_50_normal = data_4[((data_4['PM'] <= 50000) & (data_4['行驶时间'] <= 120))]
    data_50_abnormal = data_4[((data_4['PM'] <= 50000) & (data_4['行驶时间'] > 120))]
    data_50_200_normal = data_4[(data_4['PM'] > 50000) & (data_4['PM'] <= 200000) & (data_4['行驶时间'] <= (data_4['PM'] * 60 / 50000) + 1)]
    data_50_200_abnormal = data_4[(data_4['PM'] > 50000) & (data_4['PM'] <= 200000) & (data_4['行驶时间'] > (data_4['PM'] * 60 / 50000) + 1)]
    data_200_400_normal = data_4[
        ((data_4['PM'] > 200000) & (data_4['PM'] <= 400000) & (data_4['行驶时间'] <= (data_4['PM'] / 1000) + 2))]
    data_200_400_abnormal = data_4[
        ((data_4['PM'] > 200000) & (data_4['PM'] <= 400000) & (data_4['行驶时间'] > (data_4['PM'] / 1000) + 2))]
    data_400_800_normal = data_4[
        ((data_4['PM'] > 400000) & (data_4['PM'] <= 800000) & (data_4['行驶时间'] <= (data_4['PM'] / 1000) + 3))]
    data_400_800_abnormal = data_4[
        ((data_4['PM'] > 400000) & (data_4['PM'] <= 800000) & (data_4['行驶时间'] > (data_4['PM'] / 1000) + 3))]
    data_800_1200_normal = data_4[
        ((data_4['PM'] > 800000) & (data_4['PM'] <= 1200000) & (data_4['行驶时间'] <= (data_4['PM'] / 1000) + 4))]
    data_800_1200_abnormal = data_4[
        ((data_4['PM'] > 800000) & (data_4['PM'] <= 1200000) & (data_4['行驶时间'] > (data_4['PM'] / 1000) + 4))]
    data_1200_normal = data_4[((data_4['PM'] > 1200000) & (data_4['行驶时间'] <= (data_4['PM'] / 1000) + 5))]
    data_1200_abnormal = data_4[((data_4['PM'] > 1200000) & (data_4['行驶时间'] > (data_4['PM'] / 1000) + 5))]

    data_50_normal['是否最小费额路径超时'] = 0
    data_50_abnormal['是否最小费额路径超时'] = 1
    data_50_200_normal['是否最小费额路径超时'] = 0
    data_50_200_abnormal['是否最小费额路径超时'] = 1
    data_200_400_normal['是否最小费额路径超时'] = 0
    data_200_400_abnormal['是否最小费额路径超时'] = 1
    data_400_800_normal['是否最小费额路径超时'] = 0
    data_400_800_abnormal['是否最小费额路径超时'] = 1
    data_800_1200_normal['是否最小费额路径超时'] = 0
    data_800_1200_abnormal['是否最小费额路径超时'] = 1
    data_1200_normal['是否最小费额路径超时'] = 0
    data_1200_abnormal['是否最小费额路径超时'] = 1
    data_ab['是否最小费额路径超时'] = 0

    data_50_normal['最小费额路径超时时长'] = 0
    data_50_abnormal['最小费额路径超时时长'] = data_50_abnormal['行驶时间'] - 2
    data_50_200_normal['最小费额路径超时时长'] = 0
    data_50_200_abnormal['最小费额路径超时时长'] = data_50_200_abnormal['行驶时间'] - data_50_200_abnormal['PM'] * 60 / 50000 - 1
    data_200_400_normal['最小费额路径超时时长'] = 0
    data_200_400_abnormal['最小费额路径超时时长'] = data_200_400_abnormal['行驶时间'] - data_200_400_abnormal['PM'] / 1000 - 2
    data_400_800_normal['最小费额路径超时时长'] = 0
    data_400_800_abnormal['最小费额路径超时时长'] = data_400_800_abnormal['行驶时间'] - data_400_800_abnormal['PM'] / 1000 - 3
    data_800_1200_normal['最小费额路径超时时长'] = 0
    data_800_1200_abnormal['最小费额路径超时时长'] = data_800_1200_abnormal['行驶时间'] - data_800_1200_abnormal['PM'] / 1000 - 4
    data_1200_normal['最小费额路径超时时长'] = 0
    data_1200_abnormal['最小费额路径超时时长'] = data_1200_abnormal['行驶时间'] - data_1200_abnormal['PM'] / 1000 - 5
    data_ab['最小费额路径超时时长'] = 0

    data_4 = pd.concat((data_50_normal, data_50_abnormal, data_50_200_normal, data_50_200_abnormal,
                        data_200_400_normal, data_200_400_abnormal, data_400_800_normal, data_400_800_abnormal,
                        data_800_1200_normal, data_800_1200_abnormal, data_1200_normal, data_1200_abnormal, data_ab), axis=0)

    # 对车辆收费单元路径的路径类型判断，判断是否为U型、J型、往复还是正常路径,2022/2/18添加
    interval = data_4['收费单元路径'].values  # 得到所有车辆的收费单元路径
    path_type_list = []  # 保存各路径的类型结果
    for i in range(len(interval)):
        path_type = dbf.service_charge_Upath_Jpath_Cyclepath(interval[i])
        if path_type == 0:  # 如果返回结果为0，则为正常路径
            path_type_list.append('正常路径')
        elif path_type == 1:  # 如果返回结果为1，判断为U型路径
            path_type_list.append('U型路径')
        elif path_type == 2:  # 如果返回结果为2，判断为J型路径
            path_type_list.append('J型路径')
        else:  # 如果返回结果为3，判断为往复型路径;  bug:如果返回结果是除了1到3的数字外，也会判断为往复型路径
            path_type_list.append('往复型路径')
    data_4['路径类型'] = path_type_list

    return data_4


'''
    创建时间：2021/11/5
    完成时间：2021/11/7
    函数功能：不同时间段的门架数据基础中间表合并函数
    修改时间：No.1 2022/2/24，根据新的字段情况进行修改
            No.2 2022/4/8，对跨省或者无入或无出的数据进行收费单元路径判断，是否首尾为省界单元，是否结尾为临近省界单元，并进行相应的处理
'''


def concat_middle_data_noiden(datas):
    """
    不同时间段的门架数据基础中间表合并函数
    :param datas: 门架数据集
    :return:
    """
    data_list = []  # 保存所有完整的数据
    # 将出入口信息均有的数据进行处理
    for i, data in enumerate(datas):
        data[['门架路径', '门架时间串', '门架类型串']] = data[['门架路径', '门架时间串', '门架类型串']].fillna('')
        data_whole = data[data['middle_type'] == 'whole']  # 获取出入口完整的数据
        data_list.append(data_whole)

    # 将所有的完整数据进行合并
    data_whole_all = dp.Combine_Document(data_list)

    data_whole_list = []  # 用于保存所有拼接后出入口完整的数据
    # 对出口或入口缺失，和出入口均缺失的数据进行处理
    for i, data in enumerate(datas):
        data[['门架路径', '门架时间串', '门架类型串', '门架费用串']] = data[['门架路径', '门架时间串', '门架类型串', '门架费用串']].fillna('')
        data[['门架数']] = data[['门架数']].fillna(0)
        data['门架类型串'] = data['门架类型串'].map(lambda x: str(x))
        data['门架费用串'] = data['门架费用串'].map(lambda x: str(x))
        # 如果i为0，为手数据，以该数据为底进行后续的合并
        if i == 0:
            # 分别获取缺失入口、缺失出口和出入口均缺失的数据
            data_out_basic = data[data['middle_type'] == 'out']  # 获取有出无入的数据
            data_in_basic = data[data['middle_type'] == 'in']  # 获取有入无出的数据
            data_middle_basic = data[data['middle_type'] == 'none']  # 获取无入无出的数据
            # 缺失入口数据的字段提取及重命名，方便后续合并
            data_out_basic = data_out_basic[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                             '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                             '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                                             '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                                             '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串', 'pay_fee', '总里程',
                                             '入口门架时间', '出口门架时间', 'middle_type']]
            data_out_basic.columns = ['入口车牌(全)_出', '入口识别车牌_出', '入口车牌_出', '入口车牌颜色_出', '入口ID_出', '入口HEX码_出', '入口时间_出',
                                      '入口通行介质_出', '入口车型_出', '入口车种_出', '入口重量_出', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                      '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                                      '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                                      '门架路径_出', '收费单元路径_出', '门架数_出', '门架费用_出', '门架费用串_出', '门架时间串_出', '门架类型串_出',
                                      'pay_fee', '总里程_出', '入口门架时间_出', '出口门架时间_出', 'middle_type_出']
            # 缺失出口数据的字段提取及重命名，方便后续合并
            data_in_basic = data_in_basic[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                           '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数',
                                           '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_in_basic.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                     '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径_入', '收费单元路径_入',
                                     '门架数_入', '门架费用_入', '门架费用串_入', '门架时间串_入', '门架类型串_入', '总里程_入', '入口门架时间_入', '出口门架时间_入',
                                     'middle_type_入']
            # 缺失出入口数据的字段提取及重命名，方便后续合并
            data_middle_basic = data_middle_basic[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                                   '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数',
                                                   '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_middle_basic.columns = ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中', '入口时间_中',
                                         '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径_中', '收费单元路径_中', '门架数_中',
                                         '门架费用_中', '门架费用串_中', '门架时间串_中', '门架类型串_中', '总里程_中', '入口门架时间_中', '出口门架时间_中',
                                         'middle_type_中']

        # 如果i不为0，说明是之后的中间表，将其与之前的底数据进行合并
        else:
            data_out_next = data[data['middle_type'] == 'out']
            data_in_next = data[data['middle_type'] == 'in']
            data_middle_next = data[data['middle_type'] == 'none']

            data_out_next = data_out_next[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                           '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                           '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                                           '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                                           '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串', 'pay_fee', '总里程',
                                           '入口门架时间', '出口门架时间', 'middle_type']]
            data_out_next.columns = ['入口车牌(全)_出', '入口识别车牌_出', '入口车牌_出', '入口车牌颜色_出', '入口ID_出', '入口HEX码_出', '入口时间_出',
                                     '入口通行介质_出', '入口车型_出', '入口车种_出', '入口重量_出', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                     '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                                     '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                                     '门架路径_出', '收费单元路径_出', '门架数_出', '门架费用_出', '门架费用串_出', '门架时间串_出', '门架类型串_出',
                                     'pay_fee', '总里程_出', '入口门架时间_出', '出口门架时间_出', 'middle_type_出']

            data_in_next = data_in_next[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                         '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数',
                                         '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_in_next.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                    '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径_入', '收费单元路径_入',
                                    '门架数_入', '门架费用_入', '门架费用串_入', '门架时间串_入', '门架类型串_入', '总里程_入', '入口门架时间_入', '出口门架时间_入',
                                    'middle_type_入']

            data_middle_next = data_middle_next[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                                 '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数',
                                                 '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_middle_next.columns = ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中', '入口时间_中',
                                        '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径_中', '收费单元路径_中', '门架数_中',
                                        '门架费用_中', '门架费用串_中', '门架时间串_中', '门架类型串_中', '总里程_中', '入口门架时间_中', '出口门架时间_中',
                                        'middle_type_中']

            # 先用basic的只有入口数据的对应next的出口数据
            data_in_basic_all = pd.merge(left=data_in_basic, right=data_out_next, how='outer', left_index=True,
                                         right_index=True)
            # 获取匹配后出入完整的数据
            data_in_basic_whole = data_in_basic_all[
                (data_in_basic_all['middle_type_入'].notnull()) & (data_in_basic_all['middle_type_出'].notnull())]
            # 获取匹配后入口缺失的数据，用于后续与底座中间表中出入均无数据的匹配
            data_in_basic_out = data_in_basic_all[
                (data_in_basic_all['middle_type_入'].isnull()) & (data_in_basic_all['middle_type_出'].notnull())]
            # 获取匹配后出口缺失的数据，用于和next数据中出入均无数据的匹配
            data_in_basic_in = data_in_basic_all[
                (data_in_basic_all['middle_type_入'].notnull()) & (data_in_basic_all['middle_type_出'].isnull())]

            # 将完整数据中的门架和牌识路径信息进行合并
            data_in_basic_whole['入口门架时间'] = data_in_basic_whole['入口门架时间_入']
            data_in_basic_whole['出口门架时间'] = data_in_basic_whole['出口门架时间_出']
            data_in_basic_whole['middle_type'] = 'whole'  # 完整的middle_type赋值为 whole，2022/2/24
            data_in_basic_whole_1 = data_in_basic_whole[data_in_basic_whole['门架数_出'] != 0]
            data_in_basic_whole_2 = data_in_basic_whole[data_in_basic_whole['门架数_出'] == 0]

            data_in_basic_whole_1['门架路径'] = data_in_basic_whole_1['门架路径_入'].map(lambda x: x + '|' if x != '' else x) + \
                                            data_in_basic_whole_1['门架路径_出']
            data_in_basic_whole_1['收费单元路径'] = data_in_basic_whole_1['收费单元路径_入'].map(
                lambda x: x + '|' if x != '' else x) + \
                                              data_in_basic_whole_1['收费单元路径_出']
            data_in_basic_whole_1['门架数'] = data_in_basic_whole_1['门架数_入'] + data_in_basic_whole_1['门架数_出']
            data_in_basic_whole_1['门架费用'] = data_in_basic_whole_1['门架费用_入'] + data_in_basic_whole_1['门架费用_出']
            data_in_basic_whole_1['门架费用串'] = data_in_basic_whole_1['门架费用串_入'].map(lambda x: x + '|' if x != '' else x) + data_in_basic_whole_1['门架费用串_出']
            data_in_basic_whole_1['门架时间串'] = data_in_basic_whole_1['门架时间串_入'].map(lambda x: x + '|' if x != '' else x) + \
                                             data_in_basic_whole_1['门架时间串_出']
            data_in_basic_whole_1['门架类型串'] = data_in_basic_whole_1['门架类型串_入'].map(lambda x: x + '|' if x != '' else x) + \
                                             data_in_basic_whole_1['门架类型串_出']
            data_in_basic_whole_1['总里程'] = data_in_basic_whole_1['总里程_入'] + data_in_basic_whole_1['总里程_出']

            data_in_basic_whole_2['门架路径'] = data_in_basic_whole_2['门架路径_入']
            data_in_basic_whole_2['收费单元路径'] = data_in_basic_whole_2['收费单元路径_入']
            data_in_basic_whole_2['门架数'] = data_in_basic_whole_2['门架数_入']
            data_in_basic_whole_2['门架费用'] = data_in_basic_whole_2['门架费用_入']
            data_in_basic_whole_2['门架费用串'] = data_in_basic_whole_2['门架费用串_入']
            data_in_basic_whole_2['总里程'] = data_in_basic_whole_2['总里程_入']
            data_in_basic_whole_2['门架时间串'] = data_in_basic_whole_2['门架时间串_入']
            data_in_basic_whole_2['门架类型串'] = data_in_basic_whole_2['门架类型串_入']
            data_in_basic_whole = pd.concat((data_in_basic_whole_1, data_in_basic_whole_2), axis=0)

            # 从匹配后的各字段中提取所需字段
            data_in_basic_whole = data_in_basic_whole[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                                       '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                                       '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种',
                                                       '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
                                                       '出口CPU车牌', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串',
                                                       'pay_fee', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_in_basic_whole.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                           '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                                           '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种',
                                           '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
                                           '出口CPU车牌', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串', 'pay_fee',
                                           '总里程', '入口门架时间', '出口门架时间', 'middle_type']
            data_whole_list.append(data_in_basic_whole)

            # 在用对应出口数据后的结果对应中间数据
            data_in_basic_in_all = pd.merge(left=data_in_basic_in, right=data_middle_next, how='outer', left_index=True,
                                            right_index=True)
            # 获取到与入口数据匹配的中间过程数据，但仍缺失出口数据，合并后作为新的底座入口数据
            data_in_basic_in_com = data_in_basic_in_all[
                (data_in_basic_in_all['middle_type_入'].notnull()) & (data_in_basic_in_all['middle_type_中'].notnull())]
            # 获取到入口数据，作为新的底座入口数据
            data_in_basic_in_in = data_in_basic_in_all[
                (data_in_basic_in_all['middle_type_入'].notnull()) & (data_in_basic_in_all['middle_type_中'].isnull())]
            # 获取到没有匹配到入口的中间数据，用于后续与底座中间表中出入均无数据的匹配，先进行列名的更改，方便后续匹配
            data_in_basic_in_middle = data_in_basic_in_all[
                (data_in_basic_in_all['middle_type_入'].isnull()) & (data_in_basic_in_all['middle_type_中'].notnull())]

            data_in_basic_in_middle = data_in_basic_in_middle[
                ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中', '入口时间_中',
                 '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径_中', '收费单元路径_中', '门架数_中',
                 '门架费用_中', '门架费用串_中', '门架时间串_中', '门架类型串_中', '总里程_中', '入口门架时间_中', '出口门架时间_中', 'middle_type_中']]
            data_in_basic_in_middle.columns = ['入口车牌(全)_中2', '入口识别车牌_中2', '入口车牌_中2', '入口车牌颜色_中2', '入口ID_中2',
                                               '入口HEX码_中2', '入口时间_中2', '入口通行介质_中2', '入口车型_中2', '入口车种_中2', '入口重量_中2',
                                               '门架路径_中2', '收费单元路径_中2', '门架数_中2', '门架费用_中2', '门架费用串_中2', '门架时间串_中2', '门架类型串_中2',
                                               '总里程_中2', '入口门架时间_中2', '出口门架时间_中2', 'middle_type_中2']

            # 将匹配的入口和中间数据的门架和牌识路径信息进行合并
            data_in_basic_in_com['入口门架时间'] = data_in_basic_in_com['入口门架时间_入']
            data_in_basic_in_com['出口门架时间'] = data_in_basic_in_com['出口门架时间_中']
            data_in_basic_in_com['middle_type'] = 'in'  # 完整的middle_type赋值为 whole，2022/2/24
            data_in_basic_in_com_1 = data_in_basic_in_com[data_in_basic_in_com['门架数_中'] != 0]
            data_in_basic_in_com_2 = data_in_basic_in_com[data_in_basic_in_com['门架数_中'] == 0]
            data_in_basic_in_com_1['门架路径'] = data_in_basic_in_com_1['门架路径_入'].map(lambda x: x + '|' if x != '' else x) + \
                                             data_in_basic_in_com_1['门架路径_中']
            data_in_basic_in_com_1['收费单元路径'] = data_in_basic_in_com_1['收费单元路径_入'].map(
                lambda x: x + '|' if x != '' else x) + \
                                               data_in_basic_in_com_1['收费单元路径_中']
            data_in_basic_in_com_1['门架数'] = data_in_basic_in_com_1['门架数_入'] + data_in_basic_in_com_1['门架数_中']
            data_in_basic_in_com_1['门架费用'] = data_in_basic_in_com_1['门架费用_入'] + data_in_basic_in_com_1['门架费用_中']
            data_in_basic_in_com_1['门架费用串'] = data_in_basic_in_com_1['门架费用串_入'].map(
                lambda x: x + '|' if x != '' else x) + data_in_basic_in_com_1['门架费用串_中']
            data_in_basic_in_com_1['总里程'] = data_in_basic_in_com_1['总里程_入'] + data_in_basic_in_com_1['总里程_中']
            data_in_basic_in_com_1['门架时间串'] = data_in_basic_in_com_1['门架时间串_入'].map(
                lambda x: x + '|' if x != '' else x) + data_in_basic_in_com_1['门架时间串_中']
            data_in_basic_in_com_1['门架类型串'] = data_in_basic_in_com_1['门架类型串_入'].map(
                lambda x: x + '|' if x != '' else x) + data_in_basic_in_com_1['门架类型串_中']

            data_in_basic_in_com_2['门架路径'] = data_in_basic_in_com_2['门架路径_入']
            data_in_basic_in_com_2['收费单元路径'] = data_in_basic_in_com_2['收费单元路径_入']
            data_in_basic_in_com_2['门架数'] = data_in_basic_in_com_2['门架数_入']
            data_in_basic_in_com_2['门架费用'] = data_in_basic_in_com_2['门架费用_入']
            data_in_basic_in_com_2['门架费用串'] = data_in_basic_in_com_2['门架费用串_入']
            data_in_basic_in_com_2['总里程'] = data_in_basic_in_com_2['总里程_入']
            data_in_basic_in_com_2['门架时间串'] = data_in_basic_in_com_2['门架时间串_入']
            data_in_basic_in_com_2['门架类型串'] = data_in_basic_in_com_2['门架类型串_入']
            data_in_basic_in_com = pd.concat((data_in_basic_in_com_1, data_in_basic_in_com_2), axis=0)

            # 去除所需字段
            data_in_basic_in_com = data_in_basic_in_com[
                ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                 '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数',
                 '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            # 更改列名用于和入口数据合并
            data_in_basic_in_com.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                            '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径_入', '收费单元路径_入',
                                            '门架数_入', '门架费用_入', '门架费用串_入', '门架时间串_入', '门架类型串_入', '总里程_入', '入口门架时间_入', '出口门架时间_入',
                                            'middle_type_入']
            #
            data_in_basic_in_in = data_in_basic_in_in[['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                                       '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径_入', '收费单元路径_入',
                                                       '门架数_入', '门架费用_入', '门架费用串_入', '门架时间串_入', '门架类型串_入', '总里程_入', '入口门架时间_入',
                                                       '出口门架时间_入', 'middle_type_入']]

            data_in_basic = pd.concat((data_in_basic_in_com, data_in_basic_in_in, data_in_next), axis=0)

            # 用basic的中间数据的对应之前对应后的出口数据
            data_middle_basic_all = pd.merge(left=data_middle_basic, right=data_in_basic_out, how='outer',
                                             left_index=True, right_index=True)
            # 获取到与出口数据匹配的中间过程数据，但仍缺失入口数据，合并后作为新的底座出口数据
            data_middle_basic_out_com = data_middle_basic_all[
                (data_middle_basic_all['middle_type_中'].notnull()) & (data_middle_basic_all['middle_type_出'].notnull())]
            # 获取到没有匹配到出口数据的中间数据，用于后续与next中间表中出入均无数据的匹配
            data_middle_basic_middle = data_middle_basic_all[
                (data_middle_basic_all['middle_type_中'].notnull()) & (data_middle_basic_all['middle_type_出'].isnull())]
            # 获取到纯出口数据，作为新的底座出口数据
            data_middle_basic_out = data_middle_basic_all[
                (data_middle_basic_all['middle_type_中'].isnull()) & (data_middle_basic_all['middle_type_出'].notnull())]

            # 将匹配出的出口数据和中间数据的路径进行合并，在和没有匹配到的出口数据和basic的出口数据，合并成新的basic出口数据
            data_middle_basic_out_com['入口门架时间'] = data_middle_basic_out_com['入口门架时间_出']
            data_middle_basic_out_com['出口门架时间'] = data_middle_basic_out_com['出口门架时间_出']
            data_middle_basic_out_com['middle_type'] = 'out'  # 完整的middle_type赋值为 out，2022/2/24
            data_middle_basic_out_com_1 = data_middle_basic_out_com[data_middle_basic_out_com['门架数_出'] != 0]
            data_middle_basic_out_com_2 = data_middle_basic_out_com[data_middle_basic_out_com['门架数_出'] == 0]
            data_middle_basic_out_com_1['门架路径'] = data_middle_basic_out_com_1['门架路径_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_out_com_1['门架路径_出']
            data_middle_basic_out_com_1['收费单元路径'] = data_middle_basic_out_com_1['收费单元路径_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_out_com_1['收费单元路径_出']
            data_middle_basic_out_com_1['门架数'] = data_middle_basic_out_com_1['门架数_中'] + data_middle_basic_out_com_1[
                '门架数_出']
            data_middle_basic_out_com_1['门架费用'] = data_middle_basic_out_com_1['门架费用_中'] + data_middle_basic_out_com_1[
                '门架费用_出']
            data_middle_basic_out_com_1['门架费用串'] = data_middle_basic_out_com_1['门架费用串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_out_com_1['门架费用串_出']
            data_middle_basic_out_com_1['总里程'] = data_middle_basic_out_com_1['总里程_中'] + data_middle_basic_out_com_1[
                '总里程_出']

            data_middle_basic_out_com_1['门架时间串'] = data_middle_basic_out_com_1['门架时间串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_out_com_1['门架时间串_出']
            data_middle_basic_out_com_1['门架类型串'] = data_middle_basic_out_com_1['门架类型串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_out_com_1['门架类型串_出']

            data_middle_basic_out_com_2['门架路径'] = data_middle_basic_out_com_2['门架路径_中']
            data_middle_basic_out_com_2['收费单元路径'] = data_middle_basic_out_com_2['收费单元路径_中']
            data_middle_basic_out_com_2['门架数'] = data_middle_basic_out_com_2['门架数_中']
            data_middle_basic_out_com_2['门架费用'] = data_middle_basic_out_com_2['门架费用_中']
            data_middle_basic_out_com_2['门架费用串'] = data_middle_basic_out_com_2['门架费用串_中']
            data_middle_basic_out_com_2['总里程'] = data_middle_basic_out_com_2['总里程_中']
            data_middle_basic_out_com_2['门架时间串'] = data_middle_basic_out_com_2['门架时间串_中']
            data_middle_basic_out_com_2['门架类型串'] = data_middle_basic_out_com_2['门架类型串_中']
            data_middle_basic_out_com = pd.concat((data_middle_basic_out_com_1, data_middle_basic_out_com_2), axis=0)

            data_middle_basic_out_com = data_middle_basic_out_com[
                ['入口车牌(全)_出', '入口识别车牌_中', '入口车牌_出', '入口车牌颜色_出', '入口ID_出', '入口HEX码_出', '入口时间_出',
                 '入口通行介质_中', '入口车型_出', '入口车种_出', '入口重量_出', '入口轴数', '出口车牌(全)', '出口识别车牌',
                 '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种',
                 '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
                 '出口CPU车牌', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串',
                 'pay_fee', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]

            data_middle_basic_out_com.columns = ['入口车牌(全)_出', '入口识别车牌_出', '入口车牌_出', '入口车牌颜色_出', '入口ID_出', '入口HEX码_出',
                                                 '入口时间_出', '入口通行介质_出', '入口车型_出', '入口车种_出', '入口重量_出', '入口轴数', '出口车牌(全)',
                                                 '出口识别车牌',
                                                 '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                                                 '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                                                 '门架路径_出', '收费单元路径_出', '门架数_出', '门架费用_出', '门架费用串_出', '门架时间串_出', '门架类型串_出',
                                                 'pay_fee', '总里程_出', '入口门架时间_出', '出口门架时间_出', 'middle_type_出']

            data_middle_basic_out = data_middle_basic_out[
                ['入口车牌(全)_出', '入口识别车牌_出', '入口车牌_出', '入口车牌颜色_出', '入口ID_出', '入口HEX码_出', '入口时间_出',
                 '入口通行介质_出', '入口车型_出', '入口车种_出', '入口重量_出', '入口轴数', '出口车牌(全)', '出口识别车牌',
                 '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种', '出口重量',
                 '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌', '出口CPU车牌',
                 '门架路径_出', '收费单元路径_出', '门架数_出', '门架费用_出', '门架费用串_出', '门架时间串_出', '门架类型串_出',
                 'pay_fee', '总里程_出', '入口门架时间_出', '出口门架时间_出', 'middle_type_出']]
            data_out_basic = pd.concat((data_middle_basic_out_com, data_middle_basic_out, data_out_basic), axis=0)

            # 在用对应中间数据后的结果对应中间数据
            data_middle_basic_middle_all = pd.merge(left=data_middle_basic_middle, right=data_in_basic_in_middle,
                                                    how='outer', left_index=True, right_index=True)
            # 获取匹配到的中间数据与中间数据，并进行合并作为新的底座中间数据
            data_middle_basic_middle_com = data_middle_basic_middle_all[
                (data_middle_basic_middle_all['middle_type_中'].notnull()) & (
                    data_middle_basic_middle_all['middle_type_中2'].notnull())]
            # basic中间数据，作为新的底座中间数据
            data_middle_basic_middle_middle1 = data_middle_basic_middle_all[
                (data_middle_basic_middle_all['middle_type_中'].notnull()) & (
                    data_middle_basic_middle_all['middle_type_中2'].isnull())]
            # next的中间数据，作为新的底座中间数据
            data_middle_basic_middle_middle2 = data_middle_basic_middle_all[
                (data_middle_basic_middle_all['middle_type_中'].isnull()) & (
                    data_middle_basic_middle_all['middle_type_中2'].notnull())]

            # 将匹配出的出口数据和中间数据的路径进行合并，在和没有匹配到的出口数据和basic的出口数据，合并成新的basic出口数据
            data_middle_basic_middle_com['入口门架时间'] = data_middle_basic_middle_com['入口门架时间_中']
            data_middle_basic_middle_com['出口门架时间'] = data_middle_basic_middle_com['出口门架时间_中2']
            data_middle_basic_middle_com['middle_type'] = 'none'  # 完整的middle_type赋值为 none，2022/2/24
            data_middle_basic_middle_com_1 = data_middle_basic_middle_com[data_middle_basic_middle_com['门架数_中2'] != 0]
            data_middle_basic_middle_com_2 = data_middle_basic_middle_com[data_middle_basic_middle_com['门架数_中2'] == 0]
            data_middle_basic_middle_com_1['门架路径'] = data_middle_basic_middle_com_1['门架路径_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_middle_com_1['门架路径_中2']
            data_middle_basic_middle_com_1['收费单元路径'] = data_middle_basic_middle_com_1['收费单元路径_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_middle_com_1['收费单元路径_中2']
            data_middle_basic_middle_com_1['门架数'] = data_middle_basic_middle_com_1['门架数_中'] + \
                                                    data_middle_basic_middle_com_1['门架数_中2']
            data_middle_basic_middle_com_1['门架费用'] = data_middle_basic_middle_com_1['门架费用_中'] + \
                                                     data_middle_basic_middle_com_1['门架费用_中2']
            data_middle_basic_middle_com_1['门架费用串'] = data_middle_basic_middle_com_1['门架费用串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_middle_com_1['门架费用串_中2']
            data_middle_basic_middle_com_1['总里程'] = data_middle_basic_middle_com_1['总里程_中'] + \
                                                    data_middle_basic_middle_com_1['总里程_中2']
            data_middle_basic_middle_com_1['门架时间串'] = data_middle_basic_middle_com_1['门架时间串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_middle_com_1['门架时间串_中2']
            data_middle_basic_middle_com_1['门架类型串'] = data_middle_basic_middle_com_1['门架类型串_中'].map(
                lambda x: x + '|' if x != '' else x) + data_middle_basic_middle_com_1['门架类型串_中2']

            data_middle_basic_middle_com_2['门架路径'] = data_middle_basic_middle_com_2['门架路径_中']
            data_middle_basic_middle_com_2['收费单元路径'] = data_middle_basic_middle_com_2['收费单元路径_中']
            data_middle_basic_middle_com_2['门架数'] = data_middle_basic_middle_com_2['门架数_中']
            data_middle_basic_middle_com_2['门架费用'] = data_middle_basic_middle_com_2['门架费用_中']
            data_middle_basic_middle_com_2['门架费用串'] = data_middle_basic_middle_com_2['门架费用串_中']
            data_middle_basic_middle_com_2['总里程'] = data_middle_basic_middle_com_2['总里程_中']
            data_middle_basic_middle_com_2['门架时间串'] = data_middle_basic_middle_com_2['门架时间串_中']
            data_middle_basic_middle_com_2['门架类型串'] = data_middle_basic_middle_com_2['门架类型串_中']
            data_middle_basic_middle_com = pd.concat((data_middle_basic_middle_com_1, data_middle_basic_middle_com_2),
                                                     axis=0)

            # 进行合并后的字段提取
            data_middle_basic_middle_com = data_middle_basic_middle_com[
                ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中', '入口时间_中',
                 '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径', '收费单元路径', '门架数',
                 '门架费用', '门架费用串', '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']]
            data_middle_basic_middle_com.columns = ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中',
                                                    '入口时间_中', '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径_中',
                                                    '收费单元路径_中', '门架数_中', '门架费用_中', '门架费用串_中', '门架时间串_中', '门架类型串_中', '总里程_中',
                                                    '入口门架时间_中', '出口门架时间_中', 'middle_type_中']
            data_middle_basic_middle_middle1 = data_middle_basic_middle_middle1[
                ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中', '入口HEX码_中', '入口时间_中',
                 '入口通行介质_中', '入口车型_中', '入口车种_中', '入口重量_中', '门架路径_中', '收费单元路径_中', '门架数_中',
                 '门架费用_中', '门架费用串_中', '门架时间串_中', '门架类型串_中', '总里程_中', '入口门架时间_中', '出口门架时间_中', 'middle_type_中']]
            data_middle_basic_middle_middle2 = data_middle_basic_middle_middle2[
                ['入口车牌(全)_中2', '入口识别车牌_中2', '入口车牌_中2', '入口车牌颜色_中2', '入口ID_中2', '入口HEX码_中2', '入口时间_中2',
                 '入口通行介质_中2', '入口车型_中2', '入口车种_中2', '入口重量_中2', '门架路径_中2', '收费单元路径_中2', '门架数_中2',
                 '门架费用_中2', '门架费用串_中2', '门架时间串_中2', '门架类型串_中2', '总里程_中2', '入口门架时间_中2', '出口门架时间_中2', 'middle_type_中2']]
            data_middle_basic_middle_middle2.columns = ['入口车牌(全)_中', '入口识别车牌_中', '入口车牌_中', '入口车牌颜色_中', '入口ID_中',
                                                        '入口HEX码_中', '入口时间_中', '入口通行介质_中', '入口车型_中', '入口车种_中',
                                                        '入口重量_中', '门架路径_中', '收费单元路径_中', '门架数_中', '门架费用_中', '门架费用串_中',
                                                        '门架时间串_中', '门架类型串_中', '总里程_中', '入口门架时间_中', '出口门架时间_中',
                                                        'middle_type_中']

            data_middle_basic = pd.concat(
                (data_middle_basic_middle_com, data_middle_basic_middle_middle1, data_middle_basic_middle_middle2),
                axis=0)

    # 全部中间数据跑完后，将完整的数据进行合并，形成完整路径数据集合
    data_whole_all_ls = dp.Combine_Document(data_whole_list)
    data_whole_all = pd.concat((data_whole_all, data_whole_all_ls), axis=0)

    # 将最后剩下的入口缺失、出口缺失和中间数据的列名更改
    data_middle_basic.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                                 '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串',
                                 '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']
    data_out_basic.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                              '入口通行介质', '入口车型', '入口车种', '入口重量', '入口轴数', '出口车牌(全)', '出口识别车牌',
                              '出口车牌', '出口车牌颜色', '出口ID', '出口时间', '出口通行介质', '出口车型', '出口车种',
                              '出口重量', '出口轴数', 'OBU车型', 'OBU设备号', 'ETC卡号', '出口计费方式', '出口OBU车牌',
                              '出口CPU车牌', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串', '门架时间串', '门架类型串',
                              'pay_fee', '总里程', '入口门架时间', '出口门架时间', 'middle_type']
    data_in_basic.columns = ['入口车牌(全)', '入口识别车牌', '入口车牌', '入口车牌颜色', '入口ID', '入口HEX码', '入口时间',
                             '入口通行介质', '入口车型', '入口车种', '入口重量', '门架路径', '收费单元路径', '门架数', '门架费用', '门架费用串',
                             '门架时间串', '门架类型串', '总里程', '入口门架时间', '出口门架时间', 'middle_type']

    # 2022/4/8增加，对跨省或者无入或无出的数据进行收费单元路径判断，是否首尾为省界单元，是否结尾为临近省界单元，并进行相应的处理
    # 进行收费单元的标准路径字典获取
    gantry_relation_list = dbf.get_dict_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True, key_for_N_type='list')

    # 进行收费单元的标准路径字典获取
    gantry_relation_disc = dbf.get_dict_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True)

    # 进行收费单元类型的字典获取
    interlist_type = dbf.get_dict_from_document('../Data_Origin/tollinterval.csv', ['id', 'provinceType'],
                                                encoding='utf-8')

    # 2021/11/19增加端口类型
    # 针对无出入口的数据
    data_middle_basic['门架类型串'] = data_middle_basic['门架类型串'].fillna('')
    data_middle_basic['门架端口类型'] = data_middle_basic['门架类型串'].map(lambda x: x[0] + x[-1] if len(x) >= 2 else x)

    data_middle_basic_1 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '23') | (data_middle_basic['门架端口类型'] == '22') | (
                data_middle_basic['门架端口类型'] == '32') | (data_middle_basic['门架端口类型'] == '33')]
    # 2022/4/10添加，对跨省数据进行首尾单元是否为省界单元的判断和处理
    data_middle_basic_1 = process_of_abnormal_progantry(data_middle_basic_1, '收费单元路径', '门架端口类型',
                                                        gantry_relation_list, interlist_type, 'all')
    data_middle_basic_3 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '20') | (data_middle_basic['门架端口类型'] == '2') | (
                data_middle_basic['门架端口类型'] == '30')]
    # 2022/4/10添加，对入省数据进行首尾单元是否为省界单元的判断和处理
    data_middle_basic_3 = process_of_abnormal_progantry(data_middle_basic_3, '收费单元路径', '门架端口类型',
                                                        gantry_relation_list, interlist_type, 'in')
    data_middle_basic_4 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '03') | (data_middle_basic['门架端口类型'] == '3') | (
                data_middle_basic['门架端口类型'] == '02')]
    # 2022/4/10添加，对出省数据进行首尾单元是否为省界单元的判断和处理
    data_middle_basic_4 = process_of_abnormal_progantry(data_middle_basic_4, '收费单元路径', '门架端口类型',
                                                        gantry_relation_list, interlist_type, 'out')
    data_middle_basic_2 = data_middle_basic[
        (data_middle_basic['门架端口类型'] != '23') & (data_middle_basic['门架端口类型'] != '20') &
        (data_middle_basic['门架端口类型'] != '03') & (data_middle_basic['门架端口类型'] != '2') &
        (data_middle_basic['门架端口类型'] != '3') & (data_middle_basic['门架端口类型'] != '22') &
        (data_middle_basic['门架端口类型'] != '32') & (data_middle_basic['门架端口类型'] != '30') &
        (data_middle_basic['门架端口类型'] != '02') & (data_middle_basic['门架端口类型'] != '33')]
    data_middle_basic = pd.concat((data_middle_basic_1, data_middle_basic_2, data_middle_basic_3, data_middle_basic_4),
                                  axis=0)

    # 2022/4/10添加，对结尾单元类型不是省界的进行向后追探省界单元（追一个单元）
    data_middle_basic = process_of_abnormal_progantry(data_middle_basic, '收费单元路径', '门架端口类型',
                                  gantry_relation_list, interlist_type, 'charge')

    data_middle_basic_1 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '23') | (data_middle_basic['门架端口类型'] == '22') | (
                    data_middle_basic['门架端口类型'] == '32') | (data_middle_basic['门架端口类型'] == '33')]

    data_middle_basic_3 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '20') | (data_middle_basic['门架端口类型'] == '2') | (
                    data_middle_basic['门架端口类型'] == '30')]

    data_middle_basic_4 = data_middle_basic[
        (data_middle_basic['门架端口类型'] == '03') | (data_middle_basic['门架端口类型'] == '3') | (
                    data_middle_basic['门架端口类型'] == '02')]

    data_middle_basic_2 = data_middle_basic[
        (data_middle_basic['门架端口类型'] != '23') & (data_middle_basic['门架端口类型'] != '20') &
        (data_middle_basic['门架端口类型'] != '03') & (data_middle_basic['门架端口类型'] != '2') &
        (data_middle_basic['门架端口类型'] != '3') & (data_middle_basic['门架端口类型'] != '22') &
        (data_middle_basic['门架端口类型'] != '32') & (data_middle_basic['门架端口类型'] != '30') &
        (data_middle_basic['门架端口类型'] != '02') & (data_middle_basic['门架端口类型'] != '33')]

    data_middle_basic_1['是否端口为省界'] = 1
    data_middle_basic_2['是否端口为省界'] = 0
    data_middle_basic_3['是否端口为省界'] = 2
    data_middle_basic_4['是否端口为省界'] = 3

    data_middle_basic = pd.concat((data_middle_basic_1, data_middle_basic_2, data_middle_basic_3, data_middle_basic_4),
                                  axis=0)

    # 针对无入口的数据
    data_out_basic['门架类型串'] = data_out_basic['门架类型串'].fillna('')
    data_out_basic['门架端口类型'] = data_out_basic['门架类型串'].map(lambda x: x[0] if x != '' else '')

    data_out_basic_1 = data_out_basic[
        (data_out_basic['门架端口类型'] == '2') | (data_out_basic['门架端口类型'] == '3')]

    # 2022/4/10添加，对跨省数据进行首单元是否为省界单元的判断和处理
    data_out_basic_1 = process_of_abnormal_progantry(data_out_basic_1, '收费单元路径', '门架端口类型',
                                                     gantry_relation_list, interlist_type, 'in')

    data_out_basic_2 = data_out_basic[
        (data_out_basic['门架端口类型'] != '2') & (data_out_basic['门架端口类型'] != '3')]

    data_out_basic_1['是否端口为省界'] = 1
    data_out_basic_2['是否端口为省界'] = 0

    data_out_basic = pd.concat((data_out_basic_1, data_out_basic_2), axis=0)

    # 针对无出口的数据
    data_in_basic['门架类型串'] = data_in_basic['门架类型串'].fillna('')
    data_in_basic['门架端口类型'] = data_in_basic['门架类型串'].map(lambda x: x[-1] if x != '' else '')

    # 2022/4/10添加，对结尾单元类型不是省界的进行向后追探省界单元（追一个单元）
    data_in_basic_1 = data_in_basic[
        (data_in_basic['门架端口类型'] == '2') | (data_in_basic['门架端口类型'] == '3')]
    data_in_basic_1 = process_of_abnormal_progantry(data_in_basic_1, '收费单元路径', '门架端口类型',
                                                    gantry_relation_list, interlist_type, 'out')
    data_in_basic_2 = data_in_basic[
        (data_in_basic['门架端口类型'] != '2') & (data_in_basic['门架端口类型'] != '3')]
    data_in_basic = pd.concat((data_in_basic_1, data_in_basic_2), axis=0)

    data_in_basic = process_of_abnormal_progantry(data_in_basic, '收费单元路径', '门架端口类型',
                                                  gantry_relation_list, interlist_type, 'charge')

    data_in_basic_1 = data_in_basic[
        (data_in_basic['门架端口类型'] == '2') | (data_in_basic['门架端口类型'] == '3')]

    # 2022/4/10添加，对跨省数据进行首单元是否为省界单元的判断和处理
    # data_in_basic_1 = process_of_abnormal_progantry(data_in_basic_1, '收费单元路径', '门架端口类型',
    #                                                 gantry_relation_list, interlist_type, 'out')

    data_in_basic_2 = data_in_basic[
        (data_in_basic['门架端口类型'] != '2') & (data_in_basic['门架端口类型'] != '3')]

    data_in_basic_1['是否端口为省界'] = 1
    data_in_basic_2['是否端口为省界'] = 0

    data_in_basic = pd.concat((data_in_basic_1, data_in_basic_2), axis=0)

    data_whole_all['是否端口为省界'] = 0

    # 将所有的数据进行合并，形成新的中间数据表
    datas_combine_all = pd.concat((data_whole_all, data_in_basic, data_out_basic, data_middle_basic), axis=0)

    # 去除重复冗余数据
    datas_combine_all = datas_combine_all.drop_duplicates()

    # 2022/4/13 add
    # 判断字段中有无if_haveCard，2022/5/24添加，针对partdata中有if_haveCard为1的情况
    if 'if_haveCard' in list(datas_combine_all.index.values):
        datas_combine_noCard = datas_combine_all[datas_combine_all['if_haveCard'] == 1]
        datas_combine_all = datas_combine_all[datas_combine_all['if_haveCard'] != 1]
    else:
        datas_combine_noCard = []
    data_abnormal_none = datas_combine_all[
        (datas_combine_all['出口门架时间'].isnull()) | (datas_combine_all['出口时间'].isnull())]
    datas_combine_all = datas_combine_all[
        (datas_combine_all['出口门架时间'].notnull()) & (datas_combine_all['出口时间'].notnull())]
    data_abnormal = datas_combine_all[datas_combine_all['出口门架时间'] != datas_combine_all['出口时间']]
    datas_combine_all = datas_combine_all[datas_combine_all['出口门架时间'] == datas_combine_all['出口时间']]
    out_time_list = data_abnormal['出口时间'].values
    end_gantry_time = data_abnormal['出口门架时间'].values
    time_list = data_abnormal['门架时间串'].values
    gantry_list = data_abnormal['门架路径'].values
    interval_string_list = data_abnormal['收费单元路径'].values
    length_list = data_abnormal['门架数'].values
    new_gantry_list = []
    new_interval_list = []
    new_time_list = []
    if_staycard = []
    length = []
    statistic_list = [['num_of_time_abnormal', 'num_of_passid_repeat', 'num_of_pass_province']]  # save all the statistic num
    statistic_num_list = []  # save data the statistic num
    num_of_time_abnormal = 0  # statistic the num of time wrong
    for i in range(len(out_time_list)):
        interval_disc = set(gantry_list[i].split('|'))
        if (datetime.datetime.strptime(end_gantry_time[i], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                out_time_list[i], "%Y-%m-%d %H:%M:%S")).total_seconds() > 1800:
            try:
                next_gantry = gantry_relation_disc[gantry_list[i].split('|')[-1]]
            except:
                next_gantry = ''
            if len(interval_disc.intersection(next_gantry)) == 0:
                time_list_single = time_list[i].split('|')
                gantry_list_single = gantry_list[i].split('|')
                interval_string_list_single = interval_string_list[i].split('|')
                new_gantry = gantry_list_single[
                             :(gantry_list_single.index(
                                 gantry_list_single[time_list_single.index(out_time_list[i])]) + 1)]
                new_interval = interval_string_list_single[:(interval_string_list_single.index(
                    gantry_list_single[time_list_single.index(out_time_list[i])]) + 1)]
                new_time = time_list_single[:(time_list_single.index(out_time_list[i]) + 1)]
                new_gantry_string = dbf.list_to_string_with_sign(new_gantry, '|')
                new_interval_string = dbf.list_to_string_with_sign(new_interval, '|')
                new_time_string = dbf.list_to_string_with_sign(new_time, '|')
                new_gantry_list.append(new_gantry_string)
                new_interval_list.append(new_interval_string)
                new_time_list.append(new_time_string)
                length.append(len(new_interval))
                if_staycard.append(1)
            else:
                new_gantry_list.append(gantry_list[i])
                new_interval_list.append(interval_string_list[i])
                new_time_list.append(time_list[i])
                length.append(length_list[i])
                if_staycard.append(0)
                num_of_time_abnormal += 1
        else:
            new_gantry_list.append(gantry_list[i])
            new_interval_list.append(interval_string_list[i])
            new_time_list.append(time_list[i])
            length.append(length_list[i])
            if_staycard.append(0)
            num_of_time_abnormal += 1
    data_abnormal['门架路径'] = new_gantry_list
    data_abnormal['收费单元路径'] = new_interval_list
    data_abnormal['门架时间串'] = new_time_list
    data_abnormal['门架数'] = length
    data_abnormal['出口门架时间'] = data_abnormal['出口时间']
    data_abnormal['if_haveCard'] = if_staycard
    datas_combine_all['if_haveCard'] = 0
    data_abnormal_none['if_haveCard'] = 0
    statistic_num_list.append(num_of_time_abnormal)
    if datas_combine_noCard:
        datas_combine_all = pd.concat((data_abnormal, datas_combine_all, data_abnormal_none, datas_combine_noCard), axis=0)
    else:
        datas_combine_all = pd.concat((data_abnormal, datas_combine_all, data_abnormal_none), axis=0)

    # 2022/4/13 add
    datas_combine_all = datas_combine_all.reset_index()
    data_abnormal = datas_combine_all[datas_combine_all.duplicated('PASSID', False)]
    datas_combine_all = datas_combine_all[datas_combine_all.duplicated('PASSID', False) == False]
    passid_list = list(set(list(data_abnormal['PASSID'].values)))
    data_abnormal_ls = []
    num_of_passid_repeat = 0  # statistic the num of passid repeat
    num_of_pass_province = 0  # statistic the num of pass_province
    for i in range(len(passid_list)):
        data_abnormal_1 = data_abnormal[(data_abnormal['PASSID'] == passid_list[i]) & (
                    (data_abnormal['middle_type'] == 'out') | (data_abnormal['middle_type'] == 'whole'))]
        data_abnormal_2 = data_abnormal[(data_abnormal['PASSID'] == passid_list[i]) & (
                    (data_abnormal['middle_type'] != 'out') & (data_abnormal['middle_type'] != 'whole'))]

        end_gantry_time_1 = data_abnormal_1['出口时间'].values
        end_gantry_time_2 = data_abnormal_2['出口门架时间'].values
        interval_string_list_1 = data_abnormal_1['收费单元路径'].values
        interval_string_list_2 = data_abnormal_2['收费单元路径'].values
        gantry_list_1 = data_abnormal_1['门架路径'].values
        gantry_list_2 = data_abnormal_2['门架路径'].values
        time_list_1 = data_abnormal_1['门架时间串'].values
        time_list_2 = data_abnormal_2['门架时间串'].values
        # if passid_list[i] == '011401193423920044901020211116213638' or passid_list[i] == '011401193423010002883420211118200339':
        #     print(1)
        if (data_abnormal_2.empty == False) and (data_abnormal_1.empty == True):
            data_abnormal_2 = data_abnormal_2.sort_values(['门架数'], ascending=False)
            data_abnormal_2 = data_abnormal_2.drop_duplicates(['PASSID'])
            if i == 0:
                data_abnormal_ls = data_abnormal_2
            else:
                data_abnormal_ls = pd.concat((data_abnormal_ls, data_abnormal_2), axis=0)
            num_of_passid_repeat += 1
        elif (data_abnormal_2.empty == False) and (data_abnormal_1.empty == False):
            #
            try:
                interval_disc = set(gantry_list_1[0].split('|'))
            except:
                interval_disc = ''
                print(1)
            try:
                next_gantry = gantry_relation_disc[gantry_list_2[0].split('|')[-1]]
            except:
                next_gantry = ''
            interval_key = interval_disc.intersection(next_gantry)
            if (datetime.datetime.strptime(end_gantry_time_2[0], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                    end_gantry_time_1[0], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                if len(interval_key) > 0:
                    money_1 = data_abnormal_1['门架费用'].values
                    money_2 = data_abnormal_2['门架费用'].values
                    length_1 = data_abnormal_1['总里程'].values
                    length_2 = data_abnormal_2['总里程'].values
                    interval_key = list(interval_key)
                    time_list_single = time_list_1[0].split('|')
                    gantry_list_single = gantry_list_1[0].split('|')
                    interval_string_list_single = interval_string_list_1[0].split('|')
                    new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                    new_gantry.extend(gantry_list_2)
                    new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
                    new_interval = interval_string_list_single[:interval_string_list_single.index(interval_key[0])]
                    new_interval.extend(interval_string_list_2)
                    new_interval.extend(
                        interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
                    new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                    new_time.extend(time_list_2)
                    new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                    data_abnormal_1['门架数'] = len(new_interval)
                    data_abnormal_1['门架费用'] = float(money_1[0]) + float(money_2[0])
                    data_abnormal_1['总里程'] = float(length_1[0]) + float(length_2[0])
                    data_abnormal_1['门架路径'] = dbf.list_to_string_with_sign(new_gantry, '|')
                    data_abnormal_1['收费单元路径'] = dbf.list_to_string_with_sign(new_interval, '|')
                    data_abnormal_1['门架时间串'] = dbf.list_to_string_with_sign(new_time, '|')
                    num_of_pass_province += 1
                else:
                    data_abnormal_1['if_haveCard'] = 1
            else:
                data_abnormal_1['if_haveCard'] = 1
            if i == 0:
                data_abnormal_ls = data_abnormal_1
            else:
                data_abnormal_ls = pd.concat((data_abnormal_ls, data_abnormal_1), axis=0)
        else:
            data_abnormal_1 = data_abnormal_1.drop_duplicates(['PASSID'])
            if i == 0:
                data_abnormal_ls = data_abnormal_1
            else:
                data_abnormal_ls = pd.concat((data_abnormal_ls, data_abnormal_1), axis=0)
    if len(data_abnormal_ls) == 0:
        print(1)
    else:
        datas_combine_all = pd.concat((data_abnormal_ls, datas_combine_all), axis=0).set_index(['PASSID'])
    # 2022/5/16 add
    statistic_num_list.append(num_of_passid_repeat)
    statistic_num_list.append(num_of_pass_province)
    statistic_list.append(statistic_num_list)
    statistic_data_path = kp.get_parameter_with_keyword('statistic_data_path')
    try:
        statistic_data = []
        with open(statistic_data_path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    statistic_data.append(row)
                else:
                    for j in range(len(row)):
                        sum_value = int(float(row[j])) + statistic_num_list[j]
                        row[j] = sum_value
                    statistic_data.append(row)
    except:
        statistic_data = statistic_list
        print('first time')
    finally:
        with open(statistic_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(statistic_data)

    return datas_combine_all


'''
    创建时间：2022/5/26
    完成时间：2022/5/26
    功能: 
    修改时间：
'''


def process_of_treat_part_data(date, days, data_type='manyDay'):
    """

    :param date:
    :param days:
    :return:
    """
    # 将该日期补全，用于后续进行时间类型转换
    now_time = date[:4] + '-' + date[4:6] + '-' + date[6:] + ' 00:00:00'
    # 将该日期转为时间类型
    now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
    # 获取需要获取的文件的时间的下限
    history_date = now_time + datetime.timedelta(days=(days * (-1)))
    history_date = history_date.strftime('%Y%m%d %H%M%S')[:8]

    history_date_second = now_time + datetime.timedelta(days=(days * (-1) + 1))
    history_date_second = history_date_second.strftime('%Y%m%d %H%M%S')[:8]

    if history_date == '20210831':
        return 1

    elif history_date_second == '20210831':
        history_date_second = now_time + datetime.timedelta(days=(days * (-1) + 2))
        history_date_second = history_date_second.strftime('%Y%m%d %H%M%S')[:8]

    if data_type == 'oneDay':
        part_data_path = kp.get_parameter_with_keyword('part_data_one_path')
    else:
        part_data_path = kp.get_parameter_with_keyword('part_data_many_path')

    # load data to key-value
    part_data = {}
    ifPlate = 0
    with open(part_data_path) as f:
        for k, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if k == 0:
                columns = row
                if '车牌(全)' not in row:
                    columns.append('车牌(全)')
                    ifPlate = 1
                part_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
            else:
                if ifPlate == 1:
                    if row[part_index[9]] == 'in':
                        row.append(row[part_index[15]])
                    elif row[part_index[9]] == 'out':
                        row.append(row[part_index[14]])
                    else:
                        row.append(row[part_index[15]])
                else:
                    if row[part_index[9]] == 'in':
                        row[-1] = row[part_index[15]]
                    elif row[part_index[9]] == 'out':
                        row[-1] = row[part_index[14]]
                    else:
                        row[-1] = row[part_index[15]]
                part_data[row[0]] = row

    print(len(part_data.keys()))

    # 获取合并后省内和省外的数据，2022/5/15添加，针对单日和多日分开
    if data_type == 'oneDay':  # 如果是单天的数据
        data_province_path = kp.get_parameter_with_keyword('data_province_one_path')
        data_province_back_path = kp.get_parameter_with_keyword('data_province_back_path')
        data_whole_path = kp.get_parameter_with_keyword('data_whole_one_path')
        data_whole_back_path = kp.get_parameter_with_keyword('data_whole_back_path')
    else:  # 如果是多天的数据
        data_province_path = kp.get_parameter_with_keyword('data_province_many_path')
        data_whole_path = kp.get_parameter_with_keyword('data_whole_many_path')

    # 进行收费单元的标准路径字典获取
    gantry_relation_disc = dbf.get_dict_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True)

    # drop repeat passid with the whole
    whole_key = []
    num = 0
    with open(data_whole_path + history_date + '.csv') as f:
        for k, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if k == 0:
                whole_columns = row
                whole_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
            if k > 0:
                try:
                    single_data = part_data[row[whole_index[0]]]
                    try:
                        interval_disc = set(row[whole_index[3]].split('|'))
                    except:
                        interval_disc = ''
                        print(1)
                    try:
                        next_gantry = gantry_relation_disc[single_data[part_index[3]].split('|')[-1]]
                    except:
                        next_gantry = ''
                    interval_key = interval_disc.intersection(next_gantry)
                    if (datetime.datetime.strptime(single_data[part_index[2]],
                                                   "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                        row[whole_index[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                        if len(interval_key) > 0:
                            money_1 = row[whole_index[6]]
                            money_2 = single_data[part_index[6]]
                            length_1 = row[whole_index[7]]
                            length_2 = single_data[part_index[7]]
                            interval_key = list(interval_key)
                            time_list_single = row[whole_index[5]].split('|')
                            gantry_list_single = row[whole_index[4]].split('|')
                            interval_string_list_single = row[whole_index[3]].split('|')
                            new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                            new_gantry.extend(single_data[part_index[4]].split('|'))
                            new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
                            new_interval = interval_string_list_single[
                                           :interval_string_list_single.index(interval_key[0])]
                            new_interval.extend(single_data[part_index[3]].split('|'))
                            new_interval.extend(
                                interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
                            new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                            new_time.extend(single_data[part_index[5]].split('|'))
                            new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                            row[whole_index[11]] = len(new_interval)
                            row[whole_index[6]] = float(money_1) + float(money_2)
                            row[whole_index[7]] = float(length_1) + float(length_2)
                            row[whole_index[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
                            row[whole_index[3]] = dbf.list_to_string_with_sign(new_interval, '|')
                            row[whole_index[5]] = dbf.list_to_string_with_sign(new_time, '|')
                            part_data[row[whole_index[0]]] = ''
                            whole_key.append(row)
                            num += 1
                        else:
                            part_data[row[whole_index[0]]] = ''
                            row[whole_index[8]] = 1
                            whole_key.append(row)
                            num += 1
                    else:
                        part_data[row[whole_index[0]]] = ''
                        row[whole_index[8]] = 1
                        whole_key.append(row)
                        num += 1
                except:
                    whole_key.append(row)
    print(num)
    # load the second day
    whole_key_second = []
    num = 0
    with open(data_whole_path + history_date_second + '.csv') as f:
        for k, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if k == 0:
                whole_columns_second = row
                whole_index_second = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
            if k > 0:
                try:
                    single_data = part_data[row[whole_index_second[0]]]
                    try:
                        interval_disc = set(row[whole_index_second[3]].split('|'))
                    except:
                        interval_disc = ''
                        print(1)
                    try:
                        next_gantry = gantry_relation_disc[single_data[part_index[3]].split('|')[-1]]
                    except:
                        next_gantry = ''
                    interval_key = interval_disc.intersection(next_gantry)
                    if (datetime.datetime.strptime(single_data[part_index[2]],
                                                   "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                        row[whole_index_second[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                        if len(interval_key) > 0:
                            money_1 = row[whole_index_second[6]]
                            money_2 = single_data[part_index[6]]
                            length_1 = row[whole_index_second[7]]
                            length_2 = single_data[part_index[7]]
                            interval_key = list(interval_key)
                            time_list_single = row[whole_index_second[5]].split('|')
                            gantry_list_single = row[whole_index_second[4]].split('|')
                            interval_string_list_single = row[whole_index_second[3]].split('|')
                            new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                            new_gantry.extend(single_data[part_index[4]].split('|'))
                            new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
                            new_interval = interval_string_list_single[
                                           :interval_string_list_single.index(interval_key[0])]
                            new_interval.extend(single_data[part_index[3]].split('|'))
                            new_interval.extend(
                                interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
                            new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                            new_time.extend(single_data[part_index[5]].split('|'))
                            new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                            row[whole_index_second[11]] = len(new_interval)
                            row[whole_index_second[6]] = float(money_1) + float(money_2)
                            row[whole_index_second[7]] = float(length_1) + float(length_2)
                            row[whole_index_second[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
                            row[whole_index_second[3]] = dbf.list_to_string_with_sign(new_interval, '|')
                            row[whole_index_second[5]] = dbf.list_to_string_with_sign(new_time, '|')
                            part_data[row[whole_index_second[0]]] = ''
                            whole_key_second.append(row)
                            num += 1
                        else:
                            part_data[row[whole_index_second[0]]] = ''
                            row[whole_index_second[8]] = 1
                            whole_key_second.append(row)
                            num += 1
                    else:
                        part_data[row[whole_index_second[0]]] = ''
                        row[whole_index_second[8]] = 1
                        whole_key_second.append(row)
                        num += 1
                except:
                    whole_key_second.append(row)

    print(num)
    data_w = pd.DataFrame(whole_key, columns=whole_columns)

    part_vehicle_data = {}
    num = 0
    for key in part_data.keys():
        if part_data[key] != '':
            try:
                part_vehicle_data[part_data[key][-1]].append(part_data[key])
                num += 1
            except:
                part_vehicle_data[part_data[key][-1]] = [part_data[key]]
                num += 1

    print(num)
    print(len(part_vehicle_data.keys()))

    # get the repeat vehicle data
    vehicle_repeat_part_data = [columns]
    vehicle_repeat_whole_data = [whole_columns]
    num = 0
    for i in range(len(whole_key)):
        if '默' in whole_key[i][whole_index[23]]:
            continue
        try:
            single_data = part_vehicle_data[whole_key[i][whole_index[23]]]
            for j in range(len(single_data)):
                start_time = datetime.datetime.strptime(whole_key[i][whole_index[25]],
                                                        "%Y-%m-%d %H:%M:%S")
                end_time = datetime.datetime.strptime(whole_key[i][whole_index[1]], "%Y-%m-%d %H:%M:%S")
                # 获取需要获取的文件的时间的下限
                start_time_o = start_time + datetime.timedelta(minutes=2)
                end_time_o = end_time + datetime.timedelta(minutes=-2)
                start_time_2 = start_time + datetime.timedelta(minutes=-2)
                end_time_2 = end_time + datetime.timedelta(minutes=2)
                start_time_o = start_time_o.strftime("%Y-%m-%d %H:%M:%S")
                end_time_o = end_time_o.strftime("%Y-%m-%d %H:%M:%S")
                start_time_2 = start_time_2.strftime("%Y-%m-%d %H:%M:%S")
                end_time_2 = end_time_2.strftime("%Y-%m-%d %H:%M:%S")
                if start_time_o < single_data[j][part_index[16]] < end_time_o or start_time_o < single_data[j][part_index[2]] < end_time_o:
                    vehicle_repeat_part_data.append(single_data[j])
                    vehicle_repeat_whole_data.append(whole_key[i])
                else:
                    pass
                if start_time_2 < single_data[j][part_index[16]] and single_data[j][part_index[2]] < end_time_2:
                    single_data.pop(j)
                    num += 1
                else:
                    continue
            part_vehicle_data[whole_key[i][whole_index[23]]] = single_data
        except:
            pass

    repeat_vehicle_part_data_path = kp.get_parameter_with_keyword('repeat_vehicle_part_data_path')
    repeat_vehicle_whole_data_path = kp.get_parameter_with_keyword('repeat_vehicle_whole_data_path')

    with open(repeat_vehicle_part_data_path + history_date + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vehicle_repeat_part_data)
    with open(repeat_vehicle_whole_data_path + history_date + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vehicle_repeat_whole_data)

    print(num)
    num = 0
    for i in range(len(whole_key_second)):
        if '默' in whole_key_second[i][whole_index_second[23]]:
            continue
        try:
            single_data = part_vehicle_data[whole_key_second[i][whole_index_second[23]]]
            for j in range(len(single_data)):
                start_time = datetime.datetime.strptime(whole_key_second[i][whole_index_second[25]], "%Y-%m-%d %H:%M:%S")
                end_time = datetime.datetime.strptime(whole_key_second[i][whole_index_second[1]], "%Y-%m-%d %H:%M:%S")

                # 获取需要获取的文件的时间的下限
                start_time_2 = start_time + datetime.timedelta(minutes=-2)
                end_time_2 = end_time + datetime.timedelta(minutes=2)
                start_time_1 = start_time + datetime.timedelta(minutes=2)
                end_time_1 = end_time + datetime.timedelta(minutes=-2)
                start_time_2 = start_time_2.strftime("%Y-%m-%d %H:%M:%S")
                end_time_2 = end_time_2.strftime("%Y-%m-%d %H:%M:%S")
                start_time_1 = start_time_1.strftime("%Y-%m-%d %H:%M:%S")
                end_time_1 = end_time_1.strftime("%Y-%m-%d %H:%M:%S")

                if (start_time_2 < single_data[j][part_index[16]] < start_time_1 and single_data[j][part_index[9]] != 'out') or (end_time_1 < single_data[j][part_index[2]] < end_time_2 and single_data[j][part_index[9]] == 'out'):
                    # if single_data[j][part_index[9]] == 'in':
                    #     print(1)
                    # a = single_data[j]
                    single_data.pop(j)
                    num += 1
                else:
                    continue
            part_vehicle_data[whole_key_second[i][whole_index_second[23]]] = single_data
        except:
            pass
    print(num)

    part_data = []
    for key in part_vehicle_data.keys():
        for j in range(len(part_vehicle_data[key])):
            part_data.append(part_vehicle_data[key][j])

    print(len(part_data))

    # drop the repeat data
    # for key in part_data.keys():
    #     if part_data[key][1] > 1:
    #         if part_data[key][0][0][part_index[2]] > part_data[key][0][1][part_index[2]]:
    #             data_first, data_second = part_data[key][0][1], part_data[key][0][0]
    #         elif part_data[key][0][0][part_index[2]] < part_data[key][0][1][part_index[2]]:
    #             data_first, data_second = part_data[key][0][0], part_data[key][0][1]
    #         else:
    #             if len(part_data[key][0][0][part_index[3]].split('|')) > len(part_data[key][0][1][part_index[3]].split('|')):
    #                 part_data[key][0] = [part_data[key][0][0]]
    #                 part_data[key][1] -= 1
    #             else:
    #                 part_data[key][0] = [part_data[key][0][1]]
    #                 part_data[key][1] -= 1
    #             continue
    #         if data_first[part_index[9]] == 'out' and data_second[part_index[9]] == 'none':
    #             # 进行收费单元的标准路径字典获取
    #             gantry_relation_disc = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
    #                                                               ['ENROADNODEID', 'EXROADNODEID'],
    #                                                               encoding='gbk', key_for_N=True)
    #             try:
    #                 interval_disc = set(data_first[part_index[3]].split('|'))
    #             except:
    #                 interval_disc = ''
    #                 print(1)
    #             try:
    #                 next_gantry = gantry_relation_disc[data_second[part_index[3]].split('|')[-1]]
    #             except:
    #                 next_gantry = ''
    #             interval_key = interval_disc.intersection(next_gantry)
    #             if (datetime.datetime.strptime(data_second[part_index[2]],
    #                                            "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
    #                 data_first[part_index[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
    #                 if len(interval_key) > 0:
    #                     money_1 = data_first[part_index[6]]
    #                     money_2 = data_second[part_index[6]]
    #                     length_1 = data_first[part_index[7]]
    #                     length_2 = data_second[part_index[7]]
    #                     interval_key = list(interval_key)
    #                     time_list_single = data_first[part_index[5]].split('|')
    #                     gantry_list_single = data_first[part_index[4]].split('|')
    #                     interval_string_list_single = data_first[part_index[3]].split('|')
    #                     new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
    #                     new_gantry.extend(data_second[part_index[4]].split('|'))
    #                     new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
    #                     new_interval = interval_string_list_single[
    #                                    :interval_string_list_single.index(interval_key[0])]
    #                     new_interval.extend(data_second[part_index[3]].split('|'))
    #                     new_interval.extend(
    #                         interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
    #                     new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
    #                     new_time.extend(data_second[part_index[5]].split('|'))
    #                     new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
    #                     data_first[part_index[10]] = len(new_interval)
    #                     data_first[part_index[6]] = float(money_1) + float(money_2)
    #                     data_first[part_index[7]] = float(length_1) + float(length_2)
    #                     data_first[part_index[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
    #                     data_first[part_index[3]] = dbf.list_to_string_with_sign(new_interval, '|')
    #                     data_first[part_index[5]] = dbf.list_to_string_with_sign(new_time, '|')
    #                     part_data[key][0] = [data_first]
    #                     part_data[key][1] -= 1
    #                 else:
    #                     # data_first[whole_index[8]] = 1
    #                     part_data[key][0] = [data_first]
    #                     part_data[key][1] -= 1
    #             else:
    #                 data_first[part_index[8]] = 1
    #                 part_data[key][0] = [data_first]
    #                 part_data[key][1] -= 1
    #         elif data_second[part_index[9]] == 'in' or data_first[part_index[9]] == 'out':
    #             part_data[key][0] = [data_first]
    #             part_data[key][1] -= 1
    #         else:
    #             data_second[part_index[3]] = data_first[part_index[3]] + '|' + data_second[part_index[3]]
    #             data_second[part_index[4]] = data_first[part_index[4]] + '|' + data_second[part_index[4]]
    #             data_second[part_index[5]] = data_first[part_index[5]] + '|' + data_second[part_index[5]]
    #             data_second[part_index[6]] = float(data_first[part_index[6]]) + float(
    #                 data_second[part_index[6]])
    #             data_second[part_index[7]] = float(data_first[part_index[7]]) + float(
    #                 data_second[part_index[7]])
    #             data_second[part_index[10]] = float(data_first[part_index[10]]) + float(
    #                 data_second[part_index[10]])
    #             data_second[part_index[11]] = data_first[part_index[11]] + '|' + data_second[part_index[11]]
    #             data_second[part_index[12]] = data_first[part_index[12]] + '|' + data_second[part_index[12]]
    #             if data_second[part_index[9]] == 'out' or data_first[part_index[9]] == 'in':
    #                 data_second[part_index[9]] = 'whole'
    #             elif data_second[part_index[9]] == 'none' or data_first[part_index[9]] == 'in':
    #                 data_second[part_index[9]] = 'in'
    #                 if data_second[part_index[13]] == '3':
    #                     data_second[part_index[13]] = '1'
    #             elif data_second[part_index[9]] == 'none' or data_first[part_index[9]] == 'none':
    #                 if (data_first[part_index[13]] == '2' or data_first[part_index[13]] == '1') and (data_second[part_index[13]] == '3' or data_second[part_index[13]] == '1'):
    #                     data_second[part_index[13]] = '1'
    #                 elif data_first[part_index[13]] == '3' and data_second[part_index[13]] == '1':
    #                     data_second[part_index[13]] = '3'
    #                 elif data_first[part_index[13]] == '3' and data_second[part_index[13]] == '2':
    #                     data_second[part_index[13]] = '0'
    #                 elif data_first[part_index[13]] == '0' and data_second[part_index[13]] == '2':
    #                     data_second[part_index[13]] = '0'
    #                 elif data_first[part_index[13]] == '0' and data_second[part_index[13]] == '1':
    #                     data_second[part_index[13]] = '3'
    #             part_data[key][0] = [data_second]
    #             part_data[key][1] -= 1




    # 获取收费站ID和收费站名称的对应字典
    station_name_disc = dbf.get_dict_from_document('../Data_Origin/station_OD.csv',
                                                   ['id', 'name'],
                                                   encoding='utf-8')
    interlist_inStation = dbf.get_dict_from_document('../Data_Origin/tollinterval.csv', ['id', 'enTollStation'],
                                                     encoding='utf-8')
    interlist_outStation = dbf.get_dict_from_document('../Data_Origin/tollinterval.csv',
                                                      ['id', 'exTollStation'],
                                                      encoding='utf-8')
    station_HEX = dbf.get_dict_from_document('../Data_Origin/station_HEX.xlsx', ['NEWSTATIONHEX', 'id'],
                                             encoding='utf-8')
    # get history_date data
    date_data = []
    part_data_list = [columns]
    for i in range(len(part_data)):
        if part_data[i][part_index[2]][:10].replace('-', '') == history_date:
            date_data.append(part_data[i])
        else:
            part_data_list.append(part_data[i])
    print(len(part_data))
    print(len(date_data))
    print(len(part_data_list))

    date_data = pd.DataFrame(date_data, columns=columns)
    date_data[['入口ID', '出口ID', '入口HEX码']] = date_data[['入口ID', '出口ID', '入口HEX码']].fillna('')
    print('开始进行数据缺失类型分类-------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 2022/2/25，变更了判断内容
    data_whole = date_data[(date_data['middle_type'] == 'whole') | (date_data['是否端口为省界'] == '1')]
    # 2022/2/25，变更了判断内容
    data_abnormal = date_data[((date_data['middle_type'] == 'in') | (date_data['middle_type'] == 'out') |
                                 (date_data['middle_type'] == 'none')) & (date_data['是否端口为省界'] != '1')]

    # 将出口缺失/入口缺失/出入口缺失的数据进行区分
    data_in = data_abnormal[(data_abnormal['middle_type'] == 'in')]
    data_in_none = data_abnormal[((data_abnormal['middle_type'] == 'none') & (data_abnormal['是否端口为省界'] == '2'))]
    data_out = data_abnormal[(data_abnormal['middle_type'] == 'out')]
    data_out_none = data_abnormal[((data_abnormal['middle_type'] == 'none') & (data_abnormal['是否端口为省界'] == '3'))]

    # 对入口缺失数据进行划分，划分为入省和完整
    data_out['in_pro_name'] = data_out['入口HEX码'].map(lambda x: x[:2] if len(x) > 2 else x)
    data_out['in_pro_name_2'] = data_out['入口ID'].map(lambda x: x[5:7] if len(x) > 8 else x)
    data_out_pro = data_out[(data_out['in_pro_name'] != '61') & (data_out['in_pro_name_2'] != '61')].drop(
        ['in_pro_name', 'in_pro_name_2'], axis=1)
    data_out_whole = data_out[(data_out['in_pro_name'] == '61') & (data_out['in_pro_name_2'] == '61')].drop(
        ['in_pro_name', 'in_pro_name_2'], axis=1)
    data_out_whole_hex = data_out[(data_out['in_pro_name'] == '61') & (data_out['in_pro_name_2'] != '61')].drop(
        ['in_pro_name', 'in_pro_name_2'], axis=1)
    data_out_whole_id = data_out[(data_out['in_pro_name'] != '61') & (data_out['in_pro_name_2'] == '61')].drop(
        ['in_pro_name', 'in_pro_name_2'], axis=1)

    # 对无入无出但出省的数据进行划分，划分为出省和跨省
    data_out_none['in_pro_name'] = data_out_none['入口HEX码'].map(lambda x: x[:2] if len(x) > 2 else x)
    data_none_outpro = data_out_none[
        (data_out_none['in_pro_name'] == '61') | (data_out_none['in_pro_name'] == '')].drop(['in_pro_name'], axis=1)
    data_none_pass = data_out_none[
        (data_out_none['in_pro_name'] != '61') & (data_out_none['in_pro_name'] != '')].drop(['in_pro_name'], axis=1)

    # 获取无入无出也未经过省界的数据，定位无出有入数据
    data_noout = data_abnormal[(data_abnormal['middle_type'] == 'none') & (data_abnormal['是否端口为省界'] == '0')]
    data_noout['in_pro_name'] = data_noout['入口HEX码'].map(lambda x: x[:2] if len(x) > 2 else x)
    data_noout_inpro = data_noout[
        (data_noout['in_pro_name'] == '61')].drop(['in_pro_name'], axis=1)
    data_noout_oupro = data_noout[
        (data_noout['in_pro_name'] != '61')].drop(['in_pro_name'], axis=1)

    # 2022/5/13 add
    # get the no stationID data
    data_in_noStation = data_in[((data_in['入口ID'] == '')) & ((data_in['入口HEX码'] != ''))]
    data_in_Station = data_in[((data_in['入口ID'] != '')) | ((data_in['入口HEX码'] == ''))]
    data_in_noStation['入口ID'] = data_in_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_in = pd.concat((data_in_Station, data_in_noStation), axis=0)

    # 给有入无出的数据进行出入口ID的赋值
    data_in['车型'] = data_in['入口车型']
    data_in['本省入口ID'] = data_in['入口ID']
    data_in['本省入口站id'] = data_in['入口ID']
    data_in['入口收费站名称'] = data_in['入口ID'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_in['本省入口时间'] = data_in['入口时间']
    data_in['本省出口ID'] = ''
    data_in['本省出口站id'] = ''
    data_in['出口收费站名称'] = ''
    data_in['本省出口时间'] = ''

    data_in_none['车型'] = data_in_none['入口车型']
    data_in_none['本省入口ID'] = data_in_none['收费单元路径'].map(lambda x: x.split('|', 1)[0])
    data_in_none['本省入口站id'] = data_in_none['本省入口ID'].map(lambda x: interlist_inStation[x])
    data_in_none['入口收费站名称'] = data_in_none['本省入口站id'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_in_none['本省入口时间'] = data_in_none['入口门架时间']
    data_in_none['本省出口ID'] = ''
    data_in_none['本省出口站id'] = ''
    data_in_none['出口收费站名称'] = ''
    data_in_none['本省出口时间'] = ''

    data_noout_inpro_noStation = data_noout_inpro[((data_noout_inpro['入口ID'] == '')) & (
        (data_noout_inpro['入口HEX码'] != ''))]
    data_noout_inpro_Station = data_noout_inpro[((data_noout_inpro['入口ID'] != '')) | (
        (data_noout_inpro['入口HEX码'] == ''))]
    data_noout_inpro_noStation['入口ID'] = data_noout_inpro_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_noout_inpro = pd.concat((data_noout_inpro_Station, data_noout_inpro_noStation), axis=0)

    data_noout_inpro['车型'] = data_noout_inpro['入口车型']
    data_noout_inpro['本省入口ID'] = data_noout_inpro['入口ID']
    data_noout_inpro['本省入口站id'] = data_noout_inpro['入口ID']
    data_noout_inpro['入口收费站名称'] = data_noout_inpro['入口ID'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_noout_inpro['本省入口时间'] = data_noout_inpro['入口时间']
    data_noout_inpro['本省出口ID'] = ''
    data_noout_inpro['本省出口站id'] = ''
    data_noout_inpro['出口收费站名称'] = ''
    data_noout_inpro['本省出口时间'] = ''
    data_noout_inpro['middle_type'] = 'in'

    data_noout_oupro['车型'] = data_noout_oupro['入口车型']
    data_noout_oupro['本省入口ID'] = ''
    data_noout_oupro['本省入口站id'] = ''
    data_noout_oupro['入口收费站名称'] = ''
    data_noout_oupro['本省入口时间'] = ''
    data_noout_oupro['本省出口ID'] = ''
    data_noout_oupro['本省出口站id'] = ''
    data_noout_oupro['出口收费站名称'] = ''
    data_noout_oupro['本省出口时间'] = ''

    data_noout = pd.concat((data_noout_inpro, data_noout_oupro), axis=0)

    data_in['类型'] = 'only_in'
    data_in_none['类型'] = 'only_in'
    data_noout['类型'] = 'only_in'
    data_abnormal_out = pd.concat((data_in, data_in_none, data_noout), axis=0)

    # 保存缺失数据的可视化数据，2022/2/25将该部分内容调整了位置
    print('开始保存异常缺失数据-------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 2022/5/15添加，将单日和多日的缺失数据存储位置分开
    if data_type == 'oneDay':
        save_name = kp.get_parameter_with_keyword('loss_data_one_path') + history_date + '.csv'
        save_back_path = kp.get_parameter_with_keyword('loss_data_one_back_path') + history_date + '.csv'
        data_abnormal_out.to_csv(save_back_path, index=False)
    else:
        save_name = kp.get_parameter_with_keyword('loss_data_many_path') + history_date + '.csv'
    data_abnormal_out.to_csv(save_name, index=False)

    # 没有将出入省的记录进行最短路径匹配,因为出入省的没有一端的收费站ID,无法匹配
    # 2022/3/18 add, let the province to in or out or pass
    data_whole_border_none = data_whole[
        (data_whole['是否端口为省界'] == '1') & (data_whole['middle_type'] == 'none')]  # 获取跨省的完整数据
    data_whole_border_none['类型'] = 'province_pass'
    data_whole_border_none['本省入口ID'] = data_whole_border_none['收费单元路径'].map(lambda x: x.split('|', 1)[0])
    data_whole_border_none['本省入口站id'] = data_whole_border_none['本省入口ID'].map(lambda x: interlist_inStation[x])
    data_whole_border_none['入口收费站名称'] = data_whole_border_none['本省入口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_none['本省入口时间'] = data_whole_border_none['入口门架时间']
    data_whole_border_none['本省出口ID'] = data_whole_border_none['收费单元路径'].map(lambda x: x.rsplit('|', 1)[-1])
    data_whole_border_none['本省出口站id'] = data_whole_border_none['本省出口ID'].map(lambda x: interlist_outStation[x])
    data_whole_border_none['出口收费站名称'] = data_whole_border_none['本省出口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_none['本省出口时间'] = data_whole_border_none['出口门架时间']

    data_none_pass['类型'] = 'province_pass'
    data_none_pass['本省入口ID'] = ''
    data_none_pass['本省入口站id'] = ''
    data_none_pass['入口收费站名称'] = ''
    data_none_pass['本省入口时间'] = ''
    data_none_pass['本省出口ID'] = data_none_pass['收费单元路径'].map(lambda x: x.rsplit('|', 1)[-1])
    data_none_pass['本省出口站id'] = data_none_pass['本省出口ID'].map(lambda x: interlist_outStation[x])
    data_none_pass['出口收费站名称'] = data_none_pass['本省出口站id'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_none_pass['本省出口时间'] = data_none_pass['出口门架时间']

    data_whole_border_none = pd.concat((data_whole_border_none, data_none_pass), axis=0)

    #
    data_whole_border_out = data_whole[
        (data_whole['是否端口为省界'] == '1') & (data_whole['middle_type'] == 'in')]  # 获取out省的完整数据

    # 2022/5/13 add
    # get the no stationID data
    data_whole_border_out_noStation = data_whole_border_out[
        ((data_whole_border_out['入口ID'] == '')) & ((data_whole_border_out['入口HEX码'] != ''))]
    data_whole_border_out_Station = data_whole_border_out[
        ((data_whole_border_out['入口ID'] != '')) | ((data_whole_border_out['入口HEX码'] == ''))]
    data_whole_border_out_noStation['入口ID'] = data_whole_border_out_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_whole_border_out = pd.concat((data_whole_border_out_Station, data_whole_border_out_noStation), axis=0)

    data_whole_border_out['类型'] = 'province_out'
    data_whole_border_out['本省入口ID'] = data_whole_border_out['入口ID']
    data_whole_border_out['本省入口站id'] = data_whole_border_out['入口ID']
    data_whole_border_out['入口收费站名称'] = data_whole_border_out['入口ID'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_out['本省入口时间'] = data_whole_border_out['入口时间']
    data_whole_border_out['本省出口ID'] = data_whole_border_out['收费单元路径'].map(lambda x: x.rsplit('|', 1)[-1])
    data_whole_border_out['本省出口站id'] = data_whole_border_out['本省出口ID'].map(lambda x: interlist_outStation[x])
    data_whole_border_out['出口收费站名称'] = data_whole_border_out['本省出口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_out['本省出口时间'] = data_whole_border_out['出口门架时间']

    # 2022/5/13 add
    # get the no stationID data
    # a = data_none_outpro[data_none_outpro['PASSID']=='020000610201630060403920210526082122']
    # if len(list(a['PASSID'].values)):
    #     print(1)
    data_none_outpro_noStation = data_none_outpro[
        ((data_none_outpro['入口ID'] == '')) & ((data_none_outpro['入口HEX码'] != ''))]
    data_none_outpro_Station = data_none_outpro[
        ((data_none_outpro['入口ID'] != '')) | ((data_none_outpro['入口HEX码'] == ''))]
    data_none_outpro_noStation['入口ID'] = data_none_outpro_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_none_outpro = pd.concat((data_none_outpro_Station, data_none_outpro_noStation), axis=0)

    data_none_outpro['类型'] = 'province_out'
    data_none_outpro['本省入口ID'] = data_none_outpro['入口ID']
    data_none_outpro['本省入口站id'] = data_none_outpro['入口ID']
    data_none_outpro['入口收费站名称'] = data_none_outpro['入口ID'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_none_outpro['本省入口时间'] = data_none_outpro['入口时间']
    data_none_outpro['本省出口ID'] = data_none_outpro['收费单元路径'].map(lambda x: x.rsplit('|', 1)[-1])
    data_none_outpro['本省出口站id'] = data_none_outpro['本省出口ID'].map(lambda x: interlist_outStation[x])
    data_none_outpro['出口收费站名称'] = data_none_outpro['本省出口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_none_outpro['本省出口时间'] = data_none_outpro['出口门架时间']

    data_whole_border_out = pd.concat((data_whole_border_out, data_none_outpro), axis=0)

    #
    data_whole_border_in = data_whole[(data_whole['是否端口为省界'] == '1') & (data_whole['middle_type'] == 'out')]  # 获取in省的完整数据

    # 2022/4/10,添加各出入口ID
    data_whole_border_in['类型'] = 'province_in'
    data_whole_border_in['本省入口ID'] = data_whole_border_in['收费单元路径'].map(lambda x: x.split('|', 1)[0])
    data_whole_border_in['本省入口站id'] = data_whole_border_in['本省入口ID'].map(lambda x: interlist_inStation[x])
    data_whole_border_in['入口收费站名称'] = data_whole_border_in['本省入口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_in['本省入口时间'] = data_whole_border_in['入口门架时间']
    data_whole_border_in['本省出口ID'] = data_whole_border_in['出口ID']
    data_whole_border_in['本省出口站id'] = data_whole_border_in['出口ID']
    data_whole_border_in['出口收费站名称'] = data_whole_border_in['本省出口站id'].map(
        lambda x: station_name_disc[x] if x != '' else x)
    data_whole_border_in['本省出口时间'] = data_whole_border_in['出口时间']

    data_out_pro['类型'] = 'province_in'
    data_out_pro['本省入口ID'] = ''
    data_out_pro['本省入口站id'] = ''
    data_out_pro['入口收费站名称'] = ''
    data_out_pro['本省入口时间'] = data_out_pro['入口门架时间']
    data_out_pro['本省出口ID'] = data_out_pro['出口ID']
    data_out_pro['本省出口站id'] = data_out_pro['出口ID']
    data_out_pro['出口收费站名称'] = data_out_pro['出口ID'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_out_pro['本省出口时间'] = data_out_pro['出口时间']

    data_whole_border_in = pd.concat((data_whole_border_in, data_out_pro), axis=0)

    # 将所有跨省的数据进行合并，并进行保存
    data_whole_border = pd.concat((data_whole_border_none, data_whole_border_out, data_whole_border_in), axis=0)

    data_whole = data_whole[data_whole['是否端口为省界'] != '1']  # 获取省内的完整行驶数据

    # 2022/5/13 add
    # get the no stationID data
    data_whole_noStation = data_whole[((data_whole['入口ID'] == '')) & ((data_whole['入口HEX码'] != ''))]
    data_whole_Station = data_whole[((data_whole['入口ID'] != '')) | ((data_whole['入口HEX码'] == ''))]
    data_whole_noStation['入口ID'] = data_whole_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_whole = pd.concat((data_whole_Station, data_whole_noStation), axis=0)

    data_whole['类型'] = 'whole'
    data_whole['本省入口ID'] = data_whole['入口ID']
    data_whole['本省入口站id'] = data_whole['入口ID']
    data_whole['入口收费站名称'] = data_whole['入口ID'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_whole['本省入口时间'] = data_whole['入口时间']
    data_whole['本省出口ID'] = data_whole['出口ID']
    data_whole['本省出口站id'] = data_whole['出口ID']
    data_whole['出口收费站名称'] = data_whole['本省出口站id'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_whole['本省出口时间'] = data_whole['出口时间']

    # 2022/5/13 add
    # get the no stationID data
    # data_out_whole_noStation = data_out_whole[((data_out_whole['入口ID'] == '')) & ((data_out_whole['入口HEX码'] != ''))]
    # data_out_whole_Station = data_out_whole[((data_out_whole['入口ID'] != '')) | ((data_out_whole['入口HEX码'] == ''))]
    # data_out_whole_noStation['入口ID'] = data_out_whole_noStation['入口HEX码'].map(lambda x: station_HEX[x])
    data_out_whole_hex['入口ID'] = data_out_whole_hex['入口HEX码'].map(lambda x: station_HEX[x])
    data_out_whole = pd.concat((data_out_whole, data_out_whole_hex, data_out_whole_id), axis=0)

    data_out_whole['类型'] = 'whole'
    data_out_whole['本省入口ID'] = data_out_whole['入口ID']
    data_out_whole['本省入口站id'] = data_out_whole['入口ID']
    data_out_whole['入口收费站名称'] = data_out_whole['入口ID'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_out_whole['本省入口时间'] = data_out_whole['入口时间']
    data_out_whole['本省出口ID'] = data_out_whole['出口ID']
    data_out_whole['本省出口站id'] = data_out_whole['出口ID']
    data_out_whole['出口收费站名称'] = data_out_whole['本省出口站id'].map(lambda x: station_name_disc[x] if x != '' else x)
    data_out_whole['本省出口时间'] = data_out_whole['出口时间']

    data_whole = pd.concat((data_whole, data_out_whole), axis=0)

    data_w_second = pd.DataFrame(whole_key_second, columns=whole_columns_second)
    data_w_second.to_csv(data_whole_path + history_date_second + '.csv', index=False)

    # 获取合并后省内和省外的数据，2022/5/15添加，针对单日和多日分开
    if data_type == 'oneDay':  # 如果是单天的数据
        data_province_path = kp.get_parameter_with_keyword('data_province_one_path')
        data_province_back_path = kp.get_parameter_with_keyword('data_province_back_path')
        data_whole_path = kp.get_parameter_with_keyword('data_whole_one_path')
        data_whole_back_path = kp.get_parameter_with_keyword('data_whole_back_path')
        # 先进行备份数据的保存
        data_whole_border.to_csv(data_province_back_path + date + '.csv')
        data_whole.to_csv(data_whole_back_path + date + '.csv')
    try:
        data_whole = pd.concat((data_w, data_whole), axis=0)
    except:
        print(1)

    # save new part_data
    with open(part_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(part_data_list)

    # 分别保存省内和跨省的行驶数据
    data_whole_border.to_csv(data_province_path + history_date + '.csv', index=False)
    data_whole.to_csv(data_whole_path + history_date + '.csv', index=False)



'''
    创建时间：2022/5/23
    完成时间：2022/5/24
    功能: 对合并后的中间数据与历史数据进行PASSID对比去重
    修改时间：
'''


def process_of_filter_middle_passid(history_days, treat_type='manyDay'):
    """
    对合并后的中间数据与历史数据进行PASSID对比去重
    :param history_days:
    :param treat_type: 数据处理的类型，分长期时间数据处理和单天数据处理
    :return:
    """
    if treat_type == 'oneDay':
        data_province_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_province_one_path'))
        data_whole_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_whole_one_path'))
        loss_data_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('loss_data_path'))

    else:
        data_province_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_province_many_path'), True)
        data_whole_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_whole_many_path'), True)
        loss_data_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('loss_data_many_path'), True)
    # 进行收费单元的标准路径字典获取
    gantry_relation_disc = dbf.get_dict_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True)
    # 获取part data的地址
    part_data_path = kp.get_parameter_with_keyword('part_data_many_path')
    # 用于存放近几天的缺失数据处理后的key-value内容
    loss_keys = {}
    # 用于存放近几天的省内完整数据处理后的key-value内容
    whole_keys = {}
    # 用于存放近几天的跨省数据处理后的key-value内容
    pro_keys = {}
    for i in range(len(data_whole_path)):
        # 获取该文件的日期
        now_date = data_whole_path[i][-12:-4]
        # 将该日期补全，用于后续进行时间类型转换
        now_time = now_date[:4] + '-' + now_date[4:6] + '-' + now_date[6:] + ' 00:00:00'
        # 将该日期转为时间类型
        now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d %H:%M:%S")
        # 获取需要获取的文件的时间的下限
        history_date = now_time + datetime.timedelta(days=(history_days * (-1)))
        # 获取在前一天的时间
        history_last_date = now_time + datetime.timedelta(days=(history_days * (-1) -1))
        # 获取最近一天时间
        lastly_date = now_time + datetime.timedelta(days=-1)

        history_last_date = history_last_date.strftime('%Y%m%d %H%M%S')
        lastly_date = lastly_date.strftime('%Y%m%d %H%M%S')
        history_date = history_date.strftime('%Y%m%d %H%M%S')
        if i < 1:
            for j in range(len(data_whole_path)):
                # 判断该文件的日期是否在近history_days之内的，如果在时间内，进行处理，不在直接跳过
                if loss_data_path[j][-12:-4] >= history_date[:8] and loss_data_path[j][-12:-4] < now_date:
                    # 用于存放缺失数据处理后的key-value内容
                    loss_key = {}
                    # 用于存放省内完整数据处理后的key-value内容
                    whole_key = {}
                    # 用于存放跨省数据处理后的key-value内容
                    pro_key = {}
                    with open(loss_data_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                loss_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    with open(data_province_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                pro_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    with open(data_whole_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                whole_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    # 将key-value数据和其长度均进行保存，用于后续对比是否key-value有变化
                    loss_keys[loss_data_path[j][-12:-4]] = [loss_key, len(loss_key.keys())]
                    whole_keys[loss_data_path[j][-12:-4]] = [whole_key, len(whole_key.keys())]
                    pro_keys[loss_data_path[j][-12:-4]] = [pro_key, len(pro_key.keys())]

        else:
            if len(loss_keys) >= history_days:
                # 如果此时的key-value数据超过了规定的days数量，则将最早的一天删除，删除前先进行判断如果有变动，就先进行保存
                # if len(loss_keys[history_last_date[:8]][0]) != loss_keys[history_last_date[:8]][1]:
                #     dbf.basic_save_dict_data(loss_keys[history_last_date[:8]][0],
                #                              loss_data_path.rsplit('/', 1)[0] + '/' + history_last_date[:8] + '.csv')
                #
                # if len(whole_keys[history_last_date[:8]][0]) != whole_keys[history_last_date[:8]][1]:
                #     dbf.basic_save_dict_data(whole_keys[history_last_date[:8]][0],
                #                              data_whole_path.rsplit('/', 1)[0] + '/' + history_last_date[:8] + '.csv')
                #
                # if len(pro_keys[history_last_date[:8]][0]) != pro_keys[history_last_date[:8]][1]:
                #     dbf.basic_save_dict_data(pro_keys[history_last_date[:8]][0],
                #                              data_province_path.rsplit('/', 1)[0] + '/' + history_last_date[:8] + '.csv')

                loss_keys.pop(history_last_date[:8])
                whole_keys.pop(history_last_date[:8])
                pro_keys.pop(history_last_date[:8])

            for j in range(len(data_whole_path)):
                # 判断该文件的日期是否在近history_days之内的，如果在时间内，进行处理，不在直接跳过
                if loss_data_path[j][-12:-4] == lastly_date[:8]:
                    # 用于存放缺失数据处理后的key-value内容
                    loss_key = {}
                    # 用于存放省内完整数据处理后的key-value内容
                    whole_key = {}
                    # 用于存放跨省数据处理后的key-value内容
                    pro_key = {}
                    with open(loss_data_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                loss_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    with open(data_province_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                pro_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    with open(data_whole_path[j]) as f:
                        for k, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if k > 0:
                                whole_key[row[0]] = row
                            # if k > 1000:
                            #     break
                    loss_keys[loss_data_path[j][-12:-4]] = [loss_key, len(loss_key.keys())]
                    whole_keys[loss_data_path[j][-12:-4]] = [whole_key, len(whole_key.keys())]
                    pro_keys[loss_data_path[j][-12:-4]] = [pro_key, len(pro_key.keys())]
        if len(whole_keys.keys()) == 0:
            continue
        # 用于存放当天缺失数据处理后的key-value内容
        loss_key_now = {}
        # 用于存放当天省内完整数据处理后的key-value内容
        whole_key_now = {}
        # 用于存放当天跨省数据处理后的key-value内容
        pro_key_now = {}
        # 用于存放part data处理后的key-value内容
        part_key_now = {}
        # 将当天的数据进行处理，转换成key-value形式，用于后续匹配
        with open(loss_data_path[i]) as f:
            for k, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if k == 0:
                    loss_columns = row
                    # 获取缺失数据中，后续所需字段的下标
                    loss_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
                else:
                    loss_key_now[row[0]] = row
                # if k > 1000:
                #     break
        with open(data_province_path[i]) as f:
            for k, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if k == 0:
                    pro_columns = row
                    # 获取跨省数据中，后续所需字段的下标
                    pro_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
                else:
                    pro_key_now[row[0]] = row
                # if k > 1000:
                #     break
        with open(data_whole_path[i]) as f:
            for k, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if k == 0:
                    whole_columns = row
                    # 获取省内完整数据中，后续所需字段的下标
                    whole_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
                else:
                    whole_key_now[row[0]] = row
                # if k > 1000:
                #     break
        with open(part_data_path) as f:
            for k, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if k == 0:
                    part_columns = row
                    # 获取跨省数据中，后续所需字段的下标
                    part_index = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('drop_repeat_passid'))
                else:
                    part_key_now[row[0]] = row
                # if k > 1000:
                #     break

        # 先匹配当天的loss数据
        loss_key_now_drop = []
        pro_key_now_drop = []
        whole_key_now_drop = []
        part_key_now_drop = []
        for key in loss_key_now.keys():
            for k in loss_keys.keys():
                try:
                    repeat_value = loss_keys[k][0][key]
                    repeat_value_now = loss_key_now[key]
                    # 进行排序判断处理，判断两条数据的出口门架时间谁先谁后
                    if repeat_value[loss_index[2]] > repeat_value_now[loss_index[2]]:
                        data_first = repeat_value_now
                        data_second = repeat_value
                    else:
                        data_first = repeat_value
                        data_second = repeat_value_now
                    # 判断两个数据的类型
                    if data_second[loss_index[9]] == 'in':
                        # 如果都是only in，就只保留出口门架时间靠后的
                        loss_keys[k][0].pop(key)
                        if data_second[loss_index[3]] != data_first[loss_index[3]]:
                            data_second[loss_index[3]] = data_first[loss_index[3]] + '|' + data_second[loss_index[3]]
                            data_second[loss_index[4]] = data_first[loss_index[4]] + '|' + data_second[loss_index[4]]
                            data_second[loss_index[5]] = data_first[loss_index[5]] + '|' + data_second[loss_index[5]]
                            data_second[loss_index[6]] = float(data_first[loss_index[6]]) + float(
                                data_second[loss_index[6]])
                            data_second[loss_index[7]] = float(data_first[loss_index[7]]) + float(
                                data_second[loss_index[7]])
                            data_second[loss_index[11]] = float(data_first[loss_index[11]]) + float(
                                data_second[loss_index[11]])
                            data_second[loss_index[12]] = data_first[loss_index[12]] + '|' + data_second[loss_index[12]]
                            data_second[loss_index[13]] = data_first[loss_index[13]] + '|' + data_second[loss_index[13]]
                            data_second[loss_index[14]] = data_first[loss_index[14]]
                            data_second[loss_index[15]] = data_first[loss_index[15]]
                            data_second[loss_index[16]] = data_first[loss_index[16]]
                            data_second[loss_index[17]] = data_first[loss_index[17]]
                            data_second[loss_index[8]] = 2
                        loss_key_now[key] = data_second
                    elif data_second[loss_index[9]] == 'none':
                        # 如果先的数据为onlyin，后的为none，则进行合并
                        data_second[loss_index[3]] = data_first[loss_index[3]] + '|' + data_second[loss_index[3]]
                        data_second[loss_index[4]] = data_first[loss_index[4]] + '|' + data_second[loss_index[4]]
                        data_second[loss_index[5]] = data_first[loss_index[5]] + '|' + data_second[loss_index[5]]
                        data_second[loss_index[6]] = float(data_first[loss_index[6]]) + float(data_second[loss_index[6]])
                        data_second[loss_index[7]] = float(data_first[loss_index[7]]) + float(
                            data_second[loss_index[7]])
                        data_second[loss_index[11]] = float(data_first[loss_index[11]]) + float(
                            data_second[loss_index[11]])
                        data_second[loss_index[12]] = data_first[loss_index[12]] + '|' + data_second[loss_index[12]]
                        data_second[loss_index[13]] = data_first[loss_index[13]] + '|' + data_second[loss_index[13]]
                        data_second[loss_index[14]] = data_first[loss_index[14]]
                        data_second[loss_index[15]] = data_first[loss_index[15]]
                        data_second[loss_index[16]] = data_first[loss_index[16]]
                        data_second[loss_index[17]] = data_first[loss_index[17]]
                        loss_keys[k][0].pop(key)
                        data_second[loss_index[8]] = 2
                        loss_key_now[key] = data_second

                except:
                    continue
            for k in whole_keys.keys():
                try:
                    repeat_value = whole_keys[k][0][key]
                    repeat_value_now = loss_key_now[key]
                    # 如果缺失能匹配上完整数据，直接删除缺失数据，修改完整数据的if_haveCard字段
                    loss_key_now_drop.append(key)
                    repeat_value[whole_index[8]] = 1
                    whole_keys[k][0][key] = repeat_value
                    whole_keys[k][1] += 1
                except:
                    continue
            for k in pro_keys.keys():
                try:
                    repeat_value = pro_keys[k][0][key]
                    repeat_value_now = loss_key_now[key]
                    # 如果缺失能匹配上跨省数据，直接删除缺失数据，修改完整数据的if_haveCard字段
                    loss_key_now_drop.append(key)
                    repeat_value[pro_index[8]] = 1
                    pro_keys[k][0][key] = repeat_value
                    pro_keys[k][1] += 1
                except:
                    continue

        # drop the repeat key
        loss_key_now_save = [loss_columns]
        for key in loss_key_now.keys():
            if key not in loss_key_now_drop:
                loss_key_now_save.append(loss_key_now[key])

        # 先匹配当天的跨省数据
        for key in pro_key_now.keys():
            for k in loss_keys.keys():
                try:
                    repeat_value = loss_keys[k][0][key]
                    repeat_value_now = pro_key_now[key]
                    if repeat_value_now[pro_index[2]] > repeat_value[loss_index[2]]:
                        # 如果缺失能匹配上跨省数据，直接删除缺失数据，修改完整数据的if_haveCard字段
                        repeat_value_now[pro_index[3]] = repeat_value[loss_index[3]] + '|' + repeat_value_now[pro_index[3]]
                        repeat_value_now[pro_index[4]] = repeat_value[loss_index[4]] + '|' + repeat_value_now[pro_index[4]]
                        repeat_value_now[pro_index[5]] = repeat_value[loss_index[5]] + '|' + repeat_value_now[pro_index[5]]
                        repeat_value_now[pro_index[6]] = float(repeat_value[loss_index[6]]) + float(repeat_value_now[pro_index[6]])
                        repeat_value_now[pro_index[7]] = float(repeat_value[loss_index[7]]) + float(
                            repeat_value_now[pro_index[7]])
                        repeat_value_now[pro_index[11]] = float(repeat_value[loss_index[11]]) + float(
                            repeat_value_now[pro_index[11]])
                        repeat_value_now[pro_index[12]] = repeat_value[loss_index[12]] + '|' + repeat_value_now[pro_index[12]]
                        repeat_value_now[pro_index[13]] = repeat_value[loss_index[13]] + '|' + repeat_value_now[pro_index[13]]
                        if repeat_value_now[pro_index[10]] == 'province_out':
                            loss_keys[k][0].pop(key)
                            # repeat_value_now[pro_index[8]] = 1
                            pro_key_now[key] = repeat_value_now
                        elif repeat_value[loss_index[14]] != '':
                            repeat_value_now[pro_index[14]] = repeat_value[loss_index[14]]
                            repeat_value_now[pro_index[15]] = repeat_value[loss_index[15]]
                            repeat_value_now[pro_index[16]] = repeat_value[loss_index[16]]
                            repeat_value_now[pro_index[17]] = repeat_value[loss_index[17]]

                            if repeat_value_now[pro_index[10]] == 'province_in' and repeat_value[loss_index[9]] == 'in':
                                repeat_value_now[pro_index[10]] = 'whole'
                                loss_keys[k][0].pop(key)
                                pro_key_now_drop.append(key)
                                repeat_value_now[pro_index[8]] = 1
                                whole_keys[k][0][key] = repeat_value_now
                            elif repeat_value[loss_index[9]] == 'none':
                                loss_keys[k][0].pop(key)
                                repeat_value_now[pro_index[8]] = 1
                                pro_key_now[key] = repeat_value_now
                            else:
                                repeat_value_now[pro_index[10]] = 'province_out'
                                loss_keys[k][0].pop(key)
                                # repeat_value_now[pro_index[8]] = 1
                                pro_key_now[key] = repeat_value_now
                        else:
                            loss_keys[k][0].pop(key)
                            # repeat_value_now[pro_index[8]] = 1
                            pro_key_now[key] = repeat_value_now
                    else:
                        loss_keys[k][0].pop(key)
                        repeat_value_now[pro_index[8]] = 1
                        pro_key_now[key] = repeat_value_now
                except:
                    continue
            for k in whole_keys.keys():
                try:
                    repeat_value = whole_keys[k][0][key]
                    repeat_value_now = pro_key_now[key]
                    # 如果跨省的能匹配上省内完整数据，直接当天跨省数据，并修改完整数据的if_haveCard字段
                    pro_key_now_drop.append(key)
                    repeat_value[whole_index[8]] = 1
                    whole_keys[k][0][key] = repeat_value
                    whole_keys[k][1] += 1

                except:
                    continue
            for k in pro_keys.keys():
                try:
                    repeat_value = pro_keys[k][0][key]
                    repeat_value_now = pro_key_now[key]
                    # 进行排序判断处理
                    if repeat_value[pro_index[2]] > repeat_value_now[pro_index[2]]:
                        data_first = repeat_value_now
                        data_second = repeat_value
                    else:
                        data_first = repeat_value
                        data_second = repeat_value_now
                    if data_first[pro_index[10]] == 'province_in' or data_second[pro_index[10]] == 'province_out':
                        # 如果前面的数据是onlyin，则将之前的跨省该条数据去除，只保留前一条only_in的数据到当前跨省数据中
                        try:
                            interval_disc = set(data_first[pro_index[3]].split('|'))
                        except:
                            interval_disc = ''
                            print(1)
                        try:
                            next_gantry = gantry_relation_disc[data_second[part_index[3]].split('|')[-1]]
                        except:
                            next_gantry = ''
                        interval_key = interval_disc.intersection(next_gantry)
                        if (datetime.datetime.strptime(data_second[part_index[2]],
                                                       "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                                data_first[part_index[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                            if len(interval_key) > 0:
                                money_1 = data_first[part_index[6]]
                                money_2 = data_second[part_index[6]]
                                length_1 = data_first[part_index[7]]
                                length_2 = data_second[part_index[7]]
                                interval_key = list(interval_key)
                                time_list_single = data_first[part_index[5]].split('|')
                                gantry_list_single = data_first[part_index[4]].split('|')
                                interval_string_list_single = data_first[part_index[3]].split('|')
                                new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                                new_gantry.extend(data_second[part_index[4]].split('|'))
                                new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
                                new_interval = interval_string_list_single[
                                               :interval_string_list_single.index(interval_key[0])]
                                new_interval.extend(data_second[part_index[3]].split('|'))
                                new_interval.extend(
                                    interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
                                new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                                new_time.extend(data_second[part_index[5]].split('|'))
                                new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                                data_first[whole_index[11]] = len(new_interval)
                                data_first[whole_index[6]] = float(money_1) + float(money_2)
                                data_first[whole_index[7]] = float(length_1) + float(length_2)
                                data_first[whole_index[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
                                data_first[whole_index[3]] = dbf.list_to_string_with_sign(new_interval, '|')
                                data_first[whole_index[5]] = dbf.list_to_string_with_sign(new_time, '|')
                                part_key_now_drop.append(key)
                                pro_keys[k][0][key] = data_first
                                pro_keys[k][1] += 1
                            else:
                                part_key_now_drop.append(key)
                                # data_first[whole_index[8]] = 1
                                pro_keys[k][0][key] = data_first
                                pro_keys[k][1] += 1
                        else:
                            part_key_now_drop.append(key)
                            data_first[whole_index[8]] = 1
                            pro_keys[k][0][key] = data_first
                            pro_keys[k][1] += 1
                    else:
                        # 如果前面的数据不是onlyin，则将前后两条跨省数据合并，只保留前一条only_in的数据到当前跨省数据中
                        data_second[pro_index[3]] = data_first[pro_index[3]] + '|' + data_second[pro_index[3]]
                        data_second[pro_index[4]] = data_first[pro_index[4]] + '|' + data_second[pro_index[4]]
                        data_second[pro_index[5]] = data_first[pro_index[5]] + '|' + data_second[pro_index[5]]
                        data_second[pro_index[6]] = float(data_first[pro_index[6]]) + float(
                            data_second[pro_index[6]])
                        data_second[pro_index[7]] = float(data_first[pro_index[7]]) + float(
                            data_second[pro_index[7]])
                        data_second[pro_index[11]] = float(data_first[pro_index[11]]) + float(
                            data_second[pro_index[11]])
                        data_second[pro_index[12]] = data_first[pro_index[12]] + '|' + data_second[pro_index[12]]
                        data_second[pro_index[13]] = data_first[pro_index[13]] + '|' + data_second[pro_index[13]]
                        data_second[pro_index[14]] = data_first[pro_index[14]]
                        data_second[pro_index[15]] = data_first[pro_index[15]]
                        data_second[pro_index[16]] = data_first[pro_index[16]]
                        data_second[pro_index[17]] = data_first[pro_index[17]]
                        if data_first[pro_index[10]] == 'province_out' and data_second[pro_index[10]] == 'province_in':
                            data_second[pro_index[10]] = 'whole'
                            pro_keys[k][0].pop(key)
                            pro_key_now_drop.append(key)
                            whole_keys[k][0][key] = data_second
                        elif data_first[pro_index[10]] == 'province_out' and data_second[pro_index[10]] == 'province_pass':
                            data_second[pro_index[10]] = 'province_out'
                            pro_keys[k][0].pop(key)
                            pro_key_now[key] = data_second
                        else:
                            pro_keys[k][0].pop(key)
                            pro_key_now[key] = data_second
                except:
                    continue
        # drop the repeat key
        pro_key_now_save = [pro_columns]
        for key in pro_key_now.keys():
            if key not in pro_key_now_drop:
                pro_key_now_save.append(pro_key_now[key])

        # 匹配当天的whole数据
        for key in whole_key_now.keys():
            for k in loss_keys.keys():
                try:
                    repeat_value = loss_keys[k][0][key]
                    repeat_value_now = whole_key_now[key]
                    # 如果省内完整数据能匹配上缺失数据，直接删除缺失数据，修改完整数据的if_haveCard字段
                    loss_keys[k][0].pop(key)
                    repeat_value_now[pro_index[8]] = 1
                    whole_key_now[key] = repeat_value_now
                except:
                    continue
            for k in whole_keys.keys():
                try:
                    repeat_value = whole_keys[k][0][key]
                    # 如果省内完整的能匹配上省内完整数据，直接当天省内完整数据，并修改完整数据的if_haveCard字段
                    whole_key_now_drop.append(key)
                    repeat_value[whole_index[8]] = 1
                    whole_keys[k][0][key] = repeat_value
                    whole_keys[k][1] += 1

                except:
                    continue
            for k in pro_keys.keys():
                try:
                    repeat_value = pro_keys[k][0][key]
                    # 如果省内完整的能匹配上跨省数据，直接删除当天省内完整数据，并修改跨省数据的if_haveCard字段
                    whole_key_now_drop.append(key)
                    repeat_value[whole_index[8]] = 1
                    pro_keys[k][0][key] = repeat_value
                    pro_keys[k][1] += 1
                except:
                    continue
        # drop the repeat key
        for ii in range(len(whole_key_now_drop)):
            whole_key_now.pop(whole_key_now_drop[ii])

        # 匹配part数据
        for key in part_key_now.keys():
            for k in loss_keys.keys():
                try:
                    repeat_value = loss_keys[k][0][key]
                    repeat_value_now = part_key_now[key]
                    # 如果正常缺失的能匹配上异常缺失数据，且正常缺失为in，则删除正常缺失数据
                    if repeat_value_now[part_index[9]] == 'in' and repeat_value[loss_index[10]] == 'only_in':
                        part_key_now_drop.append(key)
                    elif repeat_value_now[part_index[9]] == 'in' and repeat_value[loss_index[10]] != 'only_in':
                        if repeat_value_now[part_index[2]] >= repeat_value[loss_index[2]]:
                            part_key_now_drop.append(key)
                        else:
                            repeat_value_now[pro_index[3]] = repeat_value_now[pro_index[3]] + '|' + repeat_value[
                                loss_index[3]]
                            repeat_value_now[pro_index[4]] = repeat_value_now[pro_index[4]] + '|' + repeat_value[
                                loss_index[4]]
                            repeat_value_now[pro_index[5]] = repeat_value_now[pro_index[5]] + '|' + repeat_value[
                                loss_index[5]]
                            repeat_value_now[pro_index[6]] = float(repeat_value_now[pro_index[6]]) + float(
                                repeat_value[loss_index[6]])
                            repeat_value_now[pro_index[7]] = float(repeat_value_now[pro_index[7]]) + float(
                                repeat_value[loss_index[7]])
                            repeat_value_now[pro_index[11]] = float(repeat_value_now[pro_index[11]]) + float(
                                repeat_value[loss_index[11]])
                            repeat_value_now[pro_index[12]] = repeat_value_now[pro_index[12]] + '|' + repeat_value[
                                loss_index[12]]
                            repeat_value_now[pro_index[13]] = repeat_value_now[pro_index[13]] + '|' + repeat_value[
                                loss_index[13]]
                            loss_keys[k][0].pop(key)
                            repeat_value_now[pro_index[8]] = 2
                            part_key_now[key] = repeat_value_now
                    # 如果正常缺失的能匹配上异常缺失数据，且正常缺失为out，则删除异常缺失数据，正常缺失数据
                    elif repeat_value_now[part_index[9]] == 'out':
                        if repeat_value_now[part_index[2]] >= repeat_value[loss_index[2]]:
                            loss_keys[k][0].pop(key)
                            repeat_value_now[5] = repeat_value[5]
                            repeat_value_now[6] = repeat_value[6]
                            repeat_value_now[pro_index[3]] = repeat_value[loss_index[3]] + '|' + repeat_value_now[pro_index[3]]
                            repeat_value_now[pro_index[4]] = repeat_value[loss_index[4]] + '|' + repeat_value_now[pro_index[4]]
                            repeat_value_now[pro_index[5]] = repeat_value[loss_index[5]] + '|' + repeat_value_now[pro_index[5]]
                            repeat_value_now[pro_index[6]] = float(repeat_value[loss_index[6]]) + float(
                                repeat_value_now[pro_index[6]])
                            repeat_value_now[pro_index[7]] = float(repeat_value[loss_index[7]]) + float(
                                repeat_value_now[pro_index[7]])
                            repeat_value_now[pro_index[11]] = float(repeat_value[loss_index[11]]) + float(
                                repeat_value_now[pro_index[11]])
                            repeat_value_now[pro_index[12]] = repeat_value[loss_index[12]] + '|' + repeat_value_now[pro_index[12]]
                            repeat_value_now[pro_index[13]] = repeat_value[loss_index[13]] + '|' + repeat_value_now[pro_index[13]]
                            repeat_value_now[pro_index[8]] = 2
                            part_key_now[key] = repeat_value_now
                        else:
                            part_key_now_drop.append(key)
                    # 如果正常缺失的能匹配上异常缺失数据，且正常缺失为none，则删除异常缺失数据，正常缺失数据
                    else:
                        if repeat_value_now[part_index[2]] >= repeat_value[loss_index[2]]:
                            loss_keys[k][0].pop(key)
                            repeat_value_now[pro_index[3]] = repeat_value[loss_index[3]] + '|' + repeat_value_now[
                                pro_index[3]]
                            repeat_value_now[pro_index[4]] = repeat_value[loss_index[4]] + '|' + repeat_value_now[
                                pro_index[4]]
                            repeat_value_now[pro_index[5]] = repeat_value[loss_index[5]] + '|' + repeat_value_now[
                                pro_index[5]]
                            repeat_value_now[pro_index[6]] = float(repeat_value[loss_index[6]]) + float(
                                repeat_value_now[pro_index[6]])
                            repeat_value_now[pro_index[7]] = float(repeat_value[loss_index[7]]) + float(
                                repeat_value_now[pro_index[7]])
                            repeat_value_now[pro_index[11]] = float(repeat_value[loss_index[11]]) + float(
                                repeat_value_now[pro_index[11]])
                            repeat_value_now[pro_index[12]] = repeat_value[loss_index[12]] + '|' + repeat_value_now[
                                pro_index[12]]
                            repeat_value_now[pro_index[13]] = repeat_value[loss_index[13]] + '|' + repeat_value_now[
                                pro_index[13]]
                            repeat_value_now[pro_index[8]] = 2
                            if repeat_value[loss_index[10]] == 'only_in':
                                repeat_value_now[pro_index[9]] = 'in'
                            elif repeat_value[loss_index[10]] == 'none':
                                repeat_value_now[pro_index[9]] = 'none'
                            part_key_now[key] = repeat_value_now
                        else:
                            part_key_now_drop.append(key)
                except:
                    continue
            for k in whole_keys.keys():
                try:
                    repeat_value = whole_keys[k][0][key]
                    repeat_value_now = part_key_now[key]
                    # 进行排序判断处理
                    if repeat_value_now[part_index[9]] != 'none':
                        part_key_now_drop.append(key)
                        repeat_value[whole_index[8]] = 2
                        whole_keys[k][0][key] = repeat_value
                        whole_keys[k][1] += 1
                    else:
                        try:
                            interval_disc = set(repeat_value[whole_index[3]].split('|'))
                        except:
                            interval_disc = ''
                            print(1)
                        try:
                            next_gantry = gantry_relation_disc[repeat_value_now[part_index[3]].split('|')[-1]]
                        except:
                            next_gantry = ''
                        interval_key = interval_disc.intersection(next_gantry)
                        if (datetime.datetime.strptime(repeat_value_now[part_index[2]],
                                                       "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                                repeat_value[whole_index[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                            if len(interval_key) > 0:
                                money_1 = repeat_value[whole_index[6]]
                                money_2 = repeat_value_now[part_index[6]]
                                length_1 = repeat_value[whole_index[7]]
                                length_2 = repeat_value_now[part_index[7]]
                                interval_key = list(interval_key)
                                time_list_single = repeat_value[whole_index[5]].split('|')
                                gantry_list_single = repeat_value[whole_index[4]].split('|')
                                interval_string_list_single = repeat_value[whole_index[3]].split('|')
                                new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                                new_gantry.extend(repeat_value_now[part_index[4]].split('|'))
                                new_gantry.extend(gantry_list_single[gantry_list_single.index(interval_key[0]):])
                                new_interval = interval_string_list_single[
                                               :interval_string_list_single.index(interval_key[0])]
                                new_interval.extend(repeat_value_now[part_index[3]].split('|'))
                                new_interval.extend(
                                    interval_string_list_single[interval_string_list_single.index(interval_key[0]):])
                                new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                                new_time.extend(repeat_value_now[part_index[5]].split('|'))
                                new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                                repeat_value[whole_index[11]] = len(new_interval)
                                repeat_value[whole_index[6]] = float(money_1) + float(money_2)
                                repeat_value[whole_index[7]] = float(length_1) + float(length_2)
                                repeat_value[whole_index[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
                                repeat_value[whole_index[3]] = dbf.list_to_string_with_sign(new_interval, '|')
                                repeat_value[whole_index[5]] = dbf.list_to_string_with_sign(new_time, '|')
                                part_key_now_drop.append(key)
                                whole_keys[k][0][key] = repeat_value
                                whole_keys[k][1] += 1
                            else:
                                part_key_now_drop.append(key)
                                repeat_value[whole_index[8]] = 1
                                whole_keys[k][0][key] = repeat_value
                                whole_keys[k][1] += 1
                        else:
                            part_key_now_drop.append(key)
                            repeat_value[whole_index[8]] = 1
                            whole_keys[k][0][key] = repeat_value
                            whole_keys[k][1] += 1
                except:
                    continue
            for k in pro_keys.keys():
                try:
                    repeat_value = pro_keys[k][0][key]
                    repeat_value_now = part_key_now[key]
                    # 进行排序判断处理
                    if repeat_value[pro_index[10]] != 'province_out':
                        if repeat_value_now[part_index[9]] != 'out' and repeat_value_now[part_index[2]] < repeat_value[pro_index[2]]:
                            if repeat_value_now[part_index[9]] == 'in':
                                repeat_value[pro_index[3]] = repeat_value_now[part_index[3]] + '|' + repeat_value[
                                    pro_index[3]]
                                repeat_value[pro_index[4]] = repeat_value_now[part_index[4]] + '|' + repeat_value[
                                    pro_index[4]]
                                repeat_value[pro_index[5]] = repeat_value_now[part_index[5]] + '|' + repeat_value[
                                    pro_index[5]]
                                repeat_value[pro_index[6]] = float(repeat_value_now[part_index[6]]) + float(
                                    repeat_value[pro_index[6]])
                                repeat_value[pro_index[7]] = float(repeat_value_now[part_index[7]]) + float(
                                    repeat_value[pro_index[7]])
                                repeat_value[pro_index[11]] = float(repeat_value_now[part_index[11]]) + float(
                                    repeat_value[pro_index[11]])
                                repeat_value[pro_index[12]] = repeat_value_now[part_index[12]] + '|' + repeat_value[
                                    pro_index[12]]
                                repeat_value[pro_index[13]] = repeat_value_now[part_index[13]] + '|' + repeat_value[
                                    pro_index[13]]
                                repeat_value[pro_index[10]] = 'whole'
                                part_key_now_drop.append(key)
                                pro_keys[k][0].pop(key)
                                whole_keys[k][0][key] = repeat_value
                            else:
                                repeat_value_now[part_index[3]] = repeat_value_now[part_index[3]] + '|' + repeat_value[
                                    pro_index[3]]
                                repeat_value_now[part_index[4]] = repeat_value_now[part_index[4]] + '|' + repeat_value[
                                    pro_index[4]]
                                repeat_value_now[part_index[5]] = repeat_value_now[part_index[5]] + '|' + repeat_value[
                                    pro_index[5]]
                                repeat_value_now[part_index[6]] = float(repeat_value_now[part_index[6]]) + float(
                                    repeat_value[pro_index[6]])
                                repeat_value_now[part_index[7]] = float(repeat_value_now[part_index[7]]) + float(
                                    repeat_value[pro_index[7]])
                                repeat_value_now[part_index[11]] = float(repeat_value_now[part_index[11]]) + float(
                                    repeat_value[pro_index[11]])
                                repeat_value_now[part_index[12]] = repeat_value_now[part_index[12]] + '|' + repeat_value[
                                    pro_index[12]]
                                repeat_value_now[part_index[13]] = repeat_value_now[part_index[13]] + '|' + repeat_value[
                                    pro_index[13]]
                                repeat_value_now[part_index[9]] = 'out'
                                pro_keys[k][0].pop(key)
                                part_key_now[key] = repeat_value_now
                        elif repeat_value_now[part_index[9]] == 'out':
                            part_key_now_drop.append(key)
                        else:
                            if repeat_value_now[part_index[9]] == 'in':
                                part_key_now_drop.append(key)
                            else:
                                try:
                                    interval_disc = set(repeat_value[whole_index[3]].split('|'))
                                except:
                                    interval_disc = ''
                                    print(1)
                                try:
                                    next_gantry = gantry_relation_disc[repeat_value_now[part_index[3]].split('|')[-1]]
                                except:
                                    next_gantry = ''
                                interval_key = interval_disc.intersection(next_gantry)
                                if (datetime.datetime.strptime(repeat_value_now[part_index[2]],
                                                               "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(
                                    repeat_value[whole_index[2]], "%Y-%m-%d %H:%M:%S")).total_seconds() < 1800:
                                    if len(interval_key) > 0:
                                        money_1 = repeat_value[whole_index[6]]
                                        money_2 = repeat_value_now[part_index[6]]
                                        length_1 = repeat_value[whole_index[7]]
                                        length_2 = repeat_value_now[part_index[7]]
                                        interval_key = list(interval_key)
                                        time_list_single = repeat_value[whole_index[5]].split('|')
                                        gantry_list_single = repeat_value[whole_index[4]].split('|')
                                        interval_string_list_single = repeat_value[whole_index[3]].split('|')
                                        new_gantry = gantry_list_single[:gantry_list_single.index(interval_key[0])]
                                        new_gantry.extend(repeat_value_now[part_index[4]].split('|'))
                                        new_gantry.extend(
                                            gantry_list_single[gantry_list_single.index(interval_key[0]):])
                                        new_interval = interval_string_list_single[
                                                       :interval_string_list_single.index(interval_key[0])]
                                        new_interval.extend(repeat_value_now[part_index[3]].split('|'))
                                        new_interval.extend(
                                            interval_string_list_single[
                                            interval_string_list_single.index(interval_key[0]):])
                                        new_time = time_list_single[:gantry_list_single.index(interval_key[0])]
                                        new_time.extend(repeat_value_now[part_index[5]].split('|'))
                                        new_time.extend(time_list_single[gantry_list_single.index(interval_key[0]):])
                                        repeat_value[whole_index[11]] = len(new_interval)
                                        repeat_value[whole_index[6]] = float(money_1) + float(money_2)
                                        repeat_value[whole_index[7]] = float(length_1) + float(length_2)
                                        repeat_value[whole_index[4]] = dbf.list_to_string_with_sign(new_gantry, '|')
                                        repeat_value[whole_index[3]] = dbf.list_to_string_with_sign(new_interval, '|')
                                        repeat_value[whole_index[5]] = dbf.list_to_string_with_sign(new_time, '|')
                                        part_key_now_drop.append(key)
                                        whole_keys[k][0][key] = repeat_value
                                        whole_keys[k][1] += 1
                                    else:
                                        part_key_now_drop.append(key)
                                        repeat_value[whole_index[8]] = 1
                                        whole_keys[k][0][key] = repeat_value
                                        whole_keys[k][1] += 1
                                else:
                                    part_key_now_drop.append(key)
                                    repeat_value[whole_index[8]] = 1
                                    whole_keys[k][0][key] = repeat_value
                                    whole_keys[k][1] += 1
                    elif repeat_value[pro_index[10]] != 'province_in':
                        if repeat_value_now[part_index[9]] != 'in' and repeat_value_now[part_index[2]] > repeat_value[
                            pro_index[2]] and repeat_value_now[part_index[3]] != repeat_value[pro_index[3]].rsplit('|', 1)[1]:
                            if repeat_value_now[part_index[9]] == 'out':
                                repeat_value[13] = repeat_value_now[13]
                                repeat_value[14] = repeat_value_now[14]
                                repeat_value[15] = repeat_value_now[15]
                                repeat_value[16] = repeat_value_now[16]
                                repeat_value[17] = repeat_value_now[17]
                                repeat_value[18] = repeat_value_now[18]
                                repeat_value[19] = repeat_value_now[19]
                                repeat_value[20] = repeat_value_now[20]
                                repeat_value[21] = repeat_value_now[21]
                                repeat_value[22] = repeat_value_now[22]
                                repeat_value[23] = repeat_value_now[23]
                                repeat_value[24] = repeat_value_now[24]
                                repeat_value[25] = repeat_value_now[25]
                                repeat_value[26] = repeat_value_now[26]
                                repeat_value[27] = repeat_value_now[27]
                                repeat_value[28] = repeat_value_now[28]
                                repeat_value[29] = repeat_value_now[29]
                                repeat_value[30] = repeat_value_now[30]
                                repeat_value[pro_index[3]] = repeat_value_now[part_index[3]] + '|' + repeat_value[
                                    pro_index[3]]
                                repeat_value[pro_index[4]] = repeat_value_now[part_index[4]] + '|' + repeat_value[
                                    pro_index[4]]
                                repeat_value[pro_index[5]] = repeat_value_now[part_index[5]] + '|' + repeat_value[
                                    pro_index[5]]
                                repeat_value[pro_index[6]] = float(repeat_value_now[part_index[6]]) + float(
                                    repeat_value[pro_index[6]])
                                repeat_value[pro_index[7]] = float(repeat_value_now[part_index[7]]) + float(
                                    repeat_value[pro_index[7]])
                                repeat_value[pro_index[11]] = float(repeat_value_now[part_index[11]]) + float(
                                    repeat_value[pro_index[11]])
                                repeat_value[pro_index[12]] = repeat_value_now[part_index[12]] + '|' + repeat_value[
                                    pro_index[12]]
                                repeat_value[pro_index[13]] = repeat_value_now[part_index[13]] + '|' + repeat_value[
                                    pro_index[13]]
                                repeat_value[pro_index[2]] = repeat_value_now[part_index[2]]
                                repeat_value[pro_index[10]] = 'whole'
                                part_key_now_drop.append(key)
                                pro_keys[k][0].pop(key)
                                whole_keys[k][0][key] = repeat_value
                            else:
                                repeat_value_now[part_index[3]] = repeat_value_now[part_index[3]] + '|' + repeat_value[
                                    pro_index[3]]
                                repeat_value_now[part_index[4]] = repeat_value_now[part_index[4]] + '|' + repeat_value[
                                    pro_index[4]]
                                repeat_value_now[part_index[5]] = repeat_value_now[part_index[5]] + '|' + repeat_value[
                                    pro_index[5]]
                                repeat_value_now[part_index[6]] = float(repeat_value_now[part_index[6]]) + float(
                                    repeat_value[pro_index[6]])
                                repeat_value_now[part_index[7]] = float(repeat_value_now[part_index[7]]) + float(
                                    repeat_value[pro_index[7]])
                                repeat_value_now[part_index[11]] = float(repeat_value_now[part_index[11]]) + float(
                                    repeat_value[pro_index[11]])
                                repeat_value_now[part_index[12]] = repeat_value_now[part_index[12]] + '|' + \
                                                                   repeat_value[
                                                                       pro_index[12]]
                                repeat_value_now[part_index[13]] = repeat_value_now[part_index[13]] + '|' + \
                                                                   repeat_value[
                                                                       pro_index[13]]
                                repeat_value_now[part_index[9]] = 'in'
                                pro_keys[k][0].pop(key)
                                part_key_now[key] = repeat_value_now
                        elif repeat_value_now[part_index[9]] != 'in' and repeat_value_now[part_index[2]] > repeat_value[
                            pro_index[2]] and repeat_value_now[part_index[3]] == repeat_value[pro_index[3]].rsplit('|', 1)[1]:
                            part_key_now_drop.append(key)
                        elif repeat_value_now[part_index[2]] < repeat_value[pro_index[2]] and repeat_value_now[part_index[9]] == 'in':
                            if repeat_value[pro_index[13]].split('|', 1)[0] != '1':
                                repeat_value[pro_index[3]] = repeat_value_now[part_index[3]] + '|' + repeat_value[
                                    pro_index[3]]
                                repeat_value[pro_index[4]] = repeat_value_now[part_index[4]] + '|' + repeat_value[
                                    pro_index[4]]
                                repeat_value[pro_index[5]] = repeat_value_now[part_index[5]] + '|' + repeat_value[
                                    pro_index[5]]
                                repeat_value[pro_index[6]] = float(repeat_value_now[part_index[6]]) + float(
                                    repeat_value[pro_index[6]])
                                repeat_value[pro_index[7]] = float(repeat_value_now[part_index[7]]) + float(
                                    repeat_value[pro_index[7]])
                                repeat_value[pro_index[11]] = float(repeat_value_now[part_index[11]]) + float(
                                    repeat_value[pro_index[11]])
                                repeat_value[pro_index[12]] = repeat_value_now[part_index[12]] + '|' + repeat_value[
                                    pro_index[12]]
                                repeat_value[pro_index[13]] = repeat_value_now[part_index[13]] + '|' + repeat_value[
                                    pro_index[13]]
                                part_key_now_drop.append(key)
                                pro_keys[k][0][key] = repeat_value
                                pro_keys[k][1] += 1
                            else:
                                part_key_now_drop.append(key)
                        else:
                            part_key_now_drop.append(key)

                except:
                    continue
        # drop the repeat key
        for ii in range(len(part_key_now_drop)):
            part_key_now.pop(part_key_now_drop[ii])

        # 保存当天的数据
        # dbf.basic_save_dict_data(part_key_now, part_data_path)
        # dbf.basic_save_dict_data(whole_key_now, data_whole_path[i])
        # dbf.basic_save_dict_data(pro_key_now, data_province_path[i])
        # dbf.basic_save_dict_data(loss_key_now, loss_data_path[i])

    # for key in len(loss_keys.keys()):
    #     if len(loss_keys[key][0]) != loss_keys[key][1]:
    #         dbf.basic_save_dict_data(loss_keys[key][0],
    #                                  loss_data_path.rsplit('/', 1)[0] + '/' + key + '.csv')
    #
    #     if len(whole_keys[key][0]) != whole_keys[key][1]:
    #         dbf.basic_save_dict_data(whole_keys[key][0],
    #                                  data_whole_path.rsplit('/', 1)[0] + '/' + key + '.csv')
    #
    #     if len(pro_keys[key][0]) != pro_keys[key][1]:
    #         dbf.basic_save_dict_data(pro_keys[key][0],
    #                                  data_province_path.rsplit('/', 1)[0] + '/' + key + '.csv')


'''
    创建时间：2022/5/25
    完成时间：2022/5/25
    功能: 根据起止日期将每日的中间数据（包括省内完整、跨省和异常缺失）进行合并
    修改时间：
'''


def process_of_combine_middle_data(start_time, end_time):
    """
    根据起止日期将每日的中间数据（包括省内完整、跨省和异常缺失）进行合并
    :param start_time:数据合并起始时间
    :param end_time:数据合并截止时间
    :return:
    """
    # 获取合并所需数据的地址
    data_province_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_province_many_path'), True)
    data_whole_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_whole_many_path'), True)
    loss_data_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('loss_data_many_path'), True)
    # 获取合并后保存的位置
    data_pro_week_path = kp.get_parameter_with_keyword('data_pro_week_path')
    data_whole_week_path = kp.get_parameter_with_keyword('data_whole_week_path')
    loss_data_week_path = kp.get_parameter_with_keyword('loss_data_week_path')

    # 用于保存各类型规定时间内的全部数据
    pro_data = []  # 保存跨省单次画像数据
    whole_data = []
    loss_data = []
    if_col = 0
    for i in range(len(data_whole_path)):
        if start_time <= data_whole_path[i][-12:-4] <= end_time:
            with open(data_province_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if if_col == 0 and j == 0:
                        pro_data.append(row)
                    else:
                        pro_data.append(row)

            with open(data_whole_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if if_col == 0 and j == 0:
                        whole_data.append(row)
                    else:
                        whole_data.append(row)

            with open(loss_data_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if if_col == 0 and j == 0:
                        loss_data.append(row)
                    else:
                        loss_data.append(row)
            if_col = 1

    with open(data_pro_week_path + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(pro_data)

    with open(data_whole_week_path + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(whole_data)

    with open(loss_data_week_path + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(loss_data)


'''
    创建时间：2022/5/25
    完成时间：2022/5/25
    功能: 根据起止日期将每日的车辆单次画像数据（包括省内完整、跨省和绿通）进行合并
    修改时间：
'''


def process_of_combine_result_data(start_time, end_time):
    """
    根据起止日期将每日的车辆单次画像数据（包括省内完整、跨省）进行合并
    :param start_time:
    :param end_time:
    :return:
    """
    # 获取合并所需数据的地址
    result_data_many_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('result_data_many_path'), True)
    LT_oneDay_wash_back_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('LT_oneDay_wash_back_path'), True)

    # 获取合并后保存的位置
    result_data_week_path = kp.get_parameter_with_keyword('result_data_week_path')
    LT_week_wash_path = kp.get_parameter_with_keyword('LT_week_wash_path')

    # 用于保存各类型规定时间内的全部数据
    result_data = []
    LT_data = []
    if_col = 0  # 用于判断是否是第一个文件的第一行，如果是就保存该字段名

    # 遍历将满足时间范围的数据进行保存
    for i in range(len(result_data_many_path)):
        num = 0
        if start_time <= result_data_many_path[i][-12:-4] <= end_time:
            with open(result_data_many_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if if_col == 0 and j == 0:
                        result_data.append(row)
                    else:
                        result_data.append(row)
                        num += 1
            print(num)
            if_col = 1
    if_col = 0
    for i in range(len(LT_oneDay_wash_back_path)):
        if start_time <= LT_oneDay_wash_back_path[i][-14:-4].replace('-', '') <= end_time:
            with open(LT_oneDay_wash_back_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if if_col == 0 and j == 0:
                        LT_data.append(row)
                    else:
                        LT_data.append(row)
            if_col = 1
    print(len(result_data))
    # 保存数据到指定的地址
    with open(result_data_week_path + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)

    with open(LT_week_wash_path + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(LT_data)


'''
    创建时间：2022/5/25
    完成时间：2022/5/25
    功能: 根据起止日期将每日的车辆长期画像数据（包括上下高速时段、路网OD、绿通OD、绿通中间数据、绿通长期画像和收费上期画像1和2）进行合并
    修改时间：
'''


def process_of_combine_portary_data(start_time, end_time, treat_type='add'):
    """
    根据起止日期将每日的车辆单次画像数据（包括省内完整、跨省）进行合并
    :param start_time:
    :param end_time:
    :return:
    """
    # 用于保存各类型规定时间内的全部数据
    whole_OD_data = {}
    whole_time_data = {}
    LT_OD_data = {}
    portray_combine_data = {}
    portray_result_data = {}
    LT_middle_data = {}
    LT_midlle_vehicle = {}
    if_col = 0

    # 获取合并所需长期画像相关数据的地址
    whole_OD_num = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_whole_OD_num'), True)
    whole_time_num = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_whole_time_num'), True)
    LT_OD_num = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_LT_OD_num'), True)
    LT_middle_many_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('LT_middle_one_path'), True)
    vehicle_portray_combine = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_vehicle_portray_combine'), True)
    vehicle_portray_result = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_vehicle_portray_result'), True)
    # vehicle_portray_LT = dop.path_of_holder_document(kp.get_parameter_with_keyword('back_vehicle_portray_LT'), True)

    if treat_type == 'add':
        # 获取合并后保存的位置
        whole_OD_num_week = kp.get_parameter_with_keyword('whole_OD_num_week')
        whole_time_num_week = kp.get_parameter_with_keyword('whole_time_num_week')
        LT_OD_num_week = kp.get_parameter_with_keyword('LT_OD_num_week')
        vehicle_portray_combine_week = kp.get_parameter_with_keyword('vehicle_portray_combine_week')
        vehicle_portray_result_week = kp.get_parameter_with_keyword('vehicle_portray_result_week')
        vehicle_portray_LT_week = kp.get_parameter_with_keyword('vehicle_portray_LT_week')
        LT_middle_week = kp.get_parameter_with_keyword('LT_middle_week')
    else:
        whole_OD_num_week = kp.get_parameter_with_keyword('whole_OD_num_history')
        whole_time_num_week = kp.get_parameter_with_keyword('whole_time_num_history')
        LT_OD_num_week = kp.get_parameter_with_keyword('LT_OD_num_history')
        vehicle_portray_combine_week = kp.get_parameter_with_keyword('vehicle_portray_combine_history')
        vehicle_portray_result_week = kp.get_parameter_with_keyword('vehicle_portray_result_history')
        vehicle_portray_LT_week = kp.get_parameter_with_keyword('vehicle_portray_LT_history')
        LT_middle_week = kp.get_parameter_with_keyword('LT_middle_history')
        with open(whole_OD_num_week) as f:
            for j, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if j > 0:
                    whole_OD_data[row[0] + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4]] = float(row[5])
        with open(whole_time_num_week) as f:
            for j, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if j > 0:
                    data_ls = [float(x) for x in row[1:]]
                    whole_time_data[row[0]] = data_ls
        # with open(LT_OD_num_week) as f:
        #     for j, row in enumerate(f):
        #         row = row.split(',')
        #         row[-1] = row[-1][:-1]
        #         if j > 0:
        #             LT_OD_data[row[0] + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '-' + row[5] + '-' + row[6] + '-' + row[7] + '-' + row[8] + '-' + row[9] + '-' + row[10]] = float(row[11])
        with open(vehicle_portray_combine_week) as f:
            for j, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if j > 0:
                    portray_combine_data[row[0]] = row
        # with open(vehicle_portray_result_week) as f:
        #     for j, row in enumerate(f):
        #         row = row.split(',')
        #         row[-1] = row[-1][:-1]
        #         if j > 0:
        #             portray_result_data[row[0]] = row
        # with open(LT_middle_week) as f:
        #     for j, row in enumerate(f):
        #         row = row.split(',')
        #         row[-1] = row[-1][:-1]
        #         if j == 0:
        #             LT_middle_columns = row
        #             columns_index = dbf.get_indexs_of_list(row, LT_middle_columns)
        #         elif j > 0:
        #             LT_midlle_vehicle[row[0]] = 1
        #             for k in range(len(LT_middle_columns)):
        #                 if k > 0:
        #                     if LT_middle_columns[k] == '作弊时间':
        #                         LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = row[columns_index[k]]
        #                     else:
        #                         LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = float(row[columns_index[k]])

    # 遍历将满足时间范围的数据根据车牌进行加总
    # for i in range(len(whole_OD_num)):
    #     if start_time <= whole_OD_num[i][-12:-4] <= end_time:
    #         if '20220129' < whole_OD_num[i][-12:-4] < '20220208' or '20210929' < whole_OD_num[i][-12:-4] < '20211009':
    #             continue
    #         print(whole_OD_num[i][-12:-4])
            # with open(whole_OD_num[i]) as f:
            #     for j, row in enumerate(f):
            #         row = row.split(',')
            #         row[-1] = row[-1][:-1]
            #         if if_col == 0 and j == 0:
            #             whole_OD_columns = row
            #         elif j > 0:
            #             if treat_type == 'add':
            #                 try:
            #                     whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]] = float(whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]]) + float(row[5])
            #                 except:
            #                     whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]] = float(row[5])
            #             else:
            #                 if float(whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]]) - float(row[5]) == 0:
            #                     whole_OD_data.pop(row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4])
            #                 else:
            #                     whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]] = float(whole_OD_data[row[0]+'-'+row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]]) - float(row[5])

                    # if j > 10000:
                    #     break
            #
            # with open(whole_time_num[i]) as f:
            #     for j, row in enumerate(f):
            #         row = row.split(',')
            #         row[-1] = row[-1][:-1]
            #         if if_col == 0 and j == 0:
            #             whole_time_columns = row
            #         elif j > 0:
            #             if treat_type == 'add':
            #                 try:
            #                     data_ls = whole_time_data[row[0]]
            #                     for k in range(len(data_ls)):
            #                         data_ls[k] = float(data_ls[k]) + float(row[k+1])
            #                     whole_time_data[row[0]] = data_ls
            #                 except:
            #                     data_ls = [float(x) for x in row[1:]]
            #                     whole_time_data[row[0]] = data_ls
            #             else:
            #                 try:
            #                     data_ls = whole_time_data[row[0]]
            #                 except:
            #                     continue
            #                 for k in range(len(data_ls)):
            #                     data_ls[k] = float(data_ls[k]) - float(row[k + 1])
            #                 if sum(data_ls) == 0:
            #                     whole_time_data.pop(row[0])
            #                 else:
            #                     whole_time_data[row[0]] = data_ls
            #
            # with open(vehicle_portray_combine[i]) as f:
            #     for j, row in enumerate(f):
            #         row = row.split(',')
            #         row[-1] = row[-1][:-1]
            #         if if_col == 0 and j == 0:
            #             portray_combine_columns = row
            #         elif j > 0:
            #             if treat_type == 'add':
            #                 try:
            #                     data_ls = portray_combine_data[row[0]]
            #                     for k in range(len(data_ls)):
            #                         if k > 0 and k != 9 and k != 13 and k != 17 and k != 18:
            #                             data_ls[k] = float(data_ls[k]) + float(row[k])
            #                         elif k == 13:
            #                             data_ls[k] = row[k]
            #                         elif k > 0:
            #                             data_ls[k] = ''
            #                     portray_combine_data[row[0]] = data_ls
            #                 except:
            #                     portray_combine_data[row[0]] = row
            #             else:
            #                 # if row[0] == '豫ML0385_0':
            #                 #     print(1)
            #                 data_ls = portray_combine_data[row[0]]
            #                 for k in range(len(data_ls)):
            #                     if k > 0 and k != 9 and k != 13 and k != 17 and k != 18:
            #                         data_ls[k] = float(data_ls[k]) - float(row[k])
            #                 if data_ls[1] == 0:
            #                     portray_combine_data.pop(row[0])
            #                 else:
            #                     portray_combine_data[row[0]] = data_ls

            # with open(vehicle_portray_result[i]) as f:
            #     for j, row in enumerate(f):
            #         row = row.split(',')
            #         row[-1] = row[-1][:-1]
            #         if if_col == 0 and j == 0:
            #             portray_result_columns = row
            #         elif j > 0:
            #             if treat_type == 'add':
            #                 try:
            #                     data_ls = portray_result_data[row[0]]
            #                     for k in range(len(data_ls)):
            #                         if k > 0 and k != 1 and k != 2 and k != 19 and k != 20:
            #                             data_ls[k] = float(data_ls[k]) + float(row[k])
            #                         elif k > 0:
            #                             if row[k] == '1' or data_ls[k] == '1':
            #                                 data_ls[k] = '1'
            #                     portray_result_data[row[0]] = data_ls
            #                 except:
            #                     portray_result_data[row[0]] = row
            #             else:
            #                 data_ls = portray_result_data[row[0]]
            #                 sum = 0
            #                 for k in range(len(data_ls)):
            #                     if k > 0 and k != 1 and k != 2 and k != 19 and k != 20:
            #                         data_ls[k] = float(data_ls[k]) - float(row[k])
            #                         sum += data_ls[k]
            #                 if sum == 0:
            #                     portray_result_data.pop(row[0])
            #                 else:
            #                     portray_result_data[row[0]] = data_ls
            #         # if j > 10000:
            #         #     break
            # if_col = 1
    # portray_combine_data = fp.get_feature_of_whole_OD_time(whole_OD_data, whole_time_data, portray_combine_data)

    if_col = 0
    for i in range(len(LT_OD_num)):
        if start_time <= LT_OD_num[i][-14:-4].replace('-', '') <= end_time:
    #         # if '20220129' < LT_OD_num[i][-14:-4].replace('-', '') < '20220209' or '20210929' < LT_OD_num[i][-14:-4].replace('-', '') < '20211010':
    #         #     continue
            print(LT_OD_num[i][-14:-4])
    #         # with open(LT_OD_num[i]) as f:
    #         #     LT_OD_num[i][-14:-4]
    #         #     for j, row in enumerate(f):
    #         #         row = row.split(',')
    #         #         row[-1] = row[-1][:-1]
    #         #         if if_col == 0 and j == 0:
    #         #             LT_OD_columns = row
    #         #         elif j > 0:
    #         #             try:
    #         #                 LT_OD_data[row[0] + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '-' + row[5] + '-' + row[6] + '-' + row[7] + '-' + row[8] + '-' + row[9] + '-' + row[10]] = LT_OD_data[row[0] + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '-' + row[5] + '-' + row[6] + '-' + row[7] + '-' + row[8] + '-' + row[9] + '-' + row[10]] + float(row[11])
    #         #             except:
    #         #                 LT_OD_data[row[0] + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '-' + row[5] + '-' + row[6] + '-' + row[7] + '-' + row[8] + '-' + row[9] + '-' + row[10]] = float(row[11])
    #                 # if j > 10000:
    #                 #     break
            with open(LT_middle_many_path[i]) as f:
                for j, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if j == 0:
                        LT_middle_columns = row
                        columns_index = dbf.get_indexs_of_list(row, LT_middle_columns)
                    elif j > 0:
                        LT_midlle_vehicle[row[0]] = 1
                        for k in range(len(LT_middle_columns)):
                            if k > 0:
                                if LT_middle_columns[k] == '作弊时间':
                                    try:
                                        data_ls = LT_middle_data[row[0] + '-' + LT_middle_columns[k]]
                                        if len(data_ls) < 6:
                                            LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = row[columns_index[k]]
                                        elif len(row[columns_index[k]]) < 6:
                                            LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = LT_middle_data[row[0] + '-' + LT_middle_columns[k]]
                                        elif row[columns_index[k]] > LT_middle_data[row[0] + '-' + LT_middle_columns[k]]:
                                            LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = row[columns_index[k]]
                                    except:
                                        LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = row[columns_index[k]]
                                elif LT_middle_columns[k] == '跨省比':
                                    try:
                                        data_ls = LT_middle_data[row[0] + '-' + LT_middle_columns[k]]
                                        LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = round((LT_middle_data[row[0] + '-' + LT_middle_columns[k-1]] / (LT_middle_data[row[0] + '-' + LT_middle_columns[k-1]] + LT_middle_data[row[0] + '-' + LT_middle_columns[k-2]])) * 100, 2)
                                    except:
                                        LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = float(row[columns_index[k]])
                                else:
                                    try:
                                        data_ls = LT_middle_data[row[0] + '-' + LT_middle_columns[k]]
                                        LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = float(LT_middle_data[row[0] + '-' + LT_middle_columns[k]]) + float(
                                            row[columns_index[k]])
                                    except:
                                        LT_middle_data[row[0] + '-' + LT_middle_columns[k]] = float(row[columns_index[k]])
    #                 if j > 10000:
    #                     break
            if_col = 1

    # if treat_type == 'add':
    #     save_name = whole_OD_num_week + start_time + '-' + end_time + '.csv'
    # else:
    #     save_name = whole_OD_num_week
    # save_data = [whole_OD_columns]
    # for key in whole_OD_data.keys():
    #     key_values = key.split('-')
    #     key_values.append(whole_OD_data[key])
    #     save_data.append(key_values)
    # with open(save_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(save_data)
    #
    # if treat_type == 'add':
    #     save_name = whole_time_num_week + start_time + '-' + end_time + '.csv'
    # else:
    #     save_name = whole_time_num_week
    # save_data = [whole_time_columns]
    # for key in whole_time_data.keys():
    #     key_values = [key]
    #     key_values.extend(whole_time_data[key])
    #     save_data.append(key_values)
    # with open(save_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(save_data)

    # if treat_type == 'add':
    #     save_name = vehicle_portray_combine_week + start_time + '-' + end_time + '.csv'
    # else:
    #     save_name = vehicle_portray_combine_week
    # save_data = [portray_combine_columns]
    # for key in portray_combine_data.keys():
    #     key_values = portray_combine_data[key]
    #     save_data.append(key_values)
    # with open(save_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(save_data)

    # if treat_type == 'add':
    #     save_name = vehicle_portray_result_week + start_time + '-' + end_time + '.csv'
    # else:
    #     save_name = vehicle_portray_result_week
    # save_data = [portray_result_columns]
    # for key in portray_result_data.keys():
    #     key_values = portray_result_data[key]
    #     save_data.append(key_values)
    # with open(save_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(save_data)

    # save_data = [LT_OD_columns]
    # for key in LT_OD_data.keys():
    #     key_values = key.split('-')
    #     key_values.append(LT_OD_data[key])
    #     save_data.append(key_values)
    # with open(LT_OD_num_week + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(save_data)
    #
    LT_middle_columns = ['车牌(全)']
    LT_middle_columns.extend(kp.get_parameter_with_keyword('LT_kind'))
    LT_middle_columns.extend(kp.get_parameter_with_keyword('LT_province'))
    LT_middle_columns.extend(['绿通省内次数', '绿通跨省次数', '跨省比'])
    LT_middle_columns.extend(kp.get_parameter_with_keyword('LT_create_type'))
    LT_middle_columns.extend(['coach_change', '是否异常上下高速', '作弊次数', '作弊时间'])
    save_data = [LT_middle_columns]
    for key in LT_midlle_vehicle.keys():
        data_ls = []
        for i in range(len(LT_middle_columns)):
            if i == 0:
                data_ls.append(key)
            else:
                try:
                    value = LT_middle_data[key + '-' + LT_middle_columns[i]]
                    data_ls.append(value)
                except:
                    data_ls.append(0)
        save_data.append(data_ls)
    with open(LT_middle_week + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(save_data)

    portray_LT_value = fp.get_feature_of_LvTong_data(LT_middle_week + start_time + '-' + end_time + '.csv', ifReturn=True)
    # portray_LT_value = fp.get_feature_of_LvTong_data('./4.poratry_data/last_week_data/LT_middle.csv', ifReturn=True)
    with open(vehicle_portray_LT_week + start_time + '-' + end_time + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(portray_LT_value)


'''
    创建时间：2022/5/23
    完成时间：2022/5/23
    功能: 
    修改时间：
'''


def compute_num_of_col(data, key_column, save_columns=[], charge_columns=[], key_data={}, add_data=''):
    """

    :param data:
    :param column:
    :return:
    """
    save_data = []
    values = data[key_column].values
    for i in range(len(save_columns)):
        save_data.append(data[save_columns[i]].values)
    for i in range(len(values)):
        try:
            if charge_columns:
                para = 0
                index = dbf.get_indexs_of_list(save_columns, charge_columns)
                for k in range(len(index)):
                    if save_data[k][i] != key_data[values[i]][k+1][-1]:
                        para = 1
                if para == 0:
                    continue

            if save_columns:
                key_data[values[i]][0] += 1
                for j in range(1, len(save_columns)+1):
                    key_data[values[i]][j].append(save_data[j][i])
                if add_data != '':
                    key_data[values[i]][-1].append(add_data)
            else:
                key_data[values[i]] += 1



        except:
            if save_columns:
                data_ls = []
                key_data[values[i]] = [1]
                for j in range(len(save_columns)):
                    data_ls.append([save_data[j][i]])
                if add_data != '':
                    data_ls.append([add_data])
                key_data[values[i]].extend(data_ls)
            else:
                key_data[values[i]] = 1

    return key_data


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能: 门架路径两端省界问题处理
    修改时间：
'''


def process_of_abnormal_progantry(data, col_gantry, col_proType, disc_path, disc_type, treat_type):
    """
    门架路径两端省界问题处理
    :param data: 输入数据，DataFrame类型
    :param col_name: 处理列名称
    :param treat_type: 处理方式，如首尾处理为all，首处理为in，尾处理为out
    :return:
    """
    # 获取收费单元路径的数据，并以数组类型获得
    gantry_path_list = data[col_gantry].values
    # 获取收费单元类型串的数据，并以数组类型获得
    gantry_type_list = data[col_proType].values
    for i in range(len(gantry_path_list)):
        # 如果对最后单元是否为临近省界进行判断和处理
        if treat_type == 'charge':
            if len(gantry_type_list[i]) == 1 or (len(gantry_type_list[i]) == 2 and gantry_type_list[i][-1] == '0'):
                tail = gantry_path_list[i].split('|')[-1]
                next_tail = dbf.get_next_n_from_disc(tail, disc_path, 1)
                if next_tail:
                    result = dbf.get_corespondance_from_disc(next_tail, disc_type)
                    if '3' in result:
                        gantry_path_list[i] = gantry_path_list[i] + '|' + next_tail[result.index('3')]
                        if len(gantry_type_list[i]) == 2:
                            gantry_type_list[i] = gantry_type_list[i][0] + result[result.index('3')]
                        else:
                            gantry_type_list[i] = result[result.index('3')]
                    elif '2' in result:
                        gantry_path_list[i] = gantry_path_list[i] + '|' + next_tail[result.index('2')]
                        if len(gantry_type_list[i]) == 2:
                            gantry_type_list[i] = gantry_type_list[i][0] + result[result.index('2')]
                        else:
                            gantry_type_list[i] = result[result.index('2')]
        else:
            # 获取该收费单元路径的某些位置是否为省界类型
            result = charge_gantryType_if_province(gantry_path_list[i], disc_type, treat_type)
            # 如果首尾都需要进行判断
            if treat_type == 'all':
                if result == '00':
                    continue
                elif result == '10':
                    gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, gantry_type_list[i][0], 'in')
                elif result == '01':
                    gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, gantry_type_list[i][1], 'out')
                elif result == '11':
                    gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, [gantry_type_list[i][0], gantry_type_list[i][1]], 'all')
            # 如果只对入口是否为省界进行处理
            elif treat_type == 'in':
                if result == '0':
                    continue
                elif result == '1':
                    gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, gantry_type_list[i][0], 'in')
            # 如果只对出口是否为省界进行处理
            elif treat_type == 'out':
                if result == '0':
                    continue
                elif result == '1':
                    if len(gantry_type_list[i]) == 1:
                        gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, gantry_type_list[i][0],
                                                                 'out')
                    else:
                        gantry_path_list[i] = move_progantry_out(gantry_path_list[i], disc_type, gantry_type_list[i][1],
                                                                 'out')

    data[col_gantry] = gantry_path_list
    data[col_proType] = gantry_type_list

    return data


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能: 判断是否两端含有省界门架
    修改时间：
'''


def charge_gantryType_if_province(gantry_list, data_disc, type):
    """

    :param data_disc:
    :param gantry_list:
    :return:
    """
    if type == 'all':
        first_result = dbf.charge_someone_of_data_same_someting(gantry_list.split('|'), data_disc, 0, ['2', '3'])
        second_result = dbf.charge_someone_of_data_same_someting(gantry_list.split('|'), data_disc, -1, ['2', '3'])
        return str(first_result) + str(second_result)
    elif type == 'in':
        first_result = dbf.charge_someone_of_data_same_someting(gantry_list.split('|'), data_disc, 0, ['2', '3'])
        return str(first_result)
    elif type == 'out':
        second_result = dbf.charge_someone_of_data_same_someting(gantry_list.split('|'), data_disc, -1, ['2', '3'])
        return str(second_result)


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能: 将省界收费单元移动到最外端
    修改时间：
'''


def move_progantry_out(gantry_list, data_disc, value, move_type):
    """
    将省界收费单元移动到最外端
    :param gantry_list:
    :param data_disc:
    :param value:
    :return:
    """
    gantry_list = gantry_list.split('|')
    end = 0
    for i in range(len(gantry_list)):
        if move_type == 'in':
            if dbf.charge_someone_of_data_same_someting(gantry_list, data_disc, i, value) == 0:
                gantry_list[i], gantry_list[0] = gantry_list[0], gantry_list[i]
                break
        elif move_type == 'out':
            if dbf.charge_someone_of_data_same_someting(gantry_list, data_disc, len(gantry_list) - i - 1, value) == 0:
                gantry_list[len(gantry_list) - i - 1], gantry_list[-1] = gantry_list[-1], gantry_list[len(gantry_list) - i - 1]
                break
        else:
            if dbf.charge_someone_of_data_same_someting(gantry_list, data_disc, i, value[0]) == 0:
                gantry_list[i], gantry_list[-1] = gantry_list[-1], gantry_list[i]
                end += 1
            if dbf.charge_someone_of_data_same_someting(gantry_list, data_disc, len(gantry_list) - i - 1, value[1]) == 0:
                gantry_list[len(gantry_list) - i - 1], gantry_list[-1] = gantry_list[-1], gantry_list[len(gantry_list) - i - 1]
                end += 1
            if end == 2:
                break
    gantry_list = dbf.list_to_string_with_sign(gantry_list, '|')
    return gantry_list


def data_of_GPS_for_out(data, feature, name, encoding='utf-8'):
    """
    对GPS数据进行数据处理，整理出所需格式数据
    :param paths:
    :param features:
    :return:
    """
    feature_list = feature.split(';')
    feature_list.append('lon_lat')
    length = data.shape[0]
    range_num = int(length / 500000)
    data_list = []
    for i in range((range_num + 1)):
        start = 500000 * i
        end = (500000 * (i + 1)) - 1
        if end >= length:
            end = length
        data_ls = data.loc[start:end]
        # for j, fea in enumerate(feature_list):
        # data_ls[fea] = data_ls[feature].str.split(';').str[j]
        iplist = list(data_ls['CARNUM;CARCOLOR;TIMESTAMPSTR;LON;LAT;VEC1;VEC2;VEC3;DIRECTION;ALTITUDE;STATE;ALARM'])
        itlist = [x.split(';') for x in iplist]
        drop_list = []
        for i in range(len(itlist)):
            itlist[i][0] = itlist[i][0].strip()
            itlist[i][2] = itlist[i][2][:19]
            itlist[i][3] = str(float(itlist[i][3]) / 1000000)
            itlist[i][4] = str(float(itlist[i][4]) / 1000000)
            itlist[i].append(itlist[i][3] + ',' + itlist[i][4])

        data_ls = pd.DataFrame(itlist, columns=feature_list)
        # pd.DataFrame(itlist, columns=feature_list)
        # data_ls = data_ls.drop([feature], axis=1)
        # data_ls[feature_list[0]] = data_ls[feature_list[0]].map(lambda x: x.strip())
        # # 对时间进行处理，保留所需的部分
        # data_ls['TIMESTAMPSTR'] = data_ls['TIMESTAMPSTR'].map(lambda x: x[:19])
        # # 去掉位置没有变化的记录
        data_ls = data_ls.groupby(['CARNUM', 'LON', 'LAT']).head(1)
        # # 对经纬度进行处理
        # data_ls['LON'] = data_ls['LON'].map(lambda x: str(float(x) / 1000000))
        # data_ls['LAT'] = data_ls['LAT'].map(lambda x: str(float(x) / 1000000))
        # # 将经纬度进行合并
        # data_ls['lon_lat'] = data_ls['LON'] + ',' + data_ls['LAT']
        data_ls = data_ls.drop(['LON', 'LAT'], axis=1)
        data_list.append(data_ls)
    data_all = dp.Combine_Document(data_list)
    # feature_list.append('lon_lat')
    # data_all = pd.DataFrame(data_list, columns=feature_list)
    data_all = data_all.groupby(['CARNUM', 'lon_lat']).head(1)
    data_all.to_csv(name, encoding=encoding)


'''
    创建时间：2021/11/28
    完成时间：2021/11/28
    功能: 将所有的GPS数据的汇总数据，进行统一处理，选出所需字段，去重，合并坐标和时间
    修改时间：
'''


def data_of_vehicle_GPS(data_all):
    data_ls = data_all[['CARNUM', 'TIMESTAMPSTR', 'lon_lat']]
    data_dice = split_iden_data_by_plate(data_ls, 'CARNUM')
    dice_list = list(data_dice.keys())
    for dise in dice_list:
        name = './GPS数据分车牌/' + dise + '.csv'
        data_di = data_dice[dise]
        data_di = data_di.sort_values(['CARNUM', 'TIMESTAMPSTR'], ascending=True)
        # 针对各车辆分别将其时间序列合并在一起
        data_time = pd.DataFrame(data_di.groupby('CARNUM')['TIMESTAMPSTR'].apply(lambda x: x.str.cat(sep="|")))
        # 针对各车辆分别将其行驶坐标序列合并在一起
        data_lon_lat = pd.DataFrame(data_di.groupby('CARNUM')['lon_lat'].apply(lambda x: x.str.cat(sep="|")))
        # 将每辆车的时间序列和行驶坐标序列合在一起
        data_all = pd.concat((data_time, data_lon_lat), axis=1)
        data_all.to_csv(name)


'''
    创建时间：2021/11/09
    完成时间：2021/11/09
    修改时间：No.1 2022/2/24，将判断数据类型的条件进行了更改，目前是通过middle_type字段进行判断
            No.2 2022/4/25, let list not the DataFrame
'''


def del_duplicated_in_data(data):
    """
    将因为原始数据异常造成的出入口数据缺失的数据进行合并，去除无效的入口信息（入口信息重复但是PASSID不同）
    :return:
    """

    data_only_in = data[(data['middle_type'] == 'in') & (data['门架数'] == 1)]  # 获取到只有入口信息没有出口信息且门架数为1的数据
    data_no_in = data[(data['middle_type'] == 'out') | (data['middle_type'] == 'none')]  # 获取到所有没有入口信息的数据
    # 获取到完整的数据和只有入口信息没有出口信息且门架数不为1的数据
    data_other = data[(data['middle_type'] == 'whole') | ((data['middle_type'] == 'in') & (data['门架数'] != 1))]

    # 2022/2/24更改，将字段名从车牌和上高速时间，更改为入口车牌和入口时间
    data_only_in_ls = data_only_in[['入口车牌', '入口时间']]
    # 获取入口的时间并向前推3分钟，作为搜寻入口时间的起始时间
    data_only_in_ls['start'] = data_only_in_ls['入口时间'].map(lambda x: x + datetime.timedelta(minutes=-3))
    # 获取出口的时间并向后推3分钟，作为搜寻入口时间的截止时间
    data_only_in_ls['end'] = data_only_in_ls['入口时间'].map(lambda x: x + datetime.timedelta(minutes=3))

    result = []  # 存储每条记录的有效情况，1为有效，0为无效
    vehicle = data_only_in_ls['入口车牌'].values  # 获取所有只有入口信息的行驶记录的车辆列表
    start = data_only_in_ls['start'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).values  # 获取搜寻入口时间的起始时间列表
    end = data_only_in_ls['end'].map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).values  # 获取搜寻入口时间的截止时间列表
    for i in range(len(vehicle)):  # 遍历所有的车辆列表，进行逐条判断，如果无效进行去除，如果相互缺失进行合并
        # 从other数据中，获取车牌一致且时间符合的数据
        data_ls = data_other[
            (data_other['入口车牌'] == vehicle[i]) & (data_other['入口时间'] <= end[i]) & (data_other['入口时间'] >= start[i])]
        # 如果没有获取到数据，则在无入口的数据中，查找入口车牌一样，入口时间符合的数据
        if data_ls.empty:
            # 在无入口的数据中，查找入口车牌一样，入口时间符合的数据
            data_one = data_no_in[(data_no_in['入口车牌'] == vehicle[i]) & (
                    ((data_no_in['入口门架时间'] <= end[i]) & (data_no_in['入口门架时间'] >= start[i])) | (
                    (data_no_in['入口时间'] <= end[i]) & (data_no_in['入口时间'] >= start[i])))]
            if data_one.empty:
                # 如果没有找到，说明该条只有入口的数据是有效的，赋值为1
                result.append(1)
            else:
                # 如果找到了数据，将无入口的数据中的入口信息字段进行赋值
                # index = data_one.index.values  # 获取需要合并的记录的index
                # tack the data of only_in in the data of not in
                # data_no_in.loc[index[0], '入口车牌(全)':'入口重量'] = data_only_in.iloc[i, 0:11]  # 2022/4/24 change
                result.append(0)
        else:  # 如果获取到数据，说明该条只有入口的数据是无效的，赋值为0
            result.append(0)

    # 将无效的数据进行删除
    # 将有无效的结果赋给data_only_in数据
    data_only_in['charge'] = result
    data_only_in = data_only_in[data_only_in['charge'] == 1]  # 保留有效的数据
    data_only_in = data_only_in.drop(['charge'], axis=1)  # 删除有无效的结果字段
    # 获取到已经填补了入口信息的数据
    data_no_in_old_none = data_no_in[(data_no_in['入口ID'].notnull()) & (data_no_in['middle_type'] == 'none')]
    data_no_in_old_out = data_no_in[(data_no_in['入口通行介质'].notnull()) & (data_no_in['middle_type'] == 'out')]
    # 获取到没有填补入口信息的数据
    data_no_in_new_none = data_no_in[(data_no_in['入口ID'].isnull()) & (data_no_in['middle_type'] == 'none')]
    data_no_in_new_out = data_no_in[(data_no_in['入口通行介质'].isnull()) & (data_no_in['middle_type'] == 'out')]

    data_all = pd.concat((data_other, data_no_in_old_none, data_no_in_old_out, data_no_in_new_none, data_no_in_new_out,
                          data_only_in), axis=0)

    return data_all


'''
    创建时间：2022/2/21
    完成时间：2022/2/21
    功能：将稽查算法各阶段的数据写入数据库
    修改时间：No.1
'''


def save_data_to_Mysql(write_type, data, save_type='csv'):
    """
    将稽查算法各阶段的数据写入数据库
    :param write_type:写入数据的类型，middle代表为中间数据写入，timePart代表一段时间合并处理的数据写入，checkFeature代表稽查特征处理表写入，checkResult代表稽查结果写入
    :param data:需要写入的数据
    :return:
    """
    # 创建与稽查服务器mysql数据库连接的对象
    db = gd.OperationMysql('192.168.0.182', 3306, 'root', '123456', 'vehicle_check')
    # 如果write_type为middle, 则针对中间数据进行保存
    if write_type == 'middle':
        db.write_one('middle_data',
                     ["TID", "PASSID", "enVehiclePlateTotal", "enIdenVehiclePlate", "enVehiclePlate", "enPlateColor",
                      "entryStationID", "entryStationHEX", "entryTime", "enMediaType", "enVehicleType",
                      "enVehicleClass", "entryWeight", "enAxleCount", "exVehiclePlateTotal", "exIdenVehiclePlate",
                      "exVehiclePlate", "exPlateColor", "exitStationID", "exitTime", "exMediaType", "exVehicleType",
                      "exVehicleClass", "exitWeight", "exAxleCount", "obuVehicleType", "obuSn", "etcCardId",
                      "ExitFeeType", "exOBUVehiclePlate", "exCPUVehiclePlate", "payFee", "middle_type", "gantryPath",
                      "intervalPath", "gantryNum", "gantryTimePath", "gantryTypePath", "gantryTotalFee",
                      "gantryTotalLength", "firstGantryTime", "endGantryTime"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'datetime', 'datetime', 'float',
                      'float', 'float', 'float', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'float', 'float', 'float', 'float', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'float', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'float', 'int', 'datetime', 'datetime'])
    # 如果write_type为timePart, 则针对一段时间的合并数据进行保存
    elif write_type == 'timePart':
        db.write_one('partTimeConcat_data_20220524',
                     ["PASSID", "enVehiclePlateTotal", "enIdenVehiclePlate", "enVehiclePlate", "enPlateColor",
                      "entryStationID", "entryStationHEX", "entryTime", "enMediaType", "enVehicleType",
                      "enVehicleClass", "entryWeight", "enAxleCount", "exVehiclePlateTotal", "exIdenVehiclePlate",
                      "exVehiclePlate", "exPlateColor", "exitStationID", "exitTime", "exMediaType", "exVehicleType",
                      "exVehicleClass", "exitWeight", "exAxleCount", "obuVehicleType", "obuSn", "etcCardId",
                      "ExitFeeType", "exOBUVehiclePlate", "exCPUVehiclePlate", "payFee", "middle_type", "gantryPath",
                      "intervalPath", "gantryNum", "gantryTimePath", "gantryTypePath", "gantryTotalFee",
                      "gantryFeePath",
                      "gantryTotalLength", "firstGantryTime", "endGantryTime", "ifProvince", "endsGantryType",
                      "ifHaveCard",
                      "vehiclePlate", "vehiclePlateTotal", "endTime", "vehicleType", "dataType", "proInGantryId",
                      "proInStationId",
                      "proInStationName", "inProTime", "proOutGantryId", "proOutStationId", "proOutStationName",
                      "outProTime"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'varchar', 'datetime', 'int',
                      'int', 'int', 'int', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'int', 'int', 'int', 'int', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'int', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'int', 'varchar', 'int', 'datetime', 'datetime', 'tinyint', 'varchar', 'tinyint', 'varchar',
                      'varchar', 'datetime', 'float', 'varchar', 'varchar', 'varchar', 'varchar', 'datetime', 'varchar',
                      'varchar', 'varchar', 'datetime'])
    # 如果write_type为checkFeature, 则针对稽查特征处理后的数据进行保存
    elif write_type == 'checkFeature':
        db.write_one('checkFeature_data_20220524',
                     ["PASSID", "shortPathFee", "shortPathLength", "shortPath", "shortPathNum", "ifVehiclePlateMatch",
                      "ifVehiclePlateSame", "ifPlateColorSame", "ifOBUPlateSame", "ifETCPlateSame", "ifVehicleTypeSame",
                      "ifOBUTypeLarger", "ifUseShortFee", "ifVeTypeLargerAxle", "gantryNumRate", "ifGantryPathWhole",
                      "gantryPathIntegrity", "gantryPathMatch", "ifGrantryFeeSame", "ifShortFeeSame", "maxOutRangeTime",
                      "ifTimeOutRange", "maxOutRangeGantry", "outRangeTimePath", "outRangeGantryPath",
                      "outRangeSpeedPath", "totalTime", "ifShortTimeAbnormal", "shortOutRangeTime", "pathType"],
                     data,
                     ['varchar', 'int', 'int', 'varchar', 'int', 'tinyint', 'tinyint', 'tinyint', 'tinyint',
                      'tinyint', 'tinyint', 'tinyint', 'tinyint', 'tinyint', 'float', 'tinyint', 'int', 'int',
                      'tinyint', 'tinyint', 'int', 'tinyint', 'varchar', 'varchar', 'varchar', 'varchar', 'int',
                      'tinyint', 'int', 'varchar'])
    # 如果write_type为checkResult, 则针对稽查结果数据进行保存
    elif write_type == 'checkResult':
        db.write_one('checkResult_data_20220524',
                     ["PASSID", "GantryPathAbnormal", "ifVehiclePlateMatchCode", "ifVehiclePlateSameCode",
                      "ifPlateColorSameCode",
                      "ifOBUPlateSameCode", "ifETCPlateSameCode", "ifVehicleTypeSameCode", "ifExAxleLargerCode",
                      "ifOBUTypeLargerCode", "ifPassTimeLarger3DaysCode", "ifVehicleTypeSame_filter",
                      "ifPassTimeAbnormalCode", "ifFeeMatchCode", "ifPathTypeAbnormalCode", "combineCode",
                      "abnormalType", "abnormalCore"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'float'])
    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'checkTotal':
        db.write_one('vehicleCheckTotal_data_20220320',
                     ["PASSID", "enVehiclePlateTotal", "enIdenVehiclePlate", "enVehiclePlate", "enPlateColor",
                      "entryStationID", "entryStationHEX", "entryTime", "enMediaType", "enVehicleType",
                      "enVehicleClass", "entryWeight", "enAxleCount", "exVehiclePlateTotal", "exIdenVehiclePlate",
                      "exVehiclePlate", "exPlateColor", "exitStationID", "exitTime", "exMediaType", "exVehicleType",
                      "exVehicleClass", "exitWeight", "exAxleCount", "obuVehicleType", "obuSn", "etcCardId",
                      "ExitFeeType", "exOBUVehiclePlate", "exCPUVehiclePlate", "payFee", "middle_type", "gantryPath",
                      "intervalPath", "gantryNum", "gantryTimePath", "gantryTypePath", "gantryTotalFee",
                      "gantryTotalLength", "firstGantryTime", "endGantryTime", "ifProvince", "endsGantryType",
                      "vehiclePlate", "vehiclePlateTotal", "endTime", "dataType",

                      "shortPathFee", "shortPathLength", "shortPath", "shortPathNum", "ifVehiclePlateMatch",
                      "ifVehiclePlateSame", "ifPlateColorSame", "ifOBUPlateSame", "ifETCPlateSame", "ifVehicleTypeSame",
                      "ifOBUTypeLarger", "ifUseShortFee", "ifVeTypeLargerAxle", "gantryNumRate", "ifGantryPathWhole",
                      "gantryPathIntegrity", "gantryPathMatch", "ifGrantryFeeSame", "ifShortFeeSame", "maxOutRangeTime",
                      "ifTimeOutRange", "maxOutRangeGantry", "totalTime", "ifShortTimeAbnormal", "shortOutRangeTime",
                      "pathType",

                      "GantryPathAbnormal", "ifVehiclePlateMatchCode", "ifVehiclePlateSameCode", "ifPlateColorSameCode",
                      "ifOBUPlateSameCode", "ifETCPlateSameCode", "ifVehicleTypeSameCode", "ifExAxleLargerCode",
                      "ifOBUTypeLargerCode", "ifPassTimeLarger3DaysCode", "ifVehicleTypeSameCode_filter",
                      "ifPassTimeAbnormalCode", "ifFeeMatchCode", "ifPathTypeAbnormalCode", "combineCode",
                      "abnormalType", "abnormalCore"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'datetime', 'datetime', 'float',
                      'float', 'float', 'float', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'float', 'float', 'float', 'float', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'float', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'float', 'int', 'datetime', 'datetime', 'tinyint', 'varchar', 'varchar', 'varchar', 'datetime',
                      'varchar',

                      'float', 'float', 'varchar', 'float', 'tinyint', 'tinyint', 'tinyint', 'tinyint',
                      'tinyint', 'tinyint', 'tinyint', 'tinyint', 'tinyint', 'float', 'tinyint', 'float', 'float',
                      'tinyint', 'tinyint', 'float', 'tinyint', 'varchar', 'float', 'tinyint', 'float', 'varchar',

                      'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'float'])
    # 如果write_type为checkResult, 则针对稽查结果数据进行保存
    elif write_type == 'tollinterval':
        db.write_one('tollinterval',
                     ["id", "intervalID", "name", "type", "length", "startLat",
                      "startLng", "startStakeNum", "endStakeNum", "endLat", "endLng", "tollRoads", "endTime",
                      "provinceType", "operation", "isLoopCity", "enTollStation", "exTollStation",
                      "entrystation", "exitstation", "tollGrantry", "ownerid", "roadid", "roadidname", "roadtype",
                      "feeKtype", "feeHtype", "status", "Gantrys", "inoutprovince", "HEX", "NOTE", "SORT", "DIRECTION",
                      "BEGINTIME", "VERTICALSECTIONTYPE", "tollstaion"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'int', 'int', 'float', 'float', 'float', 'float', 'float',
                      'float', 'varchar',
                      'datetime', 'int', 'int', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'int',
                      'int',
                      'varchar', 'int', 'int', 'int', 'int', 'varchar', 'int', 'varchar', 'varchar', 'varchar', 'int',
                      'datetime', 'int', 'varchar'])

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'totaldata':
        db.write_one('vehicleCheckTotal_data_20220406',
                     ["PASSID", "enVehiclePlateTotal", "enIdenVehiclePlate", "enVehiclePlate", "enPlateColor",
                      "entryStationID", "entryStationHEX", "entryTime", "enMediaType", "enVehicleType",
                      "enVehicleClass", "entryWeight", "enAxleCount", "exVehiclePlateTotal", "exIdenVehiclePlate",
                      "exVehiclePlate", "exPlateColor", "exitStationID", "exitTime", "exMediaType", "exVehicleType",
                      "exVehicleClass", "exitWeight", "exAxleCount", "obuVehicleType", "obuSn", "etcCardId",
                      "ExitFeeType", "exOBUVehiclePlate", "exCPUVehiclePlate", "payFee", "middle_type", "gantryPath",
                      "intervalPath", "gantryNum", "gantryTimePath", "gantryTypePath", "gantryTotalFee",
                      "gantryFeePath",
                      "gantryTotalLength", "firstGantryTime", "endGantryTime", "ifProvince", "endsGantryType",
                      "ifHaveCard",
                      "vehiclePlate", "vehiclePlateTotal", "endTime", "dataType", "proInGantryId", "proInStationId",
                      "proInStationName", "inProTime", "proOutGantryId", "proOutStationId", "proOutStationName",
                      "outProTime",

                      "shortPathFee", "shortPathLength", "shortPath", "shortPathNum", "ifVehiclePlateMatch",
                      "ifVehiclePlateSame", "ifPlateColorSame", "ifOBUPlateSame", "ifETCPlateSame", "ifVehicleTypeSame",
                      "ifOBUTypeLarger", "ifUseShortFee", "ifVeTypeLargerAxle", "gantryNumRate", "ifGantryPathWhole",
                      "gantryPathIntegrity", "gantryPathMatch", "ifGrantryFeeSame", "ifShortFeeSame", "maxOutRangeTime",
                      "ifTimeOutRange", "maxOutRangeGantry", "outRangeTimePath", "outRangeGantryPath",
                      "outRangeSpeedPath", "totalTime", "ifShortTimeAbnormal", "shortOutRangeTime", "pathType",

                      "GantryPathAbnormal", "ifVehiclePlateMatchCode", "ifVehiclePlateSameCode", "ifPlateColorSameCode",
                      "ifOBUPlateSameCode", "ifETCPlateSameCode", "ifVehicleTypeSameCode", "ifExAxleLargerCode",
                      "ifOBUTypeLargerCode", "ifPassTimeLarger3DaysCode", "ifVehicleTypeSameCode_filter",
                      "ifPassTimeAbnormalCode", "ifFeeMatchCode", "ifPathTypeAbnormalCode", "combineCode",
                      "abnormalType", "abnormalCore",

                      "createType", "checkTime", "groupID", "inspectorID", "reviewerID", "checkResult", "reason",
                      "entranceTollgateName", "entranceLocation", "entranceCity", "entranceProvince",
                      "exitTollgateName", "exitLocation", "exitCity", "exitProvince", "kindType", "entranceTime",
                      "entranceProbability", "exitProbability", "kindWithVehicleProbability", "twoLocationRelevancy",
                      "ifWeightNormal", "ifQuickInOut", "inStationIdPub", "outStationIdPub"],
                     data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'varchar', 'datetime', 'int',
                      'int', 'int', 'int', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'int', 'int', 'int', 'int', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'int', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'int', 'varchar', 'int', 'datetime', 'datetime', 'tinyint', 'varchar', 'tinyint', 'varchar',
                      'varchar', 'datetime', 'varchar', 'varchar', 'varchar', 'varchar', 'datetime', 'varchar',
                      'varchar', 'varchar', 'datetime',

                      'int', 'int', 'varchar', 'int', 'tinyint', 'tinyint', 'tinyint', 'tinyint',
                      'tinyint', 'tinyint', 'tinyint', 'tinyint', 'tinyint', 'float', 'tinyint', 'int', 'int',
                      'tinyint', 'tinyint', 'int', 'tinyint', 'varchar', 'varchar', 'varchar', 'varchar', 'int',
                      'tinyint', 'int', 'varchar',

                      'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'varchar', 'float',

                      "varchar", "datetime", "int", "varchar", "varchar", "tinyint", "int", "varchar", "varchar",
                      "varchar", "varchar", "varchar", "varchar", "varchar", "varchar", "varchar", "datetime",
                      "float", "float", "float", "float", "tinyint", "tinyint", "varchar", "varchar"
                      ])
    # 绿通OD次数原始数据上传数据库
    elif write_type == 'LT_OD':
        LT_OD_table_name = kp.get_parameter_with_keyword('LT_OD_table_name')
        LT_OD_table_features = kp.get_parameter_with_keyword('LT_OD_table_features')
        LT_OD_table_type = kp.get_parameter_with_keyword('LT_OD_table_type')
        db.write_one(LT_OD_table_name, LT_OD_table_features, data, LT_OD_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'inoutTime':
        inout_time_table_name = kp.get_parameter_with_keyword('inout_time_table_name')
        inout_time_table_features = kp.get_parameter_with_keyword('inout_time_table_features')
        inout_time_table_type = kp.get_parameter_with_keyword('inout_time_table_type')
        db.write_one(inout_time_table_name, inout_time_table_features, data, inout_time_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'gantry_OD':
        gantry_OD_table_name = kp.get_parameter_with_keyword('gantry_OD_table_name')
        gantry_OD_table_features = kp.get_parameter_with_keyword('gantry_OD_table_features')
        gantry_OD_table_type = kp.get_parameter_with_keyword('gantry_OD_table_type')
        db.write_one(gantry_OD_table_name, gantry_OD_table_features, data, gantry_OD_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'LT_treated_OD':
        LT_OD_treated_table_name = kp.get_parameter_with_keyword('LT_OD_treated_table_name')
        LT_OD_treated_table_features = kp.get_parameter_with_keyword('LT_OD_treated_table_features')
        LT_OD_treated_table_type = kp.get_parameter_with_keyword('LT_OD_treated_table_type')
        db.write_one(LT_OD_treated_table_name, LT_OD_treated_table_features, data, LT_OD_treated_table_type)

    # portary_LT
    elif write_type == 'portary_LT':
        long_portray_LT_table_name = kp.get_parameter_with_keyword('long_portray_LT_table_name')
        long_portray_LT_table_features = kp.get_parameter_with_keyword('long_portray_LT_table_features')
        long_portray_LT_table_type = kp.get_parameter_with_keyword('long_portray_LT_table_type')
        db.write_one(long_portray_LT_table_name, long_portray_LT_table_features, data, long_portray_LT_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'portary_gantry':
        long_portray_gantry_table_name = kp.get_parameter_with_keyword('long_portray_gantry_table_name')
        long_portray_gantry_table_feature = kp.get_parameter_with_keyword('long_portray_gantry_table_feature')
        long_portray_gantry_table_type = kp.get_parameter_with_keyword('long_portray_gantry_table_type')
        db.write_one(long_portray_gantry_table_name, long_portray_gantry_table_feature, data,
                     long_portray_gantry_table_type)


'''
    创建时间：2022/2/21
    完成时间：2022/2/21
    功能：将输入地址内的数据写入数据库
    修改时间：No.1
'''


def save_data_of_path(write_type, path, path_type='fold'):
    """
    将输入地址内的数据写入数据库
    :param write_type: 写入数据的类型，middle代表为中间数据写入，timePart代表一段时间合并处理的数据写入，checkFeature代表稽查特征处理表写入，checkResult代表稽查结果写入
    :param path: 需要写入的数据所在的地址
    :return:
    """
    if path_type == 'fold':
        paths = dop.path_of_holder_document(path)
    else:
        paths = [path]
    for path in paths:
        save_data = []
        with open(path) as f:
            for i, row in enumerate(f):
                if i > 0:
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    # if row[1] == '020000110101840099321420211119161055':
                    #     print(row)
                    #     continue

                    # 针对绿通的OD数据进行处理
                    if write_type == 'LT_OD':
                        list_ls = [i]
                        list_ls.extend(row)
                        save_data.append(list_ls)
                    if write_type == 'inoutTime':
                        save_data.append(row)
                    if write_type == 'gantry_OD':
                        list_ls = [i]
                        list_ls.extend(row)
                        save_data.append(list_ls)
                    if write_type == 'LT_treated_OD':
                        list_ls = [i]
                        list_ls.extend(row[1:])
                        save_data.append(list_ls)
                    if write_type == 'portary_LT':
                        row[-8] = float(row[-8]) * 100
                        save_data.append(row)
                    if write_type == 'portary_gantry':
                        # save_data.append(row[:-9])
                        save_data.append(row)
                    if write_type == 'middle':
                        save_data.append(row)
                    elif write_type == 'checkFeature':
                        list_ls = [row[1]]
                        list_ls.extend(row[58:])
                        save_data.append(list_ls)
                    elif write_type == 'checkResult':
                        list_ls = [row[1]]
                        list_ls.extend(row[-17:])
                        save_data.append(list_ls)
                    elif write_type == 'loss':
                        list_ls = row[:-9]
                        list_ls.append(row[-1])
                        list_ls.extend(row[-9:-1])
                        save_data.append(list_ls)
                    elif write_type == 'whole':
                        list_ls = row[:-9]
                        list_ls.append(row[20])
                        list_ls.extend(row[-9:])
                        save_data.append(list_ls)
                    elif write_type == 'province':
                        list_ls = row[:-9]
                        if row[20] != '':
                            list_ls.append(row[20])
                        else:
                            list_ls.append(row[9])
                        list_ls.extend(row[-9:])
                        save_data.append(list_ls)
                    elif write_type == 'part':
                        row.append('part')
                        save_data.append(row)
                    elif write_type == 'checkTotal':
                        save_data.append(row[2:])
                    elif write_type == 'tollinterval':
                        list_ls = [i]
                        list_ls.extend(row)
                        save_data.append(list_ls)
                    elif write_type == 'totaldata':
                        save_data.append(row)
        if write_type == 'middle' or write_type == 'checkFeature' or write_type == 'checkResult' \
                or write_type == 'checkTotal' or write_type == 'tollinterval' or write_type == 'totaldata' \
                or write_type == 'LT_OD' or write_type == 'inoutTime' or write_type == 'gantry_OD' \
                or write_type == 'LT_treated_OD' or write_type == 'portary_LT' or write_type == 'portary_gantry':
            save_data_to_Mysql(write_type, save_data)
        else:
            save_data_to_Mysql('timePart', save_data)


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：将输入地址内的数据写入数据库
    修改时间：No.1
'''


def get_data_of_path(write_type, path, path_type='fold'):
    """

    :param write_type:
    :param path:
    :param path_type:
    :return:
    """


'''
    创建时间：2021/10/31
    完成时间：2021/10/31
    修改时间：No.1 2021/11/3，增加了匹配出口表中的总收费金额的代码
            No.2 2021/11/4，分离出门架出口或入口数据缺失的数据，并进行相应牌识路径的匹配和合并，未测试
            No.3 2021/11/10，修改了基础特征，门架和牌识函数的输入参数的处理代码，新增提取过程数据的牌识路径的代码
            No.4 2022/2/24，删除了部分备注代码，将车牌的处理进行的去除
'''


def run(data_gantry, data_gantry_path, data_enter, data_exit):
    """
    特征数据转换过程，最终生成特征中间表
    :param data_exit:
    :param data_enter:
    :param data_gantry: 门架数据
    :param data_gantry_path: 获取门架路径时所需门架数据
    :return: 各特征数据匹配合并后的DataFrame数据
    """
    print('开始进行数据基础特征提取------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
    # 门架统计特征提取
    data_gantry['BIZID'] = data_gantry['BIZID'].fillna(0)
    data_gantry['FEE'] = data_gantry['FEE'].fillna(0)
    basic_of_gantry = basic_information_of_vehicle_new(data_gantry, data_enter, data_exit)

    # data_ls = basic_of_gantry.reset_index()
    # print('No.2:', data_ls[data_ls['PASSID'] == '020000610201630028362520211118093937'].values)

    # 门架路径相关特征提取
    print('开始进行门架路径提取----------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
    passid_list = dp.Get_Feature_Data(basic_of_gantry, 'PASSID', 'PASSID')
    basic_of_gantry = basic_of_gantry.set_index('PASSID')
    data_gantry_path = data_gantry_path.set_index('PASSID')
    data_gantry_path = data_gantry_path.loc[passid_list, :]
    data_gantry_path = data_gantry_path.reset_index()
    path_of_gantry = path_of_gantry_data_new(data_gantry_path)

    # data_ls = path_of_gantry.reset_index()
    # print('No.3:', data_ls[data_ls['PASSID'] == '020000610201630028362520211118093937'].values)

    # # 两数据合并
    concat_basic_path = pd.merge(left=basic_of_gantry, right=path_of_gantry, how='left', left_index=True,
                                 right_index=True)
    # concat_basic_path = concat_basic_path.reset_index()
    # a = concat_basic_path[concat_basic_path['PASSID'] == '016102110123003656021020211119141946']
    # data_ls = concat_basic_path.reset_index()
    # print('No.4:', data_ls[data_ls['PASSID'] == '020000610201630028362520211118093937'].values)

    # # 匹配出口表中的总收费金额,2021/11/3增加
    print('开始进行总收费金额匹配---------', ti.strftime('%Y-%m-%d %H:%M:%S', ti.localtime(ti.time())))
    concat_basic_path['pay_fee'] = concat_basic_path['pay_fee'].fillna(0)
    concat_basic_path['pay_fee'] = concat_basic_path['pay_fee'] / 100
    concat_basic_path['门架费用'] = concat_basic_path['门架费用'] / 100

    return concat_basic_path


# -------------------------------------------绿通特征处理部分----------------------------------------------------------

'''
    创建时间：2021/09/26
    完成时间：2021/09/26
    修改时间：
'''


def traffic_media_transform(data, license_name, time_col, media_type):
    """
    通行媒介数据的转换
    :param data:DataFrame数据
    :param license_name:车牌字段名
    :param time_col:时间字段名
    :param media_type:通行媒介字段名
    :return:
    """
    data, len_change, len_rage, length = dp.Deal_With_Null(data, [license_name, media_type])
    # 得到每辆车的最新使用媒介信息
    data_resent = dp.get_resent_data_of_column(data, time_col, license_name, [license_name, media_type])
    # 得到每辆车使用过的媒介的信息，用于后续计算每辆车使用的媒介数量
    data_count = dp.Get_Feature_Gather_Data(data, [license_name, media_type, media_type], content='count')
    # 更换数值列的列名，避免和索引列名重复
    data_count.columns = ['n']
    data_count = data_count.reset_index()
    # 计算每辆车使用的媒介数量，用于判断是否更换过媒介
    data_c = dp.Get_Feature_Gather_Data(data_count, [license_name, media_type], content='count')
    # 将每辆车的最新媒介和使用媒介数量数据进行合并
    data_all = dp.Merge_Document('left', data_resent, data_c, index=True)
    # 给合并数据进行重命名
    data_all.columns = [license_name, 'Media_recent', 'Media_change']
    # 将每辆车的媒介数量转换为是否更换过媒介的信息
    media_change = data_all['Media_change'].map(lambda x: 1 if x > 1 else 0)
    data_all['Media_change'] = media_change

    return data_all


'''
    创建时间：2021/09/26
    完成时间：2021/09/26
    修改时间：
'''


def time_of_transport_transform(data, license_name, times, compute_time=False):
    """
    将每辆车的每次的上下高速时间，转换为0点到24点的，每个小时的行驶次数，并进行汇总，得到每辆车的行驶时间分布数据
    :param data:DataFrame数据
    :param license_name:车牌字段名
    :param times:时间字段列表，如果compute_time=False，包括上高速时间、行驶用时的字段名称，如果为TRUE，则包含下高速时间和PassID字段名称
    :param compute_time:用于判断是否需要上高速时间的计算，即从PassID中提取上高速时间
    :return:
    """
    # 指定列进行数据的去空值
    times.append(license_name)
    data, len_change, len_rage, length = dp.Deal_With_Null(data, times)
    if not compute_time:
        # 提取出关键字段数据
        data_time_hour = data[[license_name, times[0], times[1]]]

    else:  # 此时compute_time为TRUE，需要将从passID中提取上高速时间，并计算行驶时间
        # 从passID提取出上高速时间并转换为时间类型，并将该列命名为'上高速时间'
        data = dp.Get_Data_To_Othertype(data, times[1], [-14, 1], 'time', '上高速时间')
        # 上高速时间为空的用下高速时间替代
        data[['上高速时间', times[0]]] = data[['上高速时间', times[0]]].fillna(method='bfill', axis=1)
        # 得到行驶用时
        data['行驶用时'] = data[times[0]] - data['上高速时间']
        # 分别将上高速上时间和行驶用时，转换为小时单位
        data['上高速时间_hour'] = data['上高速时间'].map(lambda x: x.hour)
        data['行驶用时_hour'] = data['行驶用时'].map(lambda x: math.ceil(x.seconds / 3600))
        # 提取出关键字段数据
        data_time_hour = data[[license_name, '上高速时间_hour', '行驶用时_hour']]

    # 根据license将times[0]和times[1]列的内容进行汇总，变成list形式
    data_time_hour1 = dp.Get_Feature_Gather_Data(data_time_hour, [license_name, '上高速时间_hour'], content='list')
    data_time_hour2 = dp.Get_Feature_Gather_Data(data_time_hour, [license_name, '行驶用时_hour'], content='list')
    # 以license列内容为关键字进行匹配合并，得到同一辆车的上高速时间list和行驶时长list
    data_time_hour = dp.Merge_Document('outer', data_time_hour1, data_time_hour2, index=True)
    # 通过每辆车的上高速时间list和行驶时长list，得到每辆车的全天24小时的累计行驶分布情况
    data_time_new = dp.time_to_24hours(data_time_hour.index.values, data_time_hour['上高速时间_hour'].values,
                                       data_time_hour['行驶用时_hour'].values)

    return data_time_new


'''
    创建时间：2021/09/27
    完成时间：2021/09/27
    修改时间：
'''


def count_of_transport_transform(data, license_name, time):
    """
    获取每辆车的全年总次数和每个月的总次数
    :param data:原始数据
    :param license_name:车牌所在列名称
    :param time:时间字段列名称
    :return:包含车牌，全年总运次，各月总运次的DataFrame类型数据
    """
    # 获取每辆车的全年总次数
    # 1.通过车牌和时间，进行数据去重，去掉同一时间不同运输品种的数据
    data_user = dp.Get_Feature_Gather_Data(data, [license_name, time, license_name], content='count')
    data_user.columns = ['time']
    data_user = data_user.reset_index()
    # 2.通过车牌，获取全年运输次数
    data_user_count = dp.Get_Feature_Gather_Data(data_user, [license_name, time], content='count')
    data_user_count.columns = ['总次数']
    # 获取每辆车的全年各月的总次数
    # 1.生成新列，只显示年-月，用于筛选出各车辆该月份的所有数据
    data_user['TIME'] = data_user['PASS_TIME'].map(lambda x: str(x.year) + '-' + str(x.month))
    # 2.提取出年-月 这列的所有特征值
    time_list = dp.Get_Feature_Data(data_user, 'TIME', 'TIME')
    # 3.将每个月的所有车辆取出，再根据车牌汇总各车辆运输次数，最后再以车牌为索引进行合并
    for t in time_list:
        data_time = data_user[data_user['TIME'] == t]
        user_counts = dp.Get_Feature_Gather_Data(data_time, [license_name, time], content='count')
        user_counts.columns = [t]
        data_user_count = pd.concat((data_user_count, user_counts), axis=1)
    # 4.给空值赋值为0
    data_all = data_user_count.fillna(0)

    return data_all


'''
    创建时间：2021/09/27
    完成时间：2021/09/27
    修改时间：
'''


def high_frequency_topN_goods_transform(data, license_name, kind, num):
    """
    筛选出每个车辆运输次数最多的前num个品种，并二维化
    :param data:原始数据
    :param license_name:车牌所在字段名
    :param kind:品种所在字段名
    :param num:筛选出前num的品种
    :return:包含车牌-品种数量-运输topN的品种的DataFrame数据
    """
    # 得到各车运输最多的前N个品种
    # 1.得到各车各品种的运输次数
    data_kind = dp.Get_Feature_Gather_Data(data, [license_name, kind, license_name], content='count')
    data_kind.columns = ['次数']
    # 2.以车牌和运输次数排序，得到各车运次从大到小的品种
    kind_count = data_kind.reset_index().sort_values([license_name, '次数'], ascending=False)
    # 3.以车牌过滤，获取前num个品种
    data_kind_1 = kind_count.groupby(license_name).head(num)
    # 获取所有的品种的列表，用于后续生成二维表的列名
    kinds = dp.Get_Feature_Data(data_kind_1, kind, kind)
    # 通过车牌汇总各车辆的品种输运种类数
    data_kind_2 = dp.Get_Feature_Gather_Data(data_kind.reset_index(), [license_name, kind], content='count')
    data_kind_2.columns = ['品种量']
    # 得到各车辆运输的topN的品种的二维表
    # 1.通过车牌列 汇总每辆车的topN品种的list列表，用于后续循环遍历
    data_kind_1 = dp.Get_Feature_Gather_Data(data_kind_1, [license_name, kind], content='list')
    data_kind_list = data_kind_1[kind].values
    # 2.生成长度为车辆数，宽度为品种数的0矩阵，用于每辆车的品种的赋值
    Q = np.zeros((len(data_kind_list), len(kinds)), dtype='int')
    # 3.遍历data_kind_list，得到每辆车的topN品种list
    for i, provin in enumerate(data_kind_list):
        # 4.遍历provin，topN品种中的每一个
        for pro in provin:
            # 5.遍历kinds，得到所有品种，并与topN里的品种对比，如果相同，就在0矩阵的相应位置赋值1
            for j, ki in enumerate(kinds):
                if pro == ki:
                    Q[i][j] = 1
                    break
    # 6.将矩阵转化为DataFrame类型，行索引赋值为车牌列，列索引赋值为品种列
    data_kind_list = pd.DataFrame(Q, index=data_kind_1.index.values, columns=kinds)
    # 通过车牌匹配合并每辆车的topN品种和全年运输品种总数
    data_kind_list = dp.Merge_Document('left', data_kind_list, data_kind_2, index=True)

    return data_kind_list


'''
    创建时间：2021/09/27
    完成时间：2021/09/27
    修改时间：
'''


def high_frequency_all_goods_transform(data, license_name, kind):
    """
    将每辆车的各品种的运输次数的一维表，进行热独处理，转换为二维表
    :param data:原始数据
    :param license_name:
    :param kind:品种所在列名
    :return:
    """
    data, len_change, len_rage, length = dp.Deal_With_Null(data, kind)

    # 2022/04/06
    kind_list = data[kind].values
    license_list = data[license_name].values
    data_list = []
    for i in range(len(kind_list)):
        kind_ = kind_list[i].split('|')
        for j in range(len(kind_)):
            data_list.append([license_list[i], kind_[j]])
    data = pd.DataFrame(data_list, columns=[license_name, kind])

    data_kind = dp.Get_Feature_Gather_Data(data, [license_name, kind, license_name], content='count')
    data_kind.columns = ['次数']
    data_kind = data_kind.reset_index()
    # 将车牌-品种-次数的一维表，转换为二维表，并将转换后的空值赋值为0
    data_kind_unstack = data_kind.set_index([license_name, kind]).unstack().fillna(0)
    # 将二维表的columns进行替换
    # 1.将索引列即车牌列置换出来
    data_kind_unstack = data_kind_unstack.reset_index()
    # 2.获取所有列的名称list
    data_kind_list = data_kind_unstack.columns.values
    # 3.从所有列的名称list中，取出只有车牌和品种的list
    li = []
    for i, kind in enumerate(data_kind_list):
        if i == 0:
            li.append(license_name)
        else:
            li.append(kind[1])
    # 4.将二维表的列名替换为新的list
    data_kind_unstack.columns = li

    # 去掉运输量较少的品种，缩减数据特征，减少数据量
    # 1.获取所有品种名
    # li2 = li[1:]
    # # 循环，将每个品种的所有车辆的运次小于4000的，全部去掉
    # for l in li2:
    #     if data_kind_unstack[data_kind_unstack[l] > 0][l].count() < 4000:
    #         data_kind_unstack = data_kind_unstack.drop([l], axis=1)

    return data_kind_unstack


'''
    创建时间：2021/09/28
    完成时间：2021/09/28
    修改时间：
'''


def high_frequency_topN_province_transform(data, license_name, time, province, num):
    """
    获取各车辆的topN的高频运输省份，并转换为二维表
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param province: 数组，包括入口省列名和出口省列名
    :param num:选择运输次数最多的num个
    :return:
    """
    # 得到各车运输所运输省份的次数数据
    # 1.得到各车各省份运输记录的去重数据
    province_out = dp.Get_Feature_Gather_Data(data, [license_name, time, province[0], license_name], content='count')
    province_in = dp.Get_Feature_Gather_Data(data, [license_name, time, province[1], license_name], content='count')
    province_out.columns = ['次数']
    province_in.columns = ['次数']
    # 2.得到各车各入口省和出口省的运输次数
    province_out_count = dp.Get_Feature_Gather_Data(province_out.reset_index(),
                                                    [license_name, province[0], license_name],
                                                    content='count')
    province_in_count = dp.Get_Feature_Gather_Data(province_in.reset_index(), [license_name, province[1], license_name],
                                                   content='count')
    province_out_count.columns = ['次数']
    province_in_count.columns = ['次数']
    # 3.各车入口省和出口省次数合并
    province_out_count = dp.change_index_name(province_out_count, [license_name, '省份'], new_column=['次数'], column=True)
    province_in_count = dp.change_index_name(province_in_count, [license_name, '省份'], new_column=['次数'], column=True)
    data_province = pd.concat((province_out_count.reset_index(), province_in_count.reset_index()), axis=0)
    data_province = dp.Get_Feature_Gather_Data(data_province, [license_name, '省份', '次数'], content='sum')
    # 得到各车所运输省份的topN
    # 1.给各车的省份运输次数进行排序
    data_province = data_province.reset_index().sort_values([license_name, '次数'], ascending=False)
    # 2.以车牌过滤，获取前num个省份
    data_province_top = data_province.groupby(license_name).head(num)
    # 转换为二维表
    # 1.通过车牌列 汇总每辆车的topN省份的list列表，用于后续循环遍历
    data_province_list = dp.Get_Feature_Gather_Data(data_province_top, [license_name, '省份'], content='list')
    # 2.得到所有车牌的list、所有省份列的去重列表和各车对应的topN省份列表
    province_list = data_province_list.index.values
    province_name = dp.Get_Feature_Data(data_province_top, '省份', '省份')
    data_province_list = data_province_list['省份'].values
    # 3.得到转换后的二维表
    data_province_2D = dp.low_dimension_to_high_dimension(province_list, data_province_list, province_name,
                                                          len(province_list), len(province_name))

    return data_province_2D


'''
    创建时间：2022/03/03
    完成时间：2022/03/03
    修改时间：
'''


def get_all_num_province_transform(data, license_name, time, province):
    """
    获取各车辆的topN的高频运输省份，并转换为二维表
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param province: 数组，包括入口省列名和出口省列名
    :param num:选择运输次数最多的num个
    :return:
    """
    # 得到各车运输所运输省份的次数数据
    # 1.得到各车各省份运输记录的去重数据
    province_out = dp.Get_Feature_Gather_Data(data, [license_name, time, province[0], license_name], content='count')
    province_in = dp.Get_Feature_Gather_Data(data, [license_name, time, province[1], license_name], content='count')
    province_out.columns = ['次数']
    province_in.columns = ['次数']
    # 2.得到各车各入口省和出口省的运输次数
    province_out_count = dp.Get_Feature_Gather_Data(province_out.reset_index(),
                                                    [license_name, province[0], license_name],
                                                    content='count')
    province_in_count = dp.Get_Feature_Gather_Data(province_in.reset_index(), [license_name, province[1], license_name],
                                                   content='count')
    province_out_count.columns = ['次数']
    province_in_count.columns = ['次数']
    # 3.各车入口省和出口省次数合并
    province_out_count = dp.change_index_name(province_out_count, [license_name, '省份'], new_column=['次数'], column=True)
    province_in_count = dp.change_index_name(province_in_count, [license_name, '省份'], new_column=['次数'], column=True)
    data_province = pd.concat((province_out_count.reset_index(), province_in_count.reset_index()), axis=0)
    data_province = dp.Get_Feature_Gather_Data(data_province, [license_name, '省份', '次数'], content='sum')
    # 得到各车所运输省份的topN
    # 1.给各车的省份运输次数进行排序
    data_province = data_province.reset_index().sort_values([license_name, '次数'], ascending=False)
    data_province = data_province.set_index([license_name, '省份']).unstack().fillna(0)
    # 将二维表的columns进行替换
    # 1.将索引列即车牌列置换出来
    data_province = data_province.reset_index()
    # 2.获取所有列的名称list
    data_province_list = data_province.columns.values
    # 3.从所有列的名称list中，取出只有车牌和品种的list
    li = []
    for i, kind in enumerate(data_province_list):
        if i == 0:
            li.append(license_name)
        else:
            li.append(kind[1])
    # 4.将二维表的列名替换为新的list
    data_province.columns = li

    return data_province


'''
    创建时间：2021/09/28
    完成时间：2021/09/28
    修改时间：
'''


def radio_of_transprovincial(data, license_name, time, province):
    """
    计算得到每辆车的跨省比，跨省比=跨省运输次数/总运输次数
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param province: 数组，包括入口省列名和出口省列名
    :return:
    """
    # 得到各车跨省和省内运输的次数数据
    # 1.得到各车各省份运输记录的去重数据
    province_inout = dp.Get_Feature_Gather_Data(data, [license_name, time, province[0], province[1], license_name],
                                                content='count')
    province_inout.columns = ['test']
    province_inout = province_inout.reset_index()
    # 2.分别得到各车辆省内运输总次数province_in_count 和 跨省运输总次数province_un_count
    province_in_count = dp.Get_Feature_Gather_Data(
        province_inout[province_inout[province[0]] == province_inout[province[1]]], [license_name, license_name],
        content='count')
    province_un_count = dp.Get_Feature_Gather_Data(
        province_inout[province_inout[province[0]] != province_inout[province[1]]], [license_name, license_name],
        content='count')
    province_in_count.columns = ['绿通省内次数']
    province_un_count.columns = ['绿通跨省次数']
    # 合并各车省内和跨省的次数数据
    province_all_count = pd.concat((province_in_count, province_un_count), axis=1).fillna(0).reset_index()
    # 添加新的一列，存储跨省比，跨省比=跨省运输次数/总运输次数
    province_all_count['跨省比'] = (province_all_count['绿通跨省次数'] / (
            province_all_count['绿通跨省次数'] + province_all_count['绿通省内次数'])) * 100

    return province_all_count


'''
    创建时间：2021/09/28
    完成时间：2021/09/28
    修改时间：
'''


def type_of_truck_transform(data, license_name, time, types):
    """
    获取各车的车型数据，及是否车型变更过，并转换为二维表
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param types: 车型所在列，因数据库中有两列，该参数为数组，
    :return:
    """
    # 将车型的主副列数据进行合并
    data_truck = dp.multirow_data_supply_get(data, types)
    # 数据中车型数据有两种形式，进行替换，统一成“t1”"t2"这种表现形式
    data_truck_replace = dp.replace_some_row_data(data_truck, types[0], ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0'],
                                                  ['t1', 't2', 't3', 't4', 't5', 't6'])
    # 得到各车所有车型记录的list数据
    # 1.对车、时间和车型进行去重
    truck_type = dp.Get_Feature_Gather_Data(data_truck_replace, [license_name, time, types[0], license_name],
                                            content='count')
    truck_type.columns = ['test']
    truck_type = truck_type.reset_index()
    # 2.获取各车的所有车型记录的list数据
    truck_type_list = dp.Get_Feature_Gather_Data(truck_type, [license_name, types[0]], content='list')
    # 取出各车的所有车型记录的list数据，用于后续循环计算
    type_list = truck_type_list[types[0]].values
    # 判断数据是list形式还是string形式，如果为string形式，转换为list形式
    if type(type_list[0]) == str:
        type_list = dp.string_to_list(truck_type_list[types[0]].values)
    # 循环判断每辆车的车型记录list是否是同一个值，如果是的话就表示该车牌没有换过车型 赋值为1
    weather = []
    for truck in type_list:
        if len(set(truck)) > 1:
            weather.append(1)
        else:
            weather.append(0)
    # 将是否更换车型赋值到新列 'VEHICLE_TYPE_weather'
    truck_type_list['VEHICLE_TYPE_weather'] = weather
    # 返回车型列的各车型名称
    feature_name = dp.Get_Feature_Data(truck_type, types[0], types[0])
    # 将车型一维数据，展开成为二维数据，即各车辆对应各车型的运次
    type_high = dp.low_dimension_to_high_dimension(truck_type_list.index.values, type_list, feature_name,
                                                   len(type_list), len(feature_name), total=True)
    # 合并数据
    truck_type_list = dp.Merge_Document('left', truck_type_list, type_high, index=True).drop([types[0]], axis=1)
    # 计算车型更换率
    weather_list = truck_type_list['VEHICLE_TYPE_weather'].values
    radio_change = []
    for i, wea in enumerate(weather_list):
        if wea == 1:
            sum_value = truck_type_list.iloc[i, 1:].sum()
            max_value = truck_type_list.iloc[i, 1:].max()
            radio_change.append((1 - max_value / sum_value))
        else:
            radio_change.append(0.0)
    truck_type_list['radio_change'] = radio_change

    return truck_type_list


'''
    创建时间：2021/09/28
    完成时间：2021/09/28
    修改时间：
'''


def weight_of_truck_transform(data, license_name, time, weight):
    """
    获取各车的载重数据统计值，即MAX、MIN、AVG、Median、1/4分位值，3/4分位值等
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param weight:载重字段所在列名
    :return:
    """
    data_weight = data[(data['TON'] > 2) & (data['TON'] < 60)]
    weight_mean = dp.Get_Feature_Gather_Data(data_weight, [license_name, time, weight], content='mean')
    weight_mean.columns = ['TON']
    weight_mean = weight_mean.reset_index()
    weight_list = dp.Get_Feature_Gather_Data(weight_mean, [license_name, weight], content='list')
    weight_list = weight_list.reset_index()
    weight_new = dp.statistics_value_get(weight_list, license_name, weight)

    return weight_new


'''
    创建时间：2021/09/28
    完成时间：2021/09/28
    修改时间：
'''


def money_of_truck_transform(data, license_name, time, toll):
    """
    获取各车的免收金额数据统计值，即MAX、MIN、AVG、Median、1/4分位值，3/4分位值等
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param toll: 金额所在列
    :return:
    """
    # data_tolls = dp.filter_data_of_truck(data,license)
    # 去除金额列中空值
    data_tolls = data[data[toll].notnull()]
    # 去除单词免收金额 大于10000 的异常数据
    data_tolls = data_tolls[data_tolls[toll] < 10000]
    # 将车牌和时间重复的数据去掉
    data_tolls = dp.Get_Feature_Gather_Data(data_tolls, [license_name, time, toll], content='mean')
    data_tolls = data_tolls.reset_index()
    # 获取各车的
    data_tolls_list = dp.Get_Feature_Gather_Data(data_tolls, [license_name, toll], content='list')
    data_tolls_list = data_tolls_list.reset_index()
    data_tolls_new = dp.statistics_value_get(data_tolls_list, license_name, toll)

    return data_tolls_new


'''
    创建时间：2021/10/11
    完成时间：2021/10/11
    修改时间：
'''


def time_present(data, license_name, time):
    """
    返回最近一次运输到现在的间隔时间
    :param data: 原始数据
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :return:
    """
    data_time = data[[license_name, time]]
    data_time_sort = dp.get_resent_data_of_column(data_time, time, license_name, [license_name, time])
    # data_time_sort = data_time_sort.reset_index()
    data_time_sort['now'] = datetime.datetime.strptime(ti.strftime("%Y-%m-%d %H:%M:%S", ti.localtime()),
                                                       "%Y-%m-%d %H:%M:%S")
    data_time_sort['space_time'] = data_time_sort['now'] - data_time_sort[time]
    data_time_space = data_time_sort[['PLATE_NUMBER', 'space_time']]
    data_time_space['space_time'] = data_time_space['space_time'].map(lambda x: x.days)

    return data_time_space


'''
    创建时间：2021/10/11
    完成时间：2021/10/11
    修改时间：2021/10/12,内容：补充代码备注
'''


def type_of_coach(data, license_name, time, coach_type):
    """
    将各车辆的厢型数据进行特征转换，用于后续分群画像
    :param data: 原始数据，DataFrame类型
    :param license_name: 车牌所在列名称
    :param time: 时间数据所在列名称
    :param coach_type: 厢型所在列的列名称
    :return:
    """
    # 从数据中提取出所需的三列数据
    data_time = data[[license_name, time, coach_type]]
    # 去掉厢型中为空的数据
    data_time = data_time[data_time[coach_type].notnull()]
    data_time = data_time[data_time[coach_type] != '']
    # 去除车辆和时间的重复数据
    data_counts = dp.Get_Feature_Gather_Data(data_time, [license_name, time, time], content='count')
    # 获取各车辆的总运输次数，本来是用于计算各车辆 应用该厢型的占比情况
    data_counts = dp.Get_Feature_Gather_Data(data_counts, [license_name, time], content='count')
    # 获取
    create_type = dp.Get_Feature_Gather_Data(data_time, [license_name, coach_type, coach_type], content='count')
    create_type.columns = ['次数']
    create_type_list = dp.Get_Feature_Gather_Data(create_type.reset_index(), [license_name, coach_type], content='list')
    create_count_list = dp.Get_Feature_Gather_Data(create_type.reset_index(), [license_name, '次数'], content='list')
    create_all = pd.concat((create_type_list, create_count_list), axis=1)
    create_all = dp.Merge_Document('left', create_all, data_counts, index=True)
    create_all.columns = ['CRATE_TYPE', 'COUNT', 'ALL_COUNT']

    # columns = ['开放式', '敞篷式', '平板式敞篷', '栅栏式敞篷', '封闭式', '罐式', '水箱式', '封闭箱式', '帆布包裹式']
    columns = ['罐式', '平板式敞篷', '栅栏式敞篷', '帆布包裹式', '封闭箱式', '水箱式']
    # type_list = dp.Get_Feature_Data(data_time, type, type)
    data_count = create_all['COUNT'].values
    data_type = create_all['CRATE_TYPE'].values
    # Q = np.zeros((len(data_type), 9), dtype='int')
    Q = np.zeros((len(data_type), 6), dtype='int')
    for i, create in enumerate(data_type):
        if type(create) == str:
            create = create.replace("'", "")
            create = create[1:-1]
            create = create.split(',')
            create = [float(i) for i in create]
        if type(data_count[i]) == str:
            count = data_count[i].replace("'", "")
            count = count[1:-1]
            count = count.split(',')
            count = [int(i) for i in count]
        else:
            count = data_count[i]

        for j, cre in enumerate(create):
            try:
                cre = float(cre)
            except:
                print(cre)
                continue
            # this for ShanXi province LvTong
            # if cre == 1.0:
            #     Q[i][0] = count[j]
            # elif cre == 1.1:
            #     Q[i][1] = count[j]
            # elif cre == 1.2:
            #     Q[i][2] = count[j]
            # elif cre == 1.3:
            #     Q[i][3] = count[j]
            # elif cre == 2.0:
            #     Q[i][4] = count[j]
            # elif cre == 2.1:
            #     Q[i][5] = count[j]
            # elif cre == 2.2:
            #     Q[i][6] = count[j]
            # elif cre == 2.3:
            #     Q[i][7] = count[j]
            # elif cre == 3.1 or cre == 3.0:
            #     Q[i][8] = count[j]

            # this for all province LvTong
            if cre == 1.0:
                Q[i][0] = count[j]
            elif cre == 2.1:
                Q[i][1] = count[j]
            elif cre == 2.2:
                Q[i][2] = count[j]
            elif cre == 3.1:
                Q[i][3] = count[j]
            elif cre == 4.1:
                Q[i][4] = count[j]
            elif cre == 5.1:
                Q[i][5] = count[j]

    # dp.low_dimension_to_high_dimension(create_all,[data_count],[type_list],len(data_count),7,total=True)

    data_create_all = pd.DataFrame(Q, index=create_all.index.values, columns=columns)

    # 计算厢型更换率
    coach_length = data_create_all.shape[0]
    radio_change = []
    for i in range(coach_length):
        sum_value = data_create_all.iloc[i, :].sum()
        max_value = data_create_all.iloc[i, :].max()
        radio_change.append((1 - max_value / sum_value))
    data_create_all['coach_change'] = radio_change

    return data_create_all


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：进行绿通车辆中途异常下站次数的统计
    修改时间：No.1 2022/3/17, modify get the charge of every passid
'''


def get_num_of_abnormal_inout(vehicle_list, startTime_list, endTime_list, enWeight_list, exWeight_list, ifNum=True,
                              passid_list=[], deal_list=[]):
    """
    进行绿通车辆中途异常下站次数的统计
    :param vehicle_list: 车牌号数组
    :param startTime_list: 入口时间数组
    :param endTime_list: 出口时间数组
    :param enWeight_list: 入口重量数组
    :param exWeight_list: 出口重量数组
    :return:
    """
    # 进行遍历，将同一辆车的数据放在同一个车牌下
    vehicle_disc = {}
    if len(deal_list) != 0:
        length = len(deal_list)

        for i in range(length):
            if ifNum:
                try:
                    vehicle_disc[vehicle_list[deal_list[i]]].append(
                        [startTime_list[deal_list[i]], endTime_list[deal_list[i]], enWeight_list[deal_list[i]], exWeight_list[deal_list[i]]])
                except:
                    vehicle_disc[vehicle_list[deal_list[i]]] = [
                        [startTime_list[deal_list[i]], endTime_list[deal_list[i]], enWeight_list[deal_list[i]], exWeight_list[deal_list[i]]]]
            else:
                try:
                    vehicle_disc[vehicle_list[deal_list[i]]].append(
                        [startTime_list[deal_list[i]], endTime_list[deal_list[i]], enWeight_list[deal_list[i]], exWeight_list[deal_list[i]], passid_list[deal_list[i]]])
                except:
                    vehicle_disc[vehicle_list[i]] = [
                        [startTime_list[deal_list[i]], endTime_list[deal_list[i]], enWeight_list[deal_list[i]], exWeight_list[deal_list[i]], passid_list[deal_list[i]]]]
    else:
        for i in range(len(vehicle_list)):
            # if vehicle_list[i] == '冀D5L263_01':
            #     print(vehicle_list[i])
            if ifNum:
                try:
                    vehicle_disc[vehicle_list[i]].append(
                        [startTime_list[i], endTime_list[i], enWeight_list[i], exWeight_list[i]])
                except:
                    vehicle_disc[vehicle_list[i]] = [
                        [startTime_list[i], endTime_list[i], enWeight_list[i], exWeight_list[i]]]
            else:
                try:
                    vehicle_disc[vehicle_list[i]].append(
                        [startTime_list[i], endTime_list[i], enWeight_list[i], exWeight_list[i], passid_list[i]])
                except:
                    vehicle_disc[vehicle_list[i]] = [
                        [startTime_list[i], endTime_list[i], enWeight_list[i], exWeight_list[i], passid_list[i]]]

    #
    passid_value = {}  # 2022/3/17
    # 用于保存每个车牌的异常上下高速次数
    vehicle_abnormal_num = []
    # 循环所有的车牌，计算每一个车牌的异常上下站次数
    for key in vehicle_disc.keys():
        # 用于保存每个车牌的异常次数
        abnormal_num = 0
        # 获取每个车牌的全部数据
        vehicle_data = vehicle_disc[key]
        # 对数据进行排序，根据入口时间进行正序
        vehicle_data = dbf.basic_duplicate_remove(vehicle_data, [0, 0])
        # 获取每一条数据的下一条入口时间
        next_startTime = dbf.get_shift_list_of_list(vehicle_data, 1, index=0, inout=True)
        # 获取每一条数据的下一条入口重量
        next_enWeight = dbf.get_shift_list_of_list(vehicle_data, 1, index=2, inout=True)
        # 遍历每个车牌的全部数据，进行异常上下站次数计算
        for i in range(len(vehicle_data)):
            if i < len(vehicle_data) - 1:
                if vehicle_data[i][3] == '0.0' or next_enWeight[i] == '0.0' or vehicle_data[i][3] == '' or \
                        next_enWeight[i] == '':
                    continue
                # 计算下一次上高速时间与本次下高速的时间间隔
                intervel = datetime.datetime.strptime(str(next_startTime[i]), '%Y-%m-%d %H:%M:%S') - \
                           datetime.datetime.strptime(str(vehicle_data[i][1]), '%Y-%m-%d %H:%M:%S')
                # 将间隔时间转换为小时
                intervel = intervel.total_seconds() / 3600.0
                # 计算下一次上高速重量与本次下高速重量的差值占下高速时的重量
                weight_gap = abs(float(vehicle_data[i][3]) - float(next_enWeight[i])) / float(vehicle_data[i][3])
                # 如果间隔在一小时内，且重量差距不到原重量的2%，即判断为异常上下高速
                if intervel < 1 and weight_gap < 0.05:
                    if ifNum:
                        abnormal_num += 1
                    else:
                        passid_value[vehicle_data[i + 1][4]] = 1

        # 保存车牌和对应的异常次数
        vehicle_abnormal_num.append([key, abnormal_num])

    if not ifNum:
        return passid_value

    vehicle_abnormal_num = pd.DataFrame(vehicle_abnormal_num, columns=['车牌(全)', '异常上下高速次数'])

    return vehicle_abnormal_num


'''
    创建时间：2022/3/3
    完成时间：2022/3/4
    功能：get the cheat num of vehicle
    修改时间：
'''


def get_num_of_cheat(vehicle_list, check_result_list, check_time_list):
    """
    get the cheat num of vehicle
    :param vehicle_list:
    :param check_result_list:
    :return:
    """
    vehicle_num = {}
    vehicle_time = {}
    for i in range(len(vehicle_list)):
        if float(check_result_list[i]) != 1:
            try:
                vehicle_num[vehicle_list[i]] += 1
            except:
                vehicle_num[vehicle_list[i]] = 1

            # 2022/5/4 add,get the lastst time of abnormal
            try:
                if vehicle_time[vehicle_list[i]] < check_time_list[i]:
                    vehicle_time[vehicle_list[i]] = check_time_list[i]
            except:
                vehicle_time[vehicle_list[i]] = check_time_list[i]

    vehicle_num = pd.DataFrame.from_dict(vehicle_num, orient='index', columns=['作弊次数'])
    vehicle_time = pd.DataFrame.from_dict(vehicle_time, orient='index', columns=['作弊时间'])
    vehicle_data = pd.concat((vehicle_num, vehicle_time), axis=1)
    return vehicle_data


'''
    创建时间：2022/3/4
    完成时间：2022/3/7
    功能：获取分析所需的字段数据，并进行不同地点的时间与品种的关联度分析
    修改时间：
'''


def get_relevance_of_time_kind(LT_path):
    """
    获取分析所需的字段数据，并进行不同地点的时间与品种的关联度分析
    :param LT_path:
    :return:
    """
    paths = dop.path_of_holder_document(LT_path)
    data_analysis_out = {}  # 用于存储各入口地的时间和品种信息
    data_analysis_in = {}  # 用于存储各出口地的时间和品种信息
    for path in paths:
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    index_list = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('LT_analysis_feature'))
                else:
                    row = dbf.get_values_of_list(row, index_list)
                    if row[2] != '' and row[5] == '1':
                        # 将时间转换为到月
                        row[2] = row[2][5:-12]
                        try:
                            # 将入口地的时间和品种信息保存到各入口地下
                            data_analysis_out[row[2]].append([row[0], row[4]])
                        except:
                            data_analysis_out[row[2]] = [[row[0], row[4]]]
                    if row[3] != '' and row[5] == '1':
                        row[3] = row[3][5:-12]
                        try:
                            # 将出口地的时间和品种信息保存到各入口地下
                            data_analysis_in[row[3]].append([row[1], row[4]])
                        except:
                            data_analysis_in[row[3]] = [[row[1], row[4]]]

    # 提取不同地点的数据
    result_data_in = []
    result_data_out = []
    for key in data_analysis_in.keys():
        # result_data_in.extend(dbf.basic_apriori(data_analysis_in[key], key_name=key))
        support, confidence_first, confidence_second, upValue, total_num = dbf.process_apriori(
            data_analysis_in[key], remove_order=False, K=2)
        for j in range(len(support)):
            if i == 0 and j == 0:
                result_data_in.append(['月份', 'number', '组合', '支持度', '前置信度', '提升度'])
            result_data_in.append(
                [key, total_num[j], support[j][0], support[j][1], confidence_first[j][1], confidence_second[j][1],
                 upValue[j][1]])

    for key in data_analysis_out.keys():
        # result_data_out.extend(dbf.basic_apriori(data_analysis_out[key], key_name=key))
        support, confidence_first, confidence_second, upValue, total_num = dbf.process_apriori(
            data_analysis_out[key], remove_order=False, K=2)
        for j in range(len(support)):
            if i == 0 and j == 0:
                result_data_out.append(['月份', 'number', '组合', '支持度', '前置信度', '提升度'])
            result_data_out.append(
                [key, total_num[j], support[j][0], support[j][1], confidence_first[j][1], confidence_second[j][1],
                 upValue[j][1]])

    with open('./3.short_data/November/location_relevance_of_intime_kind.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data_in)
    with open('./3.short_data/November/location_relevance_of_outtime_kind.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data_out)


'''
    创建时间：2022/3/8
    完成时间：2022/3/8
    功能：获取分析所需的出入地数据，并进行不同的月份进行两地的关联度分析
    修改时间：
'''


def get_relevance_of_two_place(LT_path):
    """
    获取分析所需的出入地数据，并进行不同的月份（以查验时间为主）进行两地的关联度分析
    :param LT_path:
    :return:
    """
    paths = dop.path_of_holder_document(LT_path)
    data_analysis = {}  # 用于存储各入口地的时间和品种信息
    for path in paths:
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    # 如果是第一次循环，获取绿通分析所有字段在数据中的下标
                    index_list = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('LT_analysis_feature'))
                else:
                    # 根据下标，获取到所对应的值
                    row = dbf.get_values_of_list(row, index_list)
                    # 如果查验时间、入口地和出口地不为空，且数据有效，则进行数据提取
                    if row[3] != '' and row[0] != '' and row[1] != '' and row[5] == '1':
                        # 将时间转换为到月
                        row[3] = row[3][5:-12]
                        try:
                            # 将同一个月的出入口地保存到一起
                            data_analysis[row[3]].append([row[0], row[1]])
                        except:
                            data_analysis[row[3]] = [[row[0], row[1]]]

    # 提取不同地点的数据
    result_data = []
    for i, key in enumerate(data_analysis.keys()):
        support, confidence_first, confidence_second, upValue, total_num = dbf.process_apriori(data_analysis[key],
                                                                                               remove_order=False,
                                                                                               if_inout=True, K=2)
        for j in range(len(support)):
            if i == 0 and j == 0:
                result_data.append(['月份', 'number', '组合', '支持度', '前置信度', '后置信度', '提升度'])
            result_data.append(
                [key, total_num[j], support[j][0], support[j][1], confidence_first[j][1], confidence_second[j][1],
                 upValue[j][1]])

    with open('./3.short_data/November/location_relevance_of_two_place.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)


'''
    创建时间：2022/3/9
    完成时间：2022/3/9
    功能：获取分析所需的品种和车型数据，并进行不同的月份品种和车型的关联度分析
    修改时间：
'''


def get_relevance_of_kind_vehicleType(LT_path):
    """
    获取分析所需的品种和车型数据，并进行不同的月份品种和车型的关联度分析
    :param LT_path:
    :return:
    """
    paths = dop.path_of_holder_document(LT_path)
    # data_analysis = {}  # 用于存储各入口地的时间和品种信息
    data_analysis = []
    for path in paths:
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    # 如果是第一次循环，获取绿通分析所有字段在数据中的下标
                    index_list = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('LT_analysis_feature'))
                else:
                    # 根据下标，获取到所对应的值
                    row = dbf.get_values_of_list(row, index_list)
                    # 如果查验时间、入口地和出口地不为空，且数据有效，则进行数据提取
                    if row[3] != '' and row[4] != '' and row[6] != '' and row[5] == '1':
                        data_analysis.append([row[6], row[7], row[4]])
                        # 将时间转换为到月
                        # row[3] = row[3][:-12]
                        # try:
                        #     # 将同一个月的出入口地保存到一起
                        #     data_analysis[row[3]].append([row[6], row[7], row[4]])
                        # except:
                        #     data_analysis[row[3]] = [[row[6], row[7], row[4]]]

    # 提取不同地点的数据
    result_data = []
    # for i, key in enumerate(data_analysis.keys()):
    support, confidence_first, confidence_second, upValue, total_num = dbf.process_apriori(data_analysis,
                                                                                           remove_order=False, K=3)
    for j in range(len(support)):
        if i == 0 and j == 0:
            # result_data.append(['月份', 'number', '组合', '支持度', '前置信度', '后置信度', '提升度'])
            result_data.append(['number', '组合', '支持度', '前置信度', '后置信度', '提升度'])
        result_data.append([total_num[j], support[j][0], support[j][1], confidence_first[j][1], confidence_second[j][1],
                            upValue[j][1]])

    with open('./3.short_data/November/location_relevance_of_kind_vehicleType_box.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)


'''
    创建时间：2022/3/11
    完成时间：2022/3/11
    功能：根据车型、厢型、品种与重量的关系，分析各车型、厢型、品种情况下的正常重量范围
    修改时间：
'''


def get_normal_scope_weight(LT_path):
    """
    根据车型、厢型、品种与重量的关系，分析各车型、厢型、品种情况下的正常重量范围
    :param LT_path:
    :return:
    """
    paths = dop.path_of_holder_document(LT_path)
    data_analysis = []  # 用于存储车型、品种和载重信息
    disc_data = {}  # 用于不同passid的记录条数
    print('开始记录条数---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for path in paths:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    # 如果是第一次循环，获取绿通分析所有字段在数据中的下标
                    index_list = dbf.get_indexs_of_list(row, kp.get_parameter_with_keyword('LT_weight_feature'))
                else:
                    # 根据下标，获取到所对应的值
                    row = dbf.get_values_of_list(row, index_list)
                    # 如果查验时间、入口地和出口地不为空，且数据有效，则进行数据提取
                    if row[0] != '' and row[1] != '' and row[4] != '' and row[5] != '' and row[6] != '' and row[
                        2] != '' and row[3] == '1':
                        # 将'PASSID', '查验时间', '绿通品种', '查验结果', '车型', '厢型', '重量'等数据保存到一起
                        data_analysis.append(
                            [row[0], row[1], row[2], row[4], row[5], row[6], row[4] + '-' + row[5] + '-' + row[2]])
                        try:
                            disc_data[row[0]] += 1
                        except:
                            disc_data[row[0]] = 1

    print('开始进过滤掉同一次运输多种品种的记录---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 过滤掉同一次运输多种品种的记录
    deal_list = []
    for i in range(len(data_analysis)):
        if disc_data[data_analysis[i][0]] <= 1:
            deal_list.append(i)
    # print('开始进过滤---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # for i in range(len(drop_list)):
    #     data_analysis.pop(drop_list[len(drop_list)-i-1])

    # 将数据按照车型、厢型、品种为key 载重为值的格式，进行数据汇总
    print('开始进行数据汇总---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    disc_data = {}  # 用于保存不同车型、厢型、品种下的载重数据集
    for i in range(len(deal_list)):
        try:
            disc_data[data_analysis[deal_list[i]][6]].append(data_analysis[deal_list[i]][5])
        except:
            disc_data[data_analysis[deal_list[i]][6]] = [data_analysis[deal_list[i]][5]]
    with open('./3.short_data/November/weight_data_of_typeCreateKind_total.pkl', 'wb') as f:
        # writer = csv.writer(f)
        pickle.dump(disc_data, f, pickle.HIGHEST_PROTOCOL)

    # 进行数据的异常值去除,通过四分位距进行
    print('开始进行异常值去除---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for key in disc_data.keys():
        disc_data[key] = dbf.get_list_without_abnormal(disc_data[key], 'IQR')

    with open('./3.short_data/November/weight_data_of_typeCreateKind.pkl', 'wb') as f:
        # writer = csv.writer(f)
        pickle.dump(disc_data, f, pickle.HIGHEST_PROTOCOL)

    # 进行数据概率密度拟合，得到95%置信度的载重范围
    print('开始进行概率密度拟合---------------', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    data_analysis = []
    for key in disc_data.keys():
        # 进行判断，如果数据量小于100个，即跳过，不进行拟合，数据量过小
        if len(disc_data[key]) <= 100:
            continue
        # 2022/3/14 add,
        disc_data[key] = [float(x) / 1000 for x in disc_data[key]]

        start_value, end_value, x_value, y_value = dbf.get_fitter(disc_data[key])

        # save the drow
        plt.figure(figsize=(10, 8), dpi=90)
        plt.plot(x_value, y_value, label=key)
        plt.savefig('./3.short_data/November/fitter_figure_picture/' + key + '_fitter.jpg')
        sns.displot(disc_data[key])
        plt.savefig('./3.short_data/November/fitter_figure_picture/' + key + '_hist.jpg')
        plt.close()

        # 进行该情况下，载重临界值的保存
        data_analysis.append([key, round(start_value, 2), round(end_value, 2)])

    with open('./3.short_data/November/weight_scope_of_typeCreateKind.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_analysis)


'''
    创建时间：2022/6/6
    完成时间：2022/6/6
    功能：将各种异常情况下的车辆车牌和异常发生时间数据进行保存
    修改时间：
'''


def get_vehicle_time_of_abnormal(data_path, feature, out_features):
    """
    将各种异常情况下的车辆车牌和异常发生时间数据进行保存
    :param data_path: 数据地址
    :param feature: 需要处理的字段名称
    :param out_features: 输出的字段名称
    :return:
    """
    features = kp.get_parameter_with_keyword(feature)
    num = 12
    for k, fea in enumerate(features):
        if num != 22:
            num += 1
            continue
        if k != len(features) - 1:
            # continue
            path = data_path[0]
        else:
            path = data_path[1]
        # 获取该特征的待展示值
        feature_value = kp.get_parameter_with_keyword(fea)
        out_features.append(fea)

        for j in range(len(feature_value)):
            value_data = []
            for l in range(len(path)):
                if '20220129' >= path[l][-12:-4] >= '20211009' or path[l][-12:-4] <= '20210929' or path[l][-12:-4] >= '20220208':
                    with open(path[l]) as ff:
                        print(path[l])
                        for i, row in enumerate(ff):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if i == 0:
                                index = dbf.get_indexs_of_list(row, out_features)
                            else:
                                # 如果该字段为异常，则进行保存
                                a = row[index[-1]]
                                if row[index[-1]] == feature_value[j]:
                                    value_data.append([row[index[0]], row[index[1]]])

            # 将历史的各异常情况下的车辆异常时间数据进行保存
            credit_basic_path = kp.get_parameter_with_keyword('credit_data_path')
            print(credit_basic_path + str(num) + '.' + fea + '_' + feature_value[j] + '.csv')
            with open(credit_basic_path + str(num) + '.' + fea + '_' + feature_value[j] + '.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(value_data)
            num += 1


'''
    创建时间：2022/4/22
    完成时间：2022/4/25
    功能：对车辆各画像特征进行时间的衰减处理
    修改时间：
'''


def get_decay_of_abnormal_features(feature_name, out_features, now_time):
    """
    对车辆各画像特征进行时间的衰减处理
    :param out_features: 输出的字段名称
    :param data_path: 数据地址
    :param features: 需要处理的字段名称
    :param now_time: 用于对比所设定的当前时间
    :return:
    """
    total_value_key = {}
    features_name = kp.get_parameter_with_keyword(feature_name)
    k_num = 0
    # 获取所有异常情况的车辆及异常发生时间数据的地址
    credit_basic_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('credit_data_path'), True)
    # 将每一种异常情况的数据转换为key-value形式，并进行时间衰减运算
    for j in range(len(credit_basic_path)):
        # 用于存储某一种异常情况下的车牌和异常时间的对应数据
        value_key = {}
        with open(credit_basic_path[j]) as ff:
            for i, row in enumerate(ff):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                try:
                    value_key[row[0]].append(datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S'))
                except:
                    value_key[row[0]] = [datetime.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')]
        # 生成时间衰减函数
        func = sf.function_all([0.5, 0.5, float(np.pi / 360), np.pi/2], 'sin')

        for i, key in enumerate(value_key.keys()):
            time_list = value_key[key]
            weight_sum = 0
            for jj in range(len(time_list)):
                time_tip = round((now_time - time_list[jj]).total_seconds() / (3600 * 24))
                weight_sum += func.query(time_tip)
            if k_num == 0:
                total_value_key[key] = [weight_sum]
            else:
                try:
                    if len(total_value_key[key]) == k_num:
                        total_value_key[key].append(weight_sum)
                    else:
                        zeros = list(np.zeros(k_num - len(total_value_key[key]), int))
                        total_value_key[key].extend(zeros)
                        total_value_key[key].append(weight_sum)
                except:
                    zeros = list(np.zeros(k_num, int))
                    zeros.append(weight_sum)
                    total_value_key[key] = zeros
        k_num += 1
        # if j > 1:
        #     break

    # 遍历处理后的所有车牌的各特征衰减次数，合并成数组，同时对内容不够长度的数据进行补0
    data_result = []
    time_list = []
    time_list.append(out_features[0])
    time_list.extend(features_name)
    data_result.append(time_list)
    for i, key in enumerate(total_value_key.keys()):
        time_list = [key]
        time_list.extend(total_value_key[key])
        if len(time_list) < len(data_result[0]):
            zeros = list(np.zeros(len(data_result[0]) - len(time_list), int))
            time_list.extend(zeros)
        data_result.append(time_list)

    time_treat_portray_path = kp.get_parameter_with_keyword('time_treat_portray_path')
    with open(time_treat_portray_path, 'w', newline='') as ff:
        writer = csv.writer(ff)
        writer.writerows(data_result)

    # return data_result


'''
    创建时间：2022/4/29
    完成时间：2022/4/29
    功能：get the total weight of each vehicle
    修改时间：
'''


def get_total_weight_of_each_vehicle():
    """
    get the total weight of each vehicle
    :param time_path:
    :param portray_data:
    :param weight_path:
    :return:
    """
    time_path = kp.get_parameter_with_keyword('time_treat_portray_path')
    portray_path = kp.get_parameter_with_keyword('vehicle_portray_total_history')
    weight_path = kp.get_parameter_with_keyword('weight_of_features_path')
    time_data = pd.read_csv(time_path)
    portray_data = pd.read_csv(portray_path)
    time_data = time_data.set_index(['车牌(全)'])
    portray_data = portray_data.set_index(['车牌'])
    portray_data = portray_data[['OBU设备有无更换记录', 'ETC卡有无更换记录', '是否有军车变车种情况', '是否一车多型', '出入口重量不一致频次']]
    time_data = pd.merge(time_data, portray_data, how='left', left_index=True, right_index=True)

    weight_data = dbf.get_dict_from_document(weight_path, ['features', 'weight'], encoding='gbk')
    columns = time_data.columns.values

    for i in range(len(columns)):
        try:
            weight = float(weight_data[columns[i]])
        except:
            weight = kp.get_parameter_with_keyword(columns[i])
        time_data[columns[i]] = time_data[columns[i]] * weight

    time_data['total_weight'] = time_data.sum(axis=1)
    time_data = time_data.sort_values(['total_weight'], ascending=False)

    total_weight_of_features_path = kp.get_parameter_with_keyword('total_weight_of_features_path')
    time_data.to_csv(total_weight_of_features_path)


'''
    创建时间：2022/4/26
    完成时间：2022/4/26
    功能：compute the weight of abnormal features
    修改时间：
'''


def get_weight_of_abnormal_features(data_path, feature):
    """

    :param data_path:
    :param features:
    :return:
    """
    features = kp.get_parameter_with_keyword(feature)
    feature_values = {}
    with open(data_path, encoding='utf-8') as f:
        for j, row in enumerate(f):
            row = row.split(',')  # 将读入的数据通过逗号分开
            row[-1] = row[-1][:-1]
            if j == 0:
                feature_index = dbf.get_indexs_of_list(row, features)
            else:
                for i, fea in enumerate(features):
                    try:
                        feature_values[fea].append([row[feature_index[i]], row[-1]])
                    except:
                        feature_values[fea] = [[row[feature_index[i]], row[-1]]]

    result_data = []
    for i, key in enumerate(feature_values.keys()):
        support, confidence_first, confidence_second, upValue, total_num = dbf.process_apriori(
            feature_values[key], remove_order=False, K=2)
        for j in range(len(support)):
            if i == 0 and j == 0:
                result_data.append(['feature', 'number', '组合', '支持度', '前置信度', 'back置信度', '提升度'])
            result_data.append(
                [key, total_num[j], support[j][0], support[j][1], confidence_first[j][1], confidence_second[j][1],
                 upValue[j][1]])

    weight_of_features_path = kp.get_parameter_with_keyword('weight_of_features_origin')
    with open(weight_of_features_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)


'''
    创建时间：2022/4/26
    完成时间：2022/4/26
    功能：get the white and black vehicle list 
    修改时间：
'''


def get_list_of_normal_and_abnormal():
    """

    :return:
    """
    total_weight_of_features_path = kp.get_parameter_with_keyword('total_weight_of_features_path')
    vehicle_portray_total_history = kp.get_parameter_with_keyword('vehicle_portray_total_history')
    vehicle_portray_total_LT = kp.get_parameter_with_keyword('vehicle_portray_total_LT')
    vehicle_portray_total_gantry = kp.get_parameter_with_keyword('vehicle_portray_total_gantry')

    # read the weight data
    data = pd.read_csv(total_weight_of_features_path)
    data = dp.del_abnormal_vehicle(data, '车牌(全)')
    data_l = data[['车牌(全)', 'total_weight']]
    data_l = data_l.set_index('车牌(全)')
    data = dp.del_abnormal_vehicle(data, '车牌(全)')
    data = data.set_index('车牌(全)')
    start_value, end_value, x_value, y_value = dbf.get_fitter(list(data['total_weight'].values), 0.025, 0.99999999995)
    print(end_value, start_value)
    # start_value, end_value = -11.4041899655689, 42.898828939577854
    abnormal_vehicle_list = data[data['total_weight'] > end_value]
    normal_vehicle_list_1 = data_l[data_l['total_weight'] < start_value]
    print(len(abnormal_vehicle_list))

    # data_LT = pd.read_csv(vehicle_portray_total_LT)
    # data_LT = data_LT.set_index('车牌')
    # data_LT = pd.concat((data_LT, abnormal_vehicle_list), axis=1)
    # abnormal_vehicle_list_1 = data_LT[(data_LT['total_weight'].notnull()) & (data_LT['作弊次数'].notnull())]
    # print(len(abnormal_vehicle_list_1.index.values))
    #
    # data_gantry = pd.read_csv(vehicle_portray_total_gantry)
    # data_gantry = data_gantry.set_index('车牌')
    # data_gantry = pd.concat((data_gantry, abnormal_vehicle_list), axis=1)
    # abnormal_vehicle_list_2 = data_gantry[(data_gantry['total_weight'].notnull()) & (data_gantry['AVG_num_eachOD'].notnull())]
    # print(len(abnormal_vehicle_list_2.index.values))
    #
    # abnormal_vehicle_list = pd.concat((abnormal_vehicle_list_1, abnormal_vehicle_list_2), axis=0)
    # print(len(abnormal_vehicle_list_1))
    # print(len(abnormal_vehicle_list_2))
    # print(len(abnormal_vehicle_list))
    #
    # #
    # data_total = pd.read_csv(vehicle_portray_total_history)
    # data_total = dp.del_abnormal_vehicle(data_total, '车牌')
    # data_total = data_total.set_index('车牌')
    # data_total = pd.concat((data_total, data_l), axis=1)
    # normal_vehicle_list_2 = data_total[data_total['total_weight'].isnull()]
    #
    # # data_total = data_total.drop(['total_weight'], axis=1)
    # # data_total = pd.concat((data_total, abnormal_vehicle_list), axis=1)
    # # abnormal_vehicle_list = data_total[data_total['total_weight'].notnull()]
    #
    # data_total = data_total.drop(['total_weight'], axis=1)
    # data_total = pd.concat((data_total, normal_vehicle_list_1), axis=1)
    # normal_vehicle_list_1 = data_total[data_total['total_weight'].notnull()]
    #
    # normal_vehicle_list_1 = pd.concat((normal_vehicle_list_1, normal_vehicle_list_2), axis=0)
    # normal_vehicle_list_1 = normal_vehicle_list_1[normal_vehicle_list_1['总行驶次数'] > 80]

    normal_vehicle_list_path = kp.get_parameter_with_keyword('normal_vehicle_list_path')
    abnormal_vehicle_list_path = kp.get_parameter_with_keyword('abnormal_vehicle_list_path')

    abnormal_vehicle_list.to_csv(abnormal_vehicle_list_path)
    # normal_vehicle_list_1.to_csv(normal_vehicle_list_path)


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：特征数据转换过程，将1维数据，转换成各车辆的类型数据
    修改时间：
'''


def process_of_LT_portary_middle_data(path_value, treat_type='manyDay'):
    """
        特征数据转换过程，将1维数据，转换成用于分群的类型数据
        :param data: DataFrame类型
        :return: 各特征数据匹配合并后的DataFrame数据
    """
    paths = dop.path_of_holder_document(path_value, True)

    for path in paths:
        data = []
        print(path)
        with open(path, encoding='utf-8') as f:
            for j, row in enumerate(f):
                row = row.split(',')  # 将读入的数据通过逗号分开
                row[-1] = row[-1][:-1]  # 最后一个元素的结尾带"\t"，进行去除
                if j > 0:
                    for i in range(50):
                        if len(row) > 44:
                            row.pop(-1)
                        elif len(row) < 44:
                            print(row)
                        else:
                            row[-1] = int(row[-1])
                            break
                    data.append(row)
        data = pd.DataFrame(data, columns=['车种', '预约开始运输时间', '预约结束运输时间', '预约起点收费站编号', '预约终点收费站编号', '预约用户编号',
                                           '车牌(全)', '查验时间', '车型', '厢型', '车主', '出口车道号', '班组标号', '入口重量', '出口重量', '重量',
                                           '入口站ID', '出口站ID', '验货人员编号', '复核人员编号', 'PASSID', '查验结果', '通行介质', '免收费用',
                                           '不合格原因', '免检标识', '最后修改人', '最后修改时间', '入口收费站', '入口地', '入口市', '入口省',
                                           '出口收费站', '出口地', '出口市', '出口省', '绿通品种', '入口时间', '入口地输出概率', '出口地输入概率',
                                           '车型与品种匹配度', '出入地绿通运输概率', '是否载重异常', '是否异常上下高速'])

        data = data.groupby(['车牌(全)', '查验时间']).head(1)

        # 运输品种(所有的)特征转换
        data_all_4 = high_frequency_all_goods_transform(data, '车牌(全)', '绿通品种')
        data_all_4 = data_all_4.set_index('车牌(全)')

        # 运输的所有省份的次数
        data_all_6 = get_all_num_province_transform(data, '车牌(全)', '查验时间', ['入口省', '出口省'])
        data_all_6 = data_all_6.set_index('车牌(全)')
        #
        # 跨省率计算
        data_all_7 = radio_of_transprovincial(data, '车牌(全)', '查验时间', ['入口省', '出口省'])
        data_all_7 = data_all_7.set_index('车牌(全)')

        # 厢型数据的特征转换
        data_all_12 = type_of_coach(data, '车牌(全)', '查验时间', '厢型')

        data_all_13 = data.groupby(['车牌(全)'])['是否异常上下高速'].sum()

        # get the cheat num
        data_all_14 = get_num_of_cheat(data['车牌(全)'].values, data['查验结果'].values, data['查验时间'].values)

        # 将所有的特征计算及转换结果进行合并
        data_all = dp.Combine_Document(
            (data_all_4, data_all_6, data_all_7, data_all_12, data_all_13, data_all_14), axis=1)
        # 更换index索引名称，与之后的列名统一
        data_all = dp.change_index_name(data_all, '车牌(全)')
        # 将车牌列从索引列提出来，并将合并后为空的值替换为0
        data_all = data_all.reset_index().fillna(0)

        if treat_type == 'oneDay':
            LT_middle_path = kp.get_parameter_with_keyword('LT_middle_one_path')
            save_name = LT_middle_path + path[-14:]
        else:
            LT_middle_path = kp.get_parameter_with_keyword('LT_middle_many_path')
            save_name = LT_middle_path + path[-10:]

        data_all.to_csv(save_name, index=False)


'''
    创建时间：2022/6/13
    完成时间：2022/6/13
    功能: create the credit model of vehicle
    修改时间：
'''


def get_cheat_vehicle_and_time(path, if_duplicate=False):
    data = []
    with open(path) as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if i == 0:
                col_index = dbf.get_indexs_of_list(row, ['车牌(全)', 'outtime', 'ifAB'])
            else:
                if row[col_index[2]] == '1':
                    data.append([row[col_index[0]], row[col_index[1]]])
    credit_basic_path = kp.get_parameter_with_keyword('credit_data_path')
    # drop the duplicated data
    if if_duplicate:
        data = pd.DataFrame(data, columns=['车牌(全)', 'outtime'])
        data = data.sort_values(['outtime'], ascending=False)
        data = data.groupby(['车牌(全)']).head(1)
        data.to_csv(credit_basic_path + '11.是否发生过偷逃费.csv', index=False, header=False)

    else:
        with open(credit_basic_path + '11.是否发生过偷逃费.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)


'''
    创建时间：2022/6/14
    完成时间：2022/6/14
    功能: get the LT cheat vehicle and time
    修改时间：
'''


def get_LT_cheat_vehicle_and_time(paths, if_duplicate=False):
    """

    :param path:
    :param if_duplicate:
    :return:
    """
    for i, path in enumerate(paths):
        data_ls = pd.read_csv(path)
        data_ls = data_ls[['车牌(全)', '查验结果', '查验时间']]
        data_ls = data_ls[data_ls['查验结果'] > 1]
        if i == 0:
            data = data_ls[['车牌(全)', '查验时间']]
        else:
            data = pd.concat((data, data_ls[['车牌(全)', '查验时间']]), axis=0)

    data['车牌(全)'] = data['车牌(全)'].map(lambda x: x.split('_')[0] + '_' + x.split('_')[1][1:])
    credit_basic_path = kp.get_parameter_with_keyword('credit_data_path')
    if if_duplicate:
        data = data.sort_values(['查验时间'], ascending=False)
        data = data.groupby(['车牌(全)']).head(1)
    data.to_csv(credit_basic_path + '10.是否发生过绿通作弊.csv', index=False, header=False)
