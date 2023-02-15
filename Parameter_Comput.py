# coding=gbk
"""
拥堵模型所需关键参数的计算
文档创建时间：2022/7/14
文档修改时间：
"""
import csv
import datetime
import math
import pickle
# import time

import numpy as np
# import pandas as pd
import pandas as pd

import neuralNetwork
import Monte_Carlo_method as mcm
import Data_analysis_2 as da2
import Keyword_and_Parameter as kp
from data import Data_Basic_Function as dbf
import Document_process as dop
from chinese_calendar import is_workday

'''
    创建时间: 2022/7/14
    完成时间: 2022/7/14
    功能: 根据提供的路段门架ID和时间去获取前后门架及收费站的流量数据
'''


def get_all_flow_of_gantry(last_gantry, last_sta, next_gantry, next_sta, start_time, goal_time):
    """
    根据提供的路段门架ID和时间去获取前后门架及收费站的流量数据
    :param goal_time: 预测的时长
    :param start_time: 事件开始时间
    :param last_gantry: 事件路段的上一个门架ID
    :param last_sta: 事件路段的上一个收费站
    :param next_gantry: 事件路段的下一个门架ID
    :param next_sta: 事件路段的下一个收费站
    :return:
    """
    # 获取往前追溯两个小时的日期
    before_start_time = datetime.datetime.strftime(
        datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=(-2)),
        '%Y-%m-%d %H:%M:%S')
    before_start_date = before_start_time[:10]
    # 获取事件发生的日期数据
    start_date = start_time[:10]
    # 获取预测时间点对应的日期
    goal_date = datetime.datetime.strftime(
        datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=(goal_time)),
        '%Y-%m-%d %H:%M:%S')[:10]
    # 获取从事件发生前两个小时到事件发生时刻各门架收费站的全部流量数据
    last_gantry_before_data = []  # 用于存储事件前的上一个门架各车型流量
    last_sta_in_before_data = []  # 用于存储事件前的上一个收费站上站又经过事件路段门架的各车型流量
    last_sta_out_before_data = []  # 用于存储事件前的上一个收费站下站且经过上一个门架的各车型流量

    gantry_type_flow_path = kp.get_parameter_with_keyword('gantry_type_flow_path')
    station_out_flow_path = kp.get_parameter_with_keyword('station_out_flow_path')
    station_in_flow_path = kp.get_parameter_with_keyword('station_in_flow_path')
    if before_start_date == start_date:
        pass
    else:
        with open(before_start_date + start_date + '.csv') as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if row[0] >= before_start_time and row[2] == last_gantry:
                    last_gantry_before_data.append(row)
        with open(before_start_date + start_date + '.csv') as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if row[0] >= before_start_time and row[2] == last_gantry:
                    station_out_flow_path.append(row)
    with open(gantry_type_flow_path + start_date + '.csv') as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]

    next_gantry_before_data = []  # 用于存储事件前的上一个门架各车型流量
    next_sta_in_before_data = []  # 用于存储事件前的上一个门架各车型流量


'''
    创建时间: 2022/8/1
    完成时间: 2022/8/1
    功能: 针对立交位置，统计必须经过某一端的流量数量
'''


def get_flow_num_of_flyover(paths):
    """
    针对立交位置，统计必须经过某一端的流量数量
    :param paths: 地址集
    :return:
    """
    data_result = {}
    # 西宝上游所有门架ID集合
    xi_road_left = ["G003061003000810", "G003061003000820", "G003061003000910", "G003061003000920", "G003061003001010",
                    "G003061003001020", "G003061003001110", "G003061003001120", "G003061003001210",
                    "G003061003001220", "G003061003001310", "G003061003001320", "G003061003001410", "G003061003001420",
                    "G003061003001510", "G003061003001520", "G003061003001610", "G003061003001620", "G003061003001710",
                    "G003061003001720"]
    xi_road_right = ["G003061003000610", "G003061003000620", "G003061003000710", "G003061003000720", "G003061003000510",
                     "G003061003000520"]
    xi_road_down = ["G003061003000310", "G003061003000320", "G003061003000210", "G003061003000220", "G030N61001001510",
                    "G030N61001001520"]
    # 兴平方向上的门架ID
    direction4 = ['G003061003000810']
    # 西吴立交-茂陵方向上的门架ID
    direction1 = ['G030N61001001420']
    # 沣渭互通方向上的门架ID
    direction2 = ['G003061003000320', 'G030N61001001510', 'G003061003000220']
    # 西宝下游方向的门架ID
    direction3 = ['G003061003000720']
    for path in paths:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    col_index = dbf.get_index_of_list(row, ['收费单元路径'])
                else:
                    # 如果下游为茂陵方向
                    if direction1[0] in row[col_index[0]]:
                        # 获取该门架之前的所有门架列表
                        col_list = row[col_index[0]].split('|')
                        index = dbf.get_index_of_list(col_list, direction1[0])
                        col_set = set(col_list[(index - 6):index])
                        # 如果上游门架和西宝左半段路段有交集，说明从兴平过来
                        if len(col_set.intersection(xi_road_left)) > 0:
                            try:
                                data_result['G003061003000820'] += 1
                            except:
                                data_result['G003061003000820'] = 1
                        # 如果上游门架和西宝右半段路段有交集，说明从咸阳西过来
                        elif len(col_set.intersection(xi_road_right)) > 0:
                            try:
                                data_result['G003061003000710'] += 1
                            except:
                                data_result['G003061003000710'] = 1
                        # 如果上游门架和西宝下半段路段有交集，说明从咸阳南过来
                        elif len(col_set.intersection(xi_road_down)) > 0:
                            try:
                                data_result['G003061003000310'] += 1
                            except:
                                data_result['G003061003000310'] = 1
                    # 如果下游为咸阳西方向
                    if direction3[0] in row[col_index[0]]:
                        # 获取该门架之前的所有门架列表
                        col_list = row[col_index[0]].split('|')
                        index = dbf.get_index_of_list(col_list, direction1[0])
                        col_set = set(col_list[(index - 6):index])
                        # 如果上游门架和西宝路段有交集，说明从兴平过来
                        if len(col_set.intersection(xi_road_left)) > 0:
                            try:
                                data_result['G003061003000820'] += 1
                            except:
                                data_result['G003061003000820'] = 1
                        # 如果上游门架和西宝下半段路段有交集，说明从咸阳南过来
                        elif len(col_set.intersection(xi_road_down)) > 0:
                            try:
                                data_result['G003061003000310'] += 1
                            except:
                                data_result['G003061003000310'] = 1

                    # 如果下游为咸阳西方向
                    if direction4[0] in row[col_index[0]]:
                        # 获取该门架之前的所有门架列表
                        col_list = row[col_index[0]].split('|')
                        index = dbf.get_index_of_list(col_list, direction1[0])
                        col_set = set(col_list[(index - 6):index])
                        # 如果上游门架和西宝路段有交集，说明从兴平过来
                        if len(col_set.intersection(xi_road_right)) > 0:
                            try:
                                data_result['G003061003000710'] += 1
                            except:
                                data_result['G003061003000710'] = 1
                        # 如果上游门架和西宝下半段路段有交集，说明从咸阳南过来
                        elif len(col_set.intersection(xi_road_down)) > 0:
                            try:
                                data_result['G003061003000310'] += 1
                            except:
                                data_result['G003061003000310'] = 1

                    # 如果下游为沣渭立交方向
                    for j in range(len(direction2)):
                        if direction2[j] in row[col_index[0]]:
                            # 获取该门架之前的所有门架列表
                            col_list = row[col_index[0]].split('|')
                            index = dbf.get_index_of_list(col_list, direction1[0])
                            col_set = set(col_list[(index - 6):index])
                            # 如果上游门架和西宝路段有交集，说明从兴平过来
                            if len(col_set.intersection(xi_road_left)) > 0:
                                try:
                                    data_result['G003061003000820'] += 1
                                except:
                                    data_result['G003061003000820'] = 1
                            # 如果上游门架和西宝右半段路段有交集，说明从咸阳西过来
                            elif len(col_set.intersection(xi_road_right)) > 0:
                                try:
                                    data_result['G003061003000710'] += 1
                                except:
                                    data_result['G003061003000710'] = 1
                            break
    return data_result


'''
    创建时间: 2022/8/2
    完成时间: 2022/8/2
    功能: 获取相邻门架均有记录的车辆的两门架经过时间、车型
'''


def get_data_of_two_gantry_have(paths, gantry_id, corraletion='and', result_type='list'):
    """
    求取相邻门架均有记录的数量
    :param corraletion:
    :param result_type:
    :param gantry_id: 对象门架ID，查找通过这个门架，同时通过其上下游门架的车辆
    :param paths: 地址集
    :return:
    """
    # 数据保存
    if result_type == 'list':
        data_result = []
    elif result_type == 'dict':
        data_result = {}
    for path in paths:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if corraletion == 'or':
                    if i == 0:
                        col_index = dbf.get_index_of_list(row, ['门架路径', '门架时间串', '出口车型', '入口车型'])
                    else:
                        for gantry in gantry_id:
                            if gantry in row[col_index[0]]:
                                # 获取门架列表
                                gantry_list = row[col_index[0]].split('|')
                                # 获取门架时间列表
                                time_list = row[col_index[1]].split('|')
                                # 获取当前门架的下标
                                gantry_index = dbf.get_index_of_list(gantry_list, [gantry])
                                # 获取当前那门架的经过时间
                                gantry_time = time_list[gantry_index[0]]
                                # 确定车型
                                if row[col_index[2]] != '':
                                    vehicle_type = row[col_index[2]]
                                else:
                                    vehicle_type = row[col_index[3]]
                                # 确定时间所属时段
                                if '5' > gantry_time[-4] >= '0':
                                    gantry_time = gantry_time[:-4] + '0:00'
                                else:
                                    gantry_time = gantry_time[:-4] + '5:00'

                                if result_type == 'list':
                                    data_result.append([gantry, gantry_time, vehicle_type])
                                elif result_type == 'dict':
                                    try:
                                        data_result[gantry + '_' + gantry_time + '_' + vehicle_type] += 1
                                    except:
                                        data_result[gantry + '_' + gantry_time + '_' + vehicle_type] = 1
                elif corraletion == 'and':
                    if len(gantry_id) >= 2:
                        if i == 0:
                            col_index = dbf.get_index_of_list(row, ['门架路径', '门架时间串', '出口车型', '入口车型'])
                        else:
                            if gantry_id[1] in row[col_index[0]] and gantry_id[0] in row[col_index[0]]:
                                # 获取门架列表
                                gantry_list = row[col_index[0]].split('|')
                                # 获取门架时间列表
                                time_list = row[col_index[1]].split('|')
                                # 获取上游门架的下标
                                last_gantry_index = dbf.get_index_of_list(gantry_list, [gantry_id[1]])
                                # 获取当前门架的下标
                                gantry_index = dbf.get_index_of_list(gantry_list, [gantry_id[0]])
                                # 获取上游门架的经过时间
                                last_gantry_time = time_list[last_gantry_index[0]]
                                # 获取当前那门架的经过时间
                                gantry_time = time_list[gantry_index[0]]
                                # 确定车型
                                if row[col_index[2]] != '':
                                    vehicle_type = row[col_index[2]]
                                else:
                                    vehicle_type = row[col_index[3]]
                                # 确定时间所属时段
                                if '5' > gantry_time[-4] >= '0':
                                    gantry_time = gantry_time[:-4] + '0:00'
                                else:
                                    gantry_time = gantry_time[:-4] + '5:00'

                                if '5' > last_gantry_time[-4] >= '0':
                                    last_gantry_time = last_gantry_time[:-4] + '0:00'
                                else:
                                    last_gantry_time = last_gantry_time[:-4] + '5:00'

                                data_result.append(
                                    [gantry_id[0], gantry_id[1], gantry_time, last_gantry_time, vehicle_type])

                            if gantry_id[2] in row[col_index[0]] and gantry_id[0] in row[col_index[0]]:
                                # 获取门架列表
                                gantry_list = row[col_index[0]].split('|')
                                # 获取门架时间列表
                                time_list = row[col_index[1]].split('|')
                                # 获取上游门架的下标
                                next_gantry_index = dbf.get_index_of_list(gantry_list, [gantry_id[2]])
                                # 获取当前门架的下标
                                gantry_index = dbf.get_index_of_list(gantry_list, [gantry_id[0]])
                                # 获取上游门架的经过时间
                                next_gantry_time = time_list[next_gantry_index[0]]
                                # 获取当前那门架的经过时间
                                gantry_time = time_list[gantry_index[0]]
                                # 确定车型
                                if row[col_index[2]] != '':
                                    vehicle_type = row[col_index[2]]
                                else:
                                    vehicle_type = row[col_index[3]]
                                data_result.append(
                                    [gantry_id[0], gantry_id[2], gantry_time, next_gantry_time, vehicle_type])

    # with open('./data_check/flow_check/' + gantry_id[0] + '.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data_result)

    return data_result


'''
    创建时间: 2022/8/3
    完成时间: 2022/8/3
    功能: 根据同时经过两个门架的信息，进行占比统计
'''


def statistic_the_two_gantry_data(data):
    """
    根据同时经过两个门架的信息，进行占比统计
    :param data:
    :return:
    """
    # 用于保存，对象门架的各时段各车型流量
    this_gantry = {}
    other_gantry = {}

    for i in range(len(data)):
        try:
            this_gantry[data[i][0] + '_' + data[i][1] + '_' + data[i][2] + '_' + data[i][4]] += 1
        except:
            this_gantry[data[i][0] + '_' + data[i][1] + '_' + data[i][2] + '_' + data[i][4]] = 1
        try:
            other_gantry[data[i][0] + '_' + data[i][1] + '_' + data[i][2] + '_' + data[i][3] + '_' + data[i][4]] += 1
        except:
            other_gantry[data[i][0] + '_' + data[i][1] + '_' + data[i][2] + '_' + data[i][3] + '_' + data[i][4]] = 1

    for key in other_gantry.keys():
        key_list = key.split('_')
        other_gantry[key_list[0] + '_' + key_list[1] + '_' + key_list[2] + '_' + key_list[3] + '_' + key_list[4]] = \
            other_gantry[key_list[0] + '_' + key_list[1] + '_' + key_list[2] + '_' + key_list[3] + '_' + key_list[4]] / \
            this_gantry[key_list[0] + '_' + key_list[1] + '_' + key_list[2] + '_' + key_list[4]]

    return other_gantry, this_gantry


'''
    创建时间: 2022/8/4
    完成时间: 2022/8/4
    功能: 根据校对门架的车辆数在上下游各时段的占比，进行流量分配
'''


def get_real_flow_pass_gantry(gantry_id):
    """
    根据校对门架的车辆数在上下游各时段的占比，进行流量分配
    :param gantry_id:
    :return:
    """
    # 先获取当前及上下游门架的实际流量
    # 获取标准路径的正向字典数据
    standard_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv', ['ENROADNODEID', 'EXROADNODEID'],
                                               encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取标准路径的反向字典数据
    standard_back_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                                    ['EXROADNODEID', 'ENROADNODEID'],
                                                    encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取标准路径的反向字典数据
    last_station_dict = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv', ['id', 'enTollStation'],
                                                   encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取标准路径的反向字典数据
    next_station_dict = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv', ['id', 'exTollStation'],
                                                   encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取上下游门架ID
    last_gantry = standard_back_dict[gantry_id]
    next_gantry = standard_dict[gantry_id]

    # 获取当前门架ID的流量
    gantry_flow = get_data_of_two_gantry_have('', [gantry_id], 'or', 'dict')
    # 获取上游门架ID的流量
    last_gantry_flow = get_data_of_two_gantry_have('', [last_gantry], 'or', 'dict')
    # 获取下游门架ID的流量
    next_gantry_flow = get_data_of_two_gantry_have('', [next_gantry], 'or', 'dict')

    # 获取上下游门架各各时段各车型数量
    gantry_data_list = get_data_of_two_gantry_have('', [gantry_id, last_gantry, next_gantry], 'and', 'list')
    # 获取上下游各时段各车型占比
    rate_gantry_flow, this_gantry = statistic_the_two_gantry_data(gantry_data_list)

    # 获取上一个收费站的ID
    last_station = last_station_dict[gantry_id]
    # 获取下一个收费站的ID
    next_station = next_station_dict[gantry_id]

    # 获取上一个收费站的流量数据
    last_station_list = da2.get_station_flow('', last_station, gantry_id[:-2], 'dict')

    # 获取下一个收费站的流量数据
    next_station_list = da2.get_station_flow('', next_station, gantry_id[:-2], 'dict')

    for key in gantry_flow.keys():
        key_list = key.split('_')
        origin_num = gantry_flow[key]
        last_station_num = last_station[last_station + '_' + key_list[0]]


'''
    创建时间：2022/8/5
    完成时间：2022/8/5
    功能：将行驶记录的路径和时间串进行拆分，并根据所要求的格式进行返回，各车型各节点之间的时间集合
    修改时间：
'''


def get_list_of_gantry_path(paths, gantry_name, time_name, vehicleInType_name, vehicleOutType_name, return_type='dict'):
    """
    将行驶记录的路径和时间串进行拆分，并根据所要求的格式进行返回，各车型各节点之间的时间集合
    :param vehicleOutType_name: 出口车型所在的字段名称
    :param paths: 地址集
    :param gantry_name: 门架路径所在的字段名称
    :param time_name: 门架时间串所在的字段名称
    :param vehicleInType_name: 入口车型所在的字段名称
    :param return_type: 返回的数据类型，list为数据，dict为字典，save则为本地保存
    :return:
    """
    # 将每一次行驶的门架路径和时间进行拆分
    if return_type == 'dict':
        data_list = {}
    else:
        data_list = []
    for path in paths:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    index = dbf.get_index_of_list(row,
                                                   [gantry_name, time_name, vehicleInType_name, vehicleOutType_name])
                else:
                    if '|' in row[index[0]]:
                        gantry_list = row[index[0]].split('|')  # 拆分门架路径
                    else:
                        continue
                    if '|' in row[index[1]]:
                        time_list = row[index[1]].split('|')  # 拆分门架时间
                    else:
                        continue
                    if row[index[3]] != '':
                        vehicle_type = row[index[3]]
                    else:
                        vehicle_type = row[index[2]]

                    # 循环遍历门架数组
                    for j in range(len(gantry_list)):
                        if j <= len(gantry_list) - 2:
                            # 计算每段路门架之间的间隔时间
                            intervel = round(dbf.compute_intervel_of_two_time(time_list[j], time_list[j + 1], 'minite'),
                                             0)

                            if return_type == 'dict':
                                try:
                                    data_list[gantry_list[j] + '_' + gantry_list[j + 1] + '_' + vehicle_type].append(
                                        intervel)
                                except:
                                    data_list[gantry_list[j] + '_' + gantry_list[j + 1] + '_' + vehicle_type] = [
                                        intervel]

                            else:
                                data_list.append([gantry_list[j], gantry_list[j + 1], vehicle_type, intervel])

    if return_type == 'save':
        with open('./4.data_check/cangestion/gantry_type_time_length_origin_total_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
    else:
        return data_list


'''
    创建时间：2022/8/5
    完成时间：2022/8/5
    功能：获得各车型各节点之间通行时间的各统计值
    修改时间：
'''


def statistic_time_between_gantrys(data_list):
    """
    获得各车型各节点之间通行时间的各统计值
    :param data_list:
    :return:
    """
    # 进行结果数据的保存
    data = [['最小值', '四分之一值', '中位值', '四分之三值', '最大值', '平均值', '众数值', '总数量', 'key']]
    for key in data_list.keys():
        time_list = data_list[key]
        result = dbf.get_statistics_value_of_list(time_list, sortBySort=True)
        result.extend([len(time_list), key])
        data.append(result)

    with open('./4.data_check/cangestion/gantry_type_time_statistic_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


'''
    创建时间：2022/8/5
    完成时间：2022/8/5
    功能：将收费单元记录进行时间补正后
    修改时间：
'''


def get_real_flow_with_time_revise(paths, if_return=False, treat_type='statistic', filter_id=''):
    """
    将收费单元记录进行时间补正
    :param if_return: 选择是否进行结果数据的导出
    :param paths:
    :return:
    """
    # 获取标准路径的反向字典数据
    standard_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                               ['ENROADNODEID', 'EXROADNODEID'],
                                               encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取各路段行驶时间数据
    standard_time_dict = dbf.get_disc_from_document('../Data_Origin/gantry_type_time_statistic_data.csv',
                                                    ['key', '众数值'], encoding='utf-8')
    if treat_type == 'statistic':
        data_result = {}  # 保存各门架的流量值
    else:
        data_result = {}
    for path in paths:

        with open(path) as f:
            if path[-8:-4] == '0421' or path[-8:-4] == '0422':
                continue
            print(path)
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    col_index = dbf.get_index_of_list(row, ['GANTRYID', 'TRANSTIME', 'TOLLINTERVALID', 'VEHICLETYPE',
                                                             'PASSID', 'ENTOLLSTATIONHEX'])
                else:
                    # 将收费单元进行拆分
                    interval_list = row[col_index[2]].split('|')

                    # 如果|存在于收费单元字段中，说明存在补点的情况
                    if filter_id == '':
                        pass
                    else:
                        if set(filter_id).intersection(set(interval_list)):
                            # if filter_id in row[col_index[2]]:
                            pass
                        else:
                            continue

                    # 先将修正时间字段的内容
                    if len(row[col_index[1]]) <= 19:
                        data_time = row[col_index[1]]
                    else:
                        data_time = row[col_index[1]][:19]

                    if '|' in row[col_index[2]]:
                        # 遍历收费单元数组
                        for interval in interval_list:
                            if filter_id == '':
                                pass
                            else:
                                if interval in filter_id:
                                    pass
                                else:
                                    continue

                            # 如果收费单元和该条记录的门架ID前16位一致，则直接记录流量
                            if interval == row[col_index[0]][:16]:
                                # 时间段转换
                                if '5' > data_time[-4] >= '0':
                                    data_time = data_time[:-4] + '0:00'
                                else:
                                    data_time = data_time[:-4] + '5:00'
                                # 生成流量数据
                                if treat_type == 'statistic':
                                    try:
                                        data_result[row[col_index[0]][:16] + '_' + data_time + '_' + str(
                                            float(row[col_index[3]]))] += 1
                                    except:
                                        data_result[row[col_index[0]][:16] + '_' + data_time + '_' + str(
                                            float(row[col_index[3]]))] = 1
                                else:
                                    row[col_index[1]] = data_time
                                    # data_result.append(dbf.get_values_of_list(row, col_index))
                                    data_result[row[col_index[4]]] = [row[col_index[0]], row[col_index[1]],
                                                                      row[col_index[2]], row[col_index[3]],
                                                                      row[col_index[5]]]
                            # 如果收费单元与该条记录的门架ID不一致
                            else:
                                # 进行该单元到门架的时间计算
                                try:
                                    gap = set(standard_dict[interval]).intersection(set(interval_list))
                                except:
                                    continue
                                if gap:
                                    interval_min = gap.pop()
                                else:
                                    # if len(standard_dict[interval]) <= 2:
                                    #     interval_min = min(standard_dict[interval])
                                    # else:
                                    # print((interval_list, row[col_index[0]][:16]))
                                    continue
                                # 如果下一个就是该门架
                                if interval_min == row[col_index[0]][:16]:
                                    # 匹配两个门架之间的行驶时间
                                    try:
                                        go_time = float(standard_time_dict[
                                                            interval + '_' + row[col_index[0]][:16] + '_' + row[
                                                                col_index[3]] + '.0'])
                                    except:
                                        go_time = 5.0
                                    # 计算去掉该时间后的时间段
                                    data_time = datetime.datetime.strftime(
                                        datetime.datetime.strptime(data_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                            minutes=go_time * (- 1)), '%Y-%m-%d %H:%M:%S')
                                    # 时间段转换
                                    if '5' > data_time[-4] >= '0':
                                        data_time = data_time[:-4] + '0:00'
                                    else:
                                        data_time = data_time[:-4] + '5:00'
                                    # 生成流量数据
                                    if treat_type == 'statistic':
                                        try:
                                            data_result[
                                                interval + '_' + data_time + '_' + str(float(row[col_index[3]]))] += 1
                                        except:
                                            data_result[
                                                interval + '_' + data_time + '_' + str(float(row[col_index[3]]))] = 1
                                    else:

                                        row[col_index[1]] = data_time
                                        # data_result.append(dbf.get_values_of_list(row, col_index))
                                        data_result[row[col_index[4]]] = [row[col_index[0]], row[col_index[1]],
                                                                          row[col_index[2]], row[col_index[3]],
                                                                          row[col_index[5]]]
                                # 如果下一个不是该门架
                                else:
                                    # 循环获取与下一个门架之间的耗时，直到下一个门架是该条记录的门架
                                    time_list = []
                                    ii = 0
                                    while True:
                                        # 判断是否下一个门架是改天记录的门架,如果是就进行时间加总，并生成流量数据，否则就一直获取行驶时间
                                        if interval_min == row[col_index[0]][:16]:
                                            try:
                                                go_time = float(standard_time_dict[
                                                                    interval_new + '_' + interval_min + '_' + row[
                                                                        col_index[3]] + '.0'])
                                            except:
                                                go_time = 5.0
                                            go_time = sum(time_list) + go_time
                                            # 计算去掉该时间后的时间段
                                            data_time = datetime.datetime.strftime(datetime.datetime.strptime(data_time,
                                                                                                              '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                                minutes=go_time * (- 1)), '%Y-%m-%d %H:%M:%S')
                                            # 时间段转换
                                            if '5' > data_time[-4] >= '0':
                                                data_time = data_time[:-4] + '0:00'
                                            else:
                                                data_time = data_time[:-4] + '5:00'
                                            # 生成流量数据
                                            if treat_type == 'statistic':
                                                try:
                                                    data_result[interval + '_' + data_time + '_' + str(
                                                        float(row[col_index[3]]))] += 1
                                                except:
                                                    data_result[interval + '_' + data_time + '_' + str(
                                                        float(row[col_index[3]]))] = 1
                                            else:
                                                row[col_index[1]] = data_time
                                                # data_result.append(dbf.get_values_of_list(row, col_index))
                                                data_result[row[col_index[4]]] = [row[col_index[0]], row[col_index[1]],
                                                                                  row[col_index[2]], row[col_index[3]],
                                                                                  row[col_index[5]]]
                                            break
                                        else:
                                            try:
                                                if ii == 0:
                                                    time_list.append(float(standard_time_dict[
                                                                               interval + '_' + interval_min + '_' + str(
                                                                                   float(row[col_index[3]]))]))
                                                    ii += 1
                                                else:
                                                    ii += 1
                                                    time_list.append(float(standard_time_dict[
                                                                               interval_new + '_' + interval_min + '_' +
                                                                               str(float(row[col_index[3]]))]))
                                            except:
                                                time_list.append(5.0)
                                            interval_new = interval_min
                                            try:
                                                gap = set(standard_dict[interval_new]).intersection(set(interval_list))
                                            except:
                                                break
                                            if gap:
                                                interval_min = gap.pop()
                                            else:
                                                break
                                            if ii > 30:
                                                break
                    else:
                        # 时间段转换
                        try:
                            time_ls = data_time[-4]
                        except:
                            continue
                        if '5' > data_time[-4] >= '0':
                            data_time = data_time[:-4] + '0:00'
                        else:
                            data_time = data_time[:-4] + '5:00'
                        # 生成流量数据
                        if treat_type == 'statistic':
                            try:
                                data_result[
                                    row[col_index[0]][:16] + '_' + data_time + '_' + str(float(row[col_index[3]]))] += 1
                            except:
                                data_result[
                                    row[col_index[0]][:16] + '_' + data_time + '_' + str(float(row[col_index[3]]))] = 1
                        else:
                            row[col_index[1]] = data_time
                            # data_result.append(dbf.get_values_of_list(row, col_index))
                            data_result[row[col_index[4]]] = [row[col_index[0]], row[col_index[1]], row[col_index[2]],
                                                              row[col_index[3]], row[col_index[5]]]

    if if_return:
        return data_result
    else:
        # 将字典类型的流量数据，转换为数组类型，并进行保存
        data_result_list = []  # 保存各门架的流量
        for key in data_result.keys():
            key_list = key.split('_')
            key_list.append(data_result[key])
            data_result_list.append(key_list)

        with open('./4.data_check/cangestion/total_gantry_flow_num_data_202205.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_result_list)


'''
    创建时间：2022/8/9
    完成时间：2022/8/9
    功能：验证门架修正后流量结果，西宝各门架分别进行每日流量的对比，各门架各时段修正前后的对比
    修改时间：
'''


def chect_the_flow_data():
    """
    验证门架修正后流量结果，西宝各门架分别进行每日流量的对比，各门架各时段修正前后的对比
    :return:
    """
    # 得到各日期的文件地址
    paths = dop.path_of_holder_document('./1.Gantry_Data/202207/gantry_path/', True)
    # 西宝各门架的数组集
    gantrys = ["G003061003000120", "G003061003000210", "G003061003000110", "G003061003000310", "G003061003000220",
               "G003061003000410", "G003061003000320", "G003061003000510", "G003061003000420", "G003061003000520",
               "G003061003000710", "G003061003000720", "G003061003000810", "G003061003000820", "G003061003001310",
               "G003061003000610", "G003061003000620", "G003061003000910", "G003061003000920", "G003061003001010",
               "G003061003001020", "G003061003001120", "G003061003001110", "G003061003001220", "G003061003001210",
               "G003061003001320"]
    data_revise_dict = {}
    data_result = []
    for path in paths:
        if '20220723' >= path[-12:-4] or path[-12:-4] >= '20220730':
            continue
        # 获取按照收费单元补正后的各门架流量
        # for gantry in gantrys:
        col_name = ['TOLLINTERVALID' for i in range(16)]
        data_interval = dbf.get_result_from_data_with_colValue([path], col_name, gantrys, 'in')
        # 获取修正后的结果数据
        data_revise = get_real_flow_with_time_revise([path], if_return=True)
        for key in data_revise.keys():
            key_gantry = key.split('_')
            if key_gantry[0] in gantrys:
                try:
                    data_revise_dict[key_gantry[0]] += data_revise[key]
                except:
                    data_revise_dict[key_gantry[0]] = data_revise[key]
        for i, gantry in enumerate(gantrys):
            try:
                data_result.append([path[-12:-4], gantry, data_interval[i], data_revise_dict[gantry]])
            except:
                data_result.append([path[-12:-4], gantry, data_interval[i], 0])

    with open('./4.data_check/cangestion/check_flow_num.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_result)


'''
    创建时间：2022/8/10
    完成时间：2022/8/10
    功能：统计西宝各门架各时段修正前后的流量
    修改时间：
'''


def check_gantry_flow_data_after_revise():
    # 得到各日期的文件地址

    # paths = dop.path_of_holder_document('./1.Gantry_Data/20220/gantry_path/', True)
    paths = dop.path_of_holder_document('./1.Gantry_Data/202206/gantry_path/', True)
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202205/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202204/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202203/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202202/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202201/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202112/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202111/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202110/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202109/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202108/gantry_path/', True))
    paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202107/gantry_path/', True))
    middle_path = './2.Middle_Data/202207/main_middledata_'
    gantrys = ["G003061003000120", "G003061003000210", "G003061003000110", "G003061003000310", "G003061003000220",
               "G003061003000410", "G003061003000320", "G003061003000510", "G003061003000420", "G003061003000520",
               "G003061003000710", "G003061003000720", "G003061003000610", "G003061003000620", "G003061003000910",
               "G003061003000920", "G003061003000810", "G003061003000820", "G003061003001310", "G003061003001210",
               "G003061003001320", "G003061003001010", "G003061003001020", "G003061003001110", "G003061003001120",
               "G003061003001410", "G003061003001420", "G003061003001220"]
    path_list = []
    gantry_path_list = []
    for path in paths:
        if '20220730' <= path[-12:-4]:
            path_list.append(path)
            gantry_path_list.append(middle_path + path[-12:-4] + '.csv')
    # 获取修正后各门架各时段的流量
    data_revise = get_real_flow_with_time_revise(path_list, if_return=True)
    # 获取原始各门架各时段的流量
    gantry_flow = get_data_of_two_gantry_have(gantry_path_list, gantrys, 'or', 'dict')

    data_result = []
    # 进行各门架各时段各车型的对比
    for key in data_revise.keys():
        key_list = key.split('_')
        if key_list[0] in gantrys:
            key_list.append(data_revise[key])
            try:
                key_list.append(gantry_flow[key])
            except:
                key_list.append(0)
            data_result.append(key_list)

    with open('./4.data_check/cangestion/flow_data_after_revise.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_result)


'''
    创建时间：2022/9/1
    完成时间：2022/9/1
    功能：根据该路段各车型的速度进行拥堵判断
    关键词：基于速度，拥堵判定
    修改时间：
'''


@dbf.timeIf_wrapper
def charge_congestion_with_speed(data_speed, time_point, data_flow, threshold_data, target='XB', threshold=1,
                                 gantrys=None):
    """
    根据该路段各车型的速度进行拥堵判断
    :param gantrys:
    :param threshold: 判定为拥堵的阈值，默认为100%，即全部速度低于阈值才判定为拥堵
    :param data_flow: 实时获取的当前时段各路段各车型的流量数据，字典类型
    :param target: 进行判定的目标路段，默认为西宝
    :param data_speed: 实时获取的当前时段各路段各车型的速度数据，字典类型
    :param threshold_data: 各路段各车型的速度阈值，字典类型
    :return: 返回收费单元Id对应是否拥堵的字典数据
    """
    # 判断针对哪些路段
    if len(gantrys) == 0:
        if target == 'XB':
            # 获取要进行判定的收费单元集合
            gantrys = kp.get_parameter_with_keyword('XB_gantrys')
    else:
        pass
    # 获取所有车型的集合
    vehicle_type = kp.get_parameter_with_keyword('vehicle_type')

    # 用于记录各路段拥堵状态
    congestion_result = {}
    # 遍历所有西宝的收费单元ID
    for gantry in gantrys:
        # if gantry == 'G003061003000820':
        #     print(1)
        treat_num = 0  # 计算低于阈值的车辆数
        total_num = 0  # 计算参与判断所有车辆数
        # 遍历所有的车型
        for i, v_type in enumerate(vehicle_type):
            # 对每个单元的所有车型车速进行阈值对比
            # 如果无该车型数据，直接跳过
            try:
                this_flow = float(data_flow[gantry][i])
            except:
                continue
            # 如果该类型车辆数较少同时车速为0时，则不参与判断
            a = data_speed[gantry + '_' + time_point][i]
            if this_flow < 5 or data_speed[gantry + '_' + time_point][i] == 0:
                continue
            else:
                # 判断是否低于阈值
                try:
                    treshold_num = float(threshold_data[gantry + '_' + str(float(v_type))])
                except:
                    treshold_num = 30
                if float(data_speed[gantry + '_' + time_point][i]) <= treshold_num:
                    treat_num += this_flow
                else:
                    pass
                total_num += this_flow
        # 如果低阈值车辆的占比达到阈值，该路段记录拥堵
        if total_num != 0 and (treat_num / total_num) >= threshold:
            congestion_result[gantry] = 1
        else:
            congestion_result[gantry] = 0

    return congestion_result


'''
    创建时间：2022/9/6
    完成时间：2022/9/6
    功能：根据该路段前后流量变化曲线进行拥堵判断
    关键词：基于流量变化，曲线相似度对比，拥堵判定
    修改时间：
'''


@dbf.timeIf_wrapper
def charge_congestion_with_flow_curve(old_inout_data_in, old_inout_data_out, new_inout_data, gantry_back_data,
                                      target='XB', gantrys=None):
    """
    根据该路段前后流量变化曲线进行拥堵判断
    :param gantrys:
    :param target:
    :param old_inout_data_in:
    :param old_inout_data_out:
    :param dict new_inout_data:  各门架当前时段的输入输出流量数据字典
    :param dict gantry_back_data: 各门架对应上一门架的关系字典
    :return:
    """
    # 用于记录各路段拥堵状态
    congestion_result = {}
    if len(gantrys) == 0:
        if target == 'XB':
            # 获取要进行判定的收费单元集合
            gantrys = kp.get_parameter_with_keyword('XB_gantrys')
    else:
        pass
    # 遍历字典中所有的门架ID，进行前后门架流量变化对比
    for gantry in gantrys:
        gantry_in = gantry + '_in'
        gantry_out = gantry + '_out'
        # 获得当前门架的各时段流量
        try:
            this_gantry_out = old_inout_data_out[gantry][1:]
        except:
            congestion_result[gantry] = 0
            continue
        try:
            this_gantry_out.append(new_inout_data[gantry_out])
        except:
            this_gantry_out.append(0)
        # 获得上一个门架的各时段流量
        try:
            this_gantry_in = old_inout_data_in[gantry]
        except:
            # 如果没有上一个门架，显示该门架信息并进行下一个门架
            print(gantry_in)
            continue
        # 将前后门架流量数据代入函数，计算量数据的相似度
        dist = dbf.compute_distance_of_two_list('DTW', this_gantry_out, this_gantry_in,
                                                {'dist': 'abs', 'warp': 1, 'w': 2, 's': 0.6})
        # 根据阈值，判断相似度是否异常
        DTW_threshold_data = dbf.get_disc_from_document('./statistic_data/basic_data/DTW_threshold_data.csv',
                                                          ['id', 'num'], encoding='utf-8', key_for_N=False)
        try:  # 如果有新门架，默认为100
            dist_threshold = float(DTW_threshold_data[gantry])  # bug:此处阈值暂时设定为3，后续需要统计合理相似度阈值范围
        except:
            dist_threshold = 100
        if dist_threshold < 10:
            dist_threshold = 40
        if dist > dist_threshold:  # 如果超了阈值，则赋值为1，表明有拥堵情况
            congestion_result[gantry] = 1
        else:  # 如果未超阈值，则赋值为0，表明无拥堵情况
            congestion_result[gantry] = 0

    return congestion_result


'''
    创建时间：2022/9/1
    完成时间：2022/9/1
    功能：判断路段拥堵的等级(根据车道的车辆数量及路段长度判断)
    关键词：拥堵等级
    修改时间：
'''


def charge_congestion_level_by_length(have_data, charge_gantry, distance_data):
    """
    判断路段拥堵的等级(根据车道的车辆增速判断)
    :param dict distance_data: 各门架间的长度字典数据
    :param dict have_data: 各门架间的车辆保有量
    :param list charge_gantry: 需要进行拥堵等级判定的门架集合
    :return:
    """
    level = {}  # 用于保存每个路段的拥堵等级
    # 遍历需要计算的所有路段
    for gantry in charge_gantry:
        # 获取该路段该时刻道路承载量
        have_num = have_data[gantry]
        # 获取该路段长度
        distance = distance_data[gantry]
        # 获取该路段道路数, bug:道路数固定为4
        lane_num = 4
        # 计算每个路段拥堵系数
        rate = have_num / (distance * lane_num)
        # 转换为拥堵等级
        if rate > 1:
            level[gantry] = 5
        elif 1 >= rate > 0.5:
            level[gantry] = 4
        elif 0.5 >= rate > 0.1:
            level[gantry] = 3
        elif 0.1 > rate > 0.01:
            level[gantry] = 2
        else:
            level[gantry] = 1
    return level


'''
    创建时间：2022/9/1
    完成时间：2022/9/1
    功能：计算每个路段的道路承载量
    关键词：承载量
    修改时间：No.1 2022/9/30,修改了输入的流量数据为各车型各路段流量，返回的承载量也从总承载量变更为包括各车型承载量
            No.2 2023/01/28,修改了批注层次，将收费站各方向上下站流量从预测计算替换为数据库读取
'''


@dbf.timeIf_wrapper
def compute_num_of_gantry(flow_data, now_station_flow_vType_dict, last_time_num, gantry_back_data, last_station_dict,
                          this_time, province_gantrys, gantrys=None):
    """
    计算每个路段的道路承载量
    :param province_gantrys: 入省省界门架ID
    :param gantrys: 涉及到的所有门架ID
    :param now_station_flow_vType_dict: 实时计算的收费站当前时刻各方向上驶入驶出流量
    :param flow_out_station_data: 历史数据各时段各收费站各方向上的驶出流量
    :param list flow_in_station_data: 历史数据各时段各收费站各方向上的驶入流量
    :param bool if_history: 判断是针对历史数据还是针对实时数据进行处理
    :param dict last_time_num: 各路段上一个时刻各车型的承载量
    :param str this_time: 该时段的时间信息
    :param dict last_station_dict: 收费单元和上一个收费站对应关系字典
    :param dict gantry_back_data: 收费单元和上一个单元的对应关系字典
    :param dict flow_data: 各路段当前时段的流量数据
    :return:
    """
    # 各车型赋值
    vehicle_types = [['1'], ['2'], ['3'], ['4'], ['11'], ['12'], ['13'], ['14'], ['15'], ['16']]
    # 获取要进行计算的所有收费单元ID
    # if target == 'XB':
    #     gantrys = kp.get_parameter_with_keyword('XB_gantrys')
    # 遍历所有收费单元，分别计算流入流出流量和承载量
    gantry_inout_data = {}  # 用于保存所有单元的流入流出流量
    gantry_have_data = {}  # 用于保存所有单元的承载量
    gantry_have_total_data = {}  # 用于保存所有单元的总承载量
    gantry_inout_total_data = {}  # 用于保存所有单元的总流入流出
    for gantry in gantrys:
        in_num = {}  # 保存该门架下各车型的流量
        out_num = {}  # 保存该门架下各车型的流量
        this_time_num = {}  # 保存该路段下各车型的承载量
        # 1.计算流入量
        # 1)获取上一个门架的ID
        try:
            last_gantrys = gantry_back_data[gantry]
        except:
            continue
        last_gantry = ''
        if len(last_gantrys) > 2:
            continue
        elif len(last_gantrys) == 1:
            pass
        else:
            for i in range(len(last_gantrys)):
                # 获取上上个门架信息
                try:
                    last_last_gantry = gantry_back_data[last_gantrys[i]]
                except:
                    last_gantry = last_gantrys[i]
                    continue
                if last_gantrys[i] not in province_gantrys and len(gantry_back_data[last_gantrys[i]]) == 1:
                    pass
                elif last_gantrys[i] not in gantrys:
                    continue
                else:
                    last_gantry = last_gantrys[i]
            if last_gantry == '':
                continue

        # 2)获取上一个收费站的上站流量 change:2023/01/28
        # 2.1)获取上一个收费站ID
        last_station = last_station_dict[gantry]
        if last_station == '':
            continue
        # # 2.2)如果针对历史数据进行承载量计算
        # if if_history:
        #     station_in_dict = {}
        #     for vType in vehicle_types:
        #         try:
        #             station_in_dict[vType[0]] = flow_in_station_data[
        #                 last_station + '_' + gantry[:-2] + '_' + this_time + '_' + vType[0]]
        #         except:
        #             station_in_dict[vType[0]] = 0
        # else:
        #     station_in_dict = now_station_flow_vType_dict  # changed by 2023/01/28

        # 3)流入量计算
        for i, vType in enumerate(vehicle_types):
            try:
                last_gantry_num = float(flow_data[last_gantry][i])  # changed at 2023/01/28
            except:
                last_gantry_num = 0
            try:
                # changed at 2023/01/30
                last_station_num = float(now_station_flow_vType_dict[last_station + '_1_' + gantry][i])
            except:
                last_station_num = 0
            try:
                # changed at 2023/01/30
                last_station_num_maybe = int(float(now_station_flow_vType_dict[last_station + '_1_'][i]) / 2)
            except:
                last_station_num_maybe = 0
            in_num[vType[0]] = last_gantry_num + last_station_num + last_station_num_maybe

        # 2.计算流出量
        # 1)获取上一个收费站的预测下站流量
        # 1.1)如果针对历史数据进行承载量计算
        # if last_gantry == '':
        #     station_out_dict = {}
        #     for vType in vehicle_types:
        #         station_out_dict[vType[0]] = 0
        # else:
        #     if if_history:
        #         station_out_dict = {}
        #         for vType in vehicle_types:
        #             try:
        #                 station_out_dict[vType[0]] = flow_out_station_data[
        #                     last_station + '_' + last_gantry[:-2] + '_' + this_time + '_' + vType[0]]
        #             except:
        #                 station_out_dict[vType[0]] = 0
        #     else:
        #         station_out_dict = now_station_flow_vType_dict  # changed by 2023/01/28

        # 1)获取本门架的流量
        for i, vType in enumerate(vehicle_types):
            # 1.1)获取本门架的流量
            try:
                last_gantry_num = float(flow_data[gantry][i])  # changed at 2023/01/28
            except:
                last_gantry_num = 0
            # 1.2)获取上一个收费站的下站流量
            try:
                # changed by 2023/01/30
                last_station_num = float(now_station_flow_vType_dict[last_station + '_2_' + last_gantry][i])
            except:
                last_station_num = 0
            # 1.3)获取上一个收费站的补入下站流量
            try:
                # changed by 2023/01/30
                if last_gantry != '':
                    last_station_num_maybe = int(float(now_station_flow_vType_dict[last_station + '_2_'][i]) / 2)
                else:
                    last_station_num_maybe = 0
            except:
                last_station_num_maybe = 0
            # 1.4)流出总量计算
            out_num[vType[0]] = last_gantry_num + last_station_num + last_station_num_maybe
            # 1.5)获取之前该路段之前的车辆保有量
            try:
                last_num = float(last_time_num[gantry][i])
            except:
                last_num = 0
            # 1.6)进行该路段的各车型承载量计算
            # gantry_have_data[gantry + '_' + vType[0]] = last_num + in_num[vType[0]] - out_num[vType[0]]
            try:
                gantry_have_data[gantry + '_' + this_time].append(last_num + in_num[vType[0]] - out_num[vType[0]])
            except:
                gantry_have_data[gantry + '_' + this_time] = [last_num + in_num[vType[0]] - out_num[vType[0]]]
            # 1.7)保存各路段的总承载量
            try:
                gantry_have_total_data[gantry] += last_num + in_num[vType[0]] - out_num[vType[0]]
            except:
                gantry_have_total_data[gantry] = last_num + in_num[vType[0]] - out_num[vType[0]]
            # 1.8)进行该路段各车型输入量的计算
            try:
                gantry_inout_data[gantry + '_in'].append(in_num[vType[0]])
            except:
                gantry_inout_data[gantry + '_in'] = [in_num[vType[0]]]
            # 1.9)进行该路段各车型输出量的计算
            try:
                gantry_inout_data[gantry + '_out'].append(out_num[vType[0]])
            except:
                gantry_inout_data[gantry + '_out'] = [out_num[vType[0]]]
            # 1.10)进行该路段总输入量的计算
            try:
                gantry_inout_total_data[gantry + '_in'] += in_num[vType[0]]
            except:
                gantry_inout_total_data[gantry + '_in'] = in_num[vType[0]]
            # 1.11)进行该路段总输出量的计算
            try:
                gantry_inout_total_data[gantry + '_out'] += out_num[vType[0]]
            except:
                gantry_inout_total_data[gantry + '_out'] = out_num[vType[0]]

    return gantry_have_data, gantry_inout_data, gantry_have_total_data, gantry_inout_total_data


'''
    创建时间：2022/9/9
    完成时间：2022/9/9
    功能：批量计算历史各门架的输入输出流量DTW系数
    关键词：输入输出流量、DTW系数、批量处理
    修改时间：
'''


def compute_DTW_of_gantrys(start_time, end_time):
    """
    批量计算历史各门架的输入输出流量DTW系数
    :param start_time:
    :param end_time:
    :return:
    """
    start_time = start_time + ' 00:00:00'  # 获取开始时刻
    end_time = end_time + ' 00:00:00'  # 获取截止时刻
    # 读取历史输入输出数据，并分配到各门架上
    in_data_dict = {}  # 存放各门架的输入流量数据
    out_data_dict = {}  # 存放各门架的输出流量数据
    have_num_dict = {}  # 存放各门架的承载量数据
    time_data_dict = {}  # 存放各门架的记录时间数据
    with open('./4.statistic_data/basic_data/820_have.csv') as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if i == 0:
                row[0] = row[0][1:]
            if row[1] < start_time or row[1] > end_time:
                continue
            # 由于该数据没有字段名，直接运算
            try:
                in_data_dict[row[0]].append(float(row[2]))
            except:
                in_data_dict[row[0]] = [float(row[2])]
            try:
                out_data_dict[row[0]].append(float(row[3]))
            except:
                out_data_dict[row[0]] = [float(row[3])]
            try:
                have_num_dict[row[0]].append(float(row[4]))
            except:
                have_num_dict[row[0]] = [float(row[4])]
            try:
                time_data_dict[row[0]].append(row[1])
            except:
                time_data_dict[row[0]] = [row[1]]
    # 将输入输出承载量等根据时间进行排序
    # 遍历所有的门架ID，进行DTW系数计算
    result_data = []  # 用于保存各门架各时刻输入输出相似度的数据
    for key in in_data_dict.keys():
        compute_ls1 = []  # 存放每个时刻临时的计算数据——输入数据
        compute_ls2 = []  # 存放每个时刻临时的计算数据——输出数据
        # 遍历该门架的所有输入输出数据，计算DTW
        for i in range(len(in_data_dict[key])):
            if i == 0:
                compute_ls1.append(in_data_dict[key][i])
            else:
                compute_ls1.append(in_data_dict[key][i])
                compute_ls2.append(out_data_dict[key][i])
            # 判断输入输出数据是否达到了13个，如果达到了，则进行相似度计算
            if len(compute_ls2) < 13:
                continue
            else:
                # 基于DTW模型的两数据
                dist = dbf.compute_distance_of_two_list('DTW', compute_ls1[:-1], compute_ls2,
                                                        {'dist': 'abs', 'warp': 1, 'w': 2, 's': 0.6})
                sum_value = sum([compute_ls1[i - 3] - compute_ls2[i - 3] for i in range(3)])
                result_data.append([key, time_data_dict[key][i], have_num_dict[key][i], in_data_dict[key][i],
                                    out_data_dict[key][i], dist, sum_value])
                compute_ls1.pop(0)
                compute_ls2.pop(0)
    # 存储各门架相似度结果数据
    save_name = './4.statistic_data/basic_data/'
    with open(save_name + 'DTW_num_data_820.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)


'''
    创建时间：2022/9/9
    完成时间：2022/9/9
    功能：对各门架的DTW拟合系数进行分布拟合，并获得规定置信度之外的阈值DTW系数
    关键词：分布拟合、DTW系数阈值
    修改时间：
'''


def get_DTW_threshold_of_gantrys(threshold):
    """
    对各门架的DTW拟合系数进行分布拟合，并获得规定置信度之外的阈值DTW系数
    :param float threshold: 拥堵发生的概率
    :return:
    """
    # 获取每个门架的DTW系数
    DTW_dict = {}  # 存放各门架的记录时间数据
    with open('./4.statistic_data/basic_data/DTW_num_data.csv') as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            # 由于该数据没有字段名，直接运算
            if float(row[6]) < 0:
                continue
            try:
                DTW_dict[row[0]].append(round(float(row[5]), 1))
            except:
                DTW_dict[row[0]] = [round(float(row[5]), 1)]
    # 遍历每个门架，计算每个门架DTW系数的阈值
    result = []  # 保存DTW阈值结果
    for key in DTW_dict.keys():
        print(key)
        start_value, end_value, x_value, y_value = dbf.get_fitter(DTW_dict[key], threshold, (1 - threshold))
        result.append([key, end_value])
    # 保存每个门架的DTW阈值
    save_name = './4.statistic_data/basic_data/'
    with open(save_name + 'DTW_threshold_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)


'''
    创建时间：2022/9/9
    完成时间：2022/9/9
    功能：根据DTW阈值获取各门架超过DTW的承载量和时间点
    关键词：承载量阈值、DTW阈值
    修改时间：
'''


def get_have_threshold_of_gantrys():
    """"""
    # 获取各门架的DTW阈值
    DTW_threshold_dict = dbf.get_disc_from_document('./4.statistic_data/basic_data/DTW_threshold_data.csv',
                                                    [0, 1], encoding='gbk', key_for_N=False, ifIndex=False)
    # 遍历所有承载量数据，分别将每个门架超过其DTW阈值的承载量记录下来
    result = []  # 用于记录超过DTW阈值的门架和承载量信息
    with open('./4.statistic_data/basic_data/DTW_num_data.csv') as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if float(row[5]) > float(DTW_threshold_dict[row[0]]) and float(row[6]) > 2:
                result.append([row[0], row[1], row[2]])
    # 保存
    save_name = './4.statistic_data/basic_data/'
    with open(save_name + 'have_threshold_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)


def compute_XB_gantry_num():
    """

    :return:
    """
    num = 0
    with open('./Data_Origin/tollinterval.csv') as f:
        for j, row in enumerate(f):
            row = row.split(',')
            if j == 0:
                index = dbf.get_index_of_list(row, ['id'])
            if row[index[0]] == 'G0030610030':
                num += 1
    return num


'''
    创建时间：2022/9/12
    完成时间：2022/9/14
    功能：进行所有门架当前承载量与阈值的对比，如果超出阈值同时半个小时候后仍超出阈值及判定为预计拥堵
    关键词：承载量阈值、拥堵预判，半小时流量预测
    修改时间：
'''


@dbf.timeIf_wrapper
def charge_congestion_by_haveNum(now_have_num, this_time, basic_data, station_flow_history_dict_in,
                                 station_flow_history_dict_out, gantry_back_data,
                                 last_station_dict, province_gantrys, gantrys=[]):
    """
    进行所有门架当前承载量与阈值的对比，如果超出阈值同时半个小时候后仍超出阈值及判定为预计拥堵
    :param station_flow_history_dict_in:
    :param list basic_data: 包括当前时刻及之前48个时刻的流量数据
    :param str this_time: 当前时刻
    :param dict now_have_num: 当前各门架承载量字典
    :return:
    """
    congestion_charge = {}  # 保存各门架的是否拥堵结果，其中0为无拥堵，1为预计拥堵，2为通行缓慢
    # 获取各门架的承载量风险阈值
    have_threshold_dict = dbf.get_disc_from_document('./statistic_data/basic_data/have_threshold_data.csv',
                                                     [0, 1], encoding='gbk', key_for_N=False, ifIndex=False,
                                                     ifNoCol=True)
    if len(gantrys) == 0:
        gantrys = kp.get_parameter_with_keyword('XB_gantrys')
    # 对比各门架的当前承载量和风险阈值
    out_data = []  # 用于保存超出阈值的门架ID
    for gantry in gantrys:
        try:
            now_have = now_have_num[gantry]
        except:
            now_have = 0
        try:
            threshold_have = have_threshold_dict[gantry]
        except:
            threshold_have = 400
        if now_have >= (float(threshold_have) * 1.5):
            congestion_charge[gantry] = 2
        elif now_have >= float(threshold_have):
            out_data.append(gantry)
        else:
            congestion_charge[gantry] = 0

    # 遍历所有超出阈值的门架ID，预测其未来半小时的流量及承载量，如果超出阈值的50%及判定为预计拥堵，bug：拥堵预判的百分比固定为了20%
    for i in range(len(out_data)):
        # 判断有没有上一个收费站，如果没有判定为立交下游路段，不进行拥堵判定(直接判定为不拥堵)
        try:
            last_station = last_station_dict[out_data[i]]
            if last_station is None:
                congestion_charge[out_data[i]] = 0
                dop.save_data_of_important([[out_data[i], this_time, '无上一个收费站']], './error/loss_data.csv', 'check_save',
                                           [0], [1, 2])
                continue
        except:
            congestion_charge[out_data[i]] = 0
            dop.save_data_of_important([[out_data[i], this_time, '无上一个收费站']], './error/loss_data.csv', 'check_save',
                                       [0], [1, 2])
            continue

        # 获取上个门架信息，如果上一个门架不存在，判定为省界门架，直接判定为不拥堵并跳过
        try:
            last_gantrys = gantry_back_data[out_data[i]]
        except:
            congestion_charge[out_data[i]] = 0
            dop.save_data_of_important([[out_data[i], this_time, '无上一个门架']], './error/loss_data.csv', 'check_save',
                                       [0], [1, 2])
            continue
        for j in range(len(last_gantrys)):
            # 获取上上个门架信息
            try:
                last_last_gantry = gantry_back_data[last_gantrys[j]]
            except:
                last_gantry = last_gantrys[j]
                continue
            if last_gantrys[j] not in province_gantrys and len(gantry_back_data[last_gantrys[j]]) == 1:
                pass
            elif last_gantrys[j] not in gantrys:
                continue
            else:
                last_gantry = last_gantrys[j]
        # 获取未来半小时上一个门架的流量预测
        last_gantry_flow = compute_future_flow_of_gantry_station(6, last_gantry, this_time,
                                                                 basic_data, 'gantry')
        # 获取未来半小时当前门架的流量预测
        this_gantry_flow = compute_future_flow_of_gantry_station(6, out_data[i], this_time,
                                                                 basic_data, 'gantry')

        # 获取未来半小时收费站输入的流量预测
        station_in_flow = compute_future_flow_of_gantry_station(6, last_station + '_' + out_data[i][:-2],
                                                                this_time, station_flow_history_dict_in, 'station_in')
        # 获取未来半小时收费站输出的流量预测
        station_out_flow = compute_future_flow_of_gantry_station(6, last_station + '_' + last_gantry[:-2],
                                                                 this_time, station_flow_history_dict_out,
                                                                 'station_out')

        # 判断是否为空数组，并保存异常数据
        if len(last_gantry_flow) == 0:
            dop.save_data_of_important([[last_gantry, this_time, '预测失败']], './error/loss_data.csv', 'check_save', [0], [1, 2])
        if len(this_gantry_flow) == 0:
            dop.save_data_of_important([[out_data[i], this_time, '预测失败']], './error/loss_data.csv', 'check_save', [0], [1, 2])
        if len(station_in_flow) == 0:
            dop.save_data_of_important([[last_station_dict[out_data[i]] + '_' + out_data[i][:-2], this_time, '预测失败']],
                                       './error/loss_data.csv', 'check_save', [0], [1, 2])
        if len(station_out_flow) == 0:
            dop.save_data_of_important([[last_station_dict[out_data[i]] + '_' + last_gantry[:-2], this_time, '预测失败']],
                                       './error/loss_data.csv', 'check_save', [0], [1, 2])
        # 计算未来半个小时后的承载量
        try:
            future_have_num = now_have_num[out_data[i]] + dbf.compute_dict_by_group(last_gantry_flow, [], 'total_sum',
                                                                                    '') + dbf.compute_dict_by_group(
                station_in_flow, [], 'total_sum', '') - \
                              dbf.compute_dict_by_group(this_gantry_flow, [], 'total_sum',
                                                        '') - dbf.compute_dict_by_group(station_out_flow, [],
                                                                                        'total_sum', '')
        except:
            future_have_num = 0

        # 判断是否超出阈值
        # 判定是否have_threshold文件里不存在该ID
        try:
            have_threshold = float(have_threshold_dict[out_data[i]]) * 1.5
        except:
            dop.save_data_of_important([[out_data[i], this_time, 'have_threshold缺失']],
                                       './error/loss_data.csv', 'check_save', [0], [1, 2])
            congestion_charge[out_data[i]] = 0
            continue
        if future_have_num >= have_threshold:
            congestion_charge[out_data[i]] = 1
        else:
            congestion_charge[out_data[i]] = 0

    # 返回判定结果
    return congestion_charge


def get_feature_of_flow_calculate(path, treat_type='gantry', if_data=False):
    """
    提取出各门架各时段的流量训练特征数据
    :param treat_type: 处理对象类型，gantry为门架，station为收费站
    :param if_data: 判断是否是针对数据的处理，FALSE表示是对地址文件进行处理，TRUE表示输入的path为关键参数数据
    :param path: 修正后各门架各车型流量数据的地址
    :return:
    """
    # 创建假日时间字典
    holiday = {'20210919': [3, 1], '20210920': [3, 2], '20210921': [3, 3], '20211001': [7, 1], '20211002': [7, 2],
               '20211003': [7, 2], '20211004': [7, 2], '20211005': [7, 2], '20211006': [7, 2], '20211007': [7, 3],
               '20220101': [3, 1], '20220102': [3, 2], '20220103': [3, 3], '20220131': [7, 1], '20220201': [7, 2],
               '20220202': [7, 2], '20220203': [7, 2], '20220204': [7, 2], '20220205': [7, 2], '20220206': [7, 2],
               '20220403': [3, 1], '20220404': [3, 2], '20220405': [3, 3], '20220501': [5, 2], '20220502': [5, 2],
               '20220503': [5, 2], '20220504': [5, 3], '20220430': [5, 1], '20220603': [3, 1], '20220604': [3, 2],
               '20220605': [3, 3], '20220910': [3, 1], '20220911': [3, 2], '20220912': [3, 3]}
    if if_data:  # 根据输入的参数，进行模型参数处理
        data_sample_ls = []
        # 判断当前日期是否为节假日,以及假期天数和当前所处假期阶段
        try:
            holiday_list = holiday[path[:10].replace('-', '')]
            data_sample_ls.append(1)
            data_sample_ls.extend(holiday_list)
        except:
            data_sample_ls.extend([0, 0, 0])
        path = datetime.datetime.strptime(path, '%Y-%m-%d %H:%M:%S')
        # 判断是否是工作日
        if is_workday(path):
            data_sample_ls.append(1)
        else:
            data_sample_ls.append(0)
        # 判断是星期几
        day = datetime.date(path.year, path.month, path.day)
        data_sample_ls.append(day.isoweekday())
        # 计算当前时段号
        minute = path.hour * 60 + path.minute + path.second / 60
        if minute == 0:
            data_sample_ls.append(1)
        else:
            data_sample_ls.append(math.ceil(minute / 5))
        # add month
        data_sample_ls.append(path.month)

        return data_sample_ls
    else:  # 针对文件进行批量参数处理
        if treat_type == 'gantry':
            # 得到门架车型流量的对应字典
            data = dbf.get_disc_from_document(path, [0, 1, 2, 3], encoding='utf-8', key_for_N=False, key_length=3,
                                              ifIndex=False, key_for_N_type='list', sign='_')
            # 遍历各门架车型，计算每条记录的训练数据
            data_sample = []  # 用于保存训练数据
            for key in data.keys():
                data_sample_ls = []  # 用于保存单独一条的数据
                key_list = key.split('_')
                if key_list[1] <= '2021-08-01 04:00:00':
                    continue
                data_sample_ls.extend(key_list)
                # 将前4个小时的所有流量数据获取到
                this_time = datetime.datetime.strptime(key_list[1], '%Y-%m-%d %H:%M:%S')
                for i in range(48):
                    new_time = this_time + datetime.timedelta(minutes=(5 * (i + 1) * (-1)))
                    new_key = key_list[0] + '_' + datetime.datetime.strftime(new_time, '%Y-%m-%d %H:%M:%S') + '_' + \
                              key_list[2]
                    try:
                        new_flow = data[new_key]
                        data_sample_ls.append(new_flow)
                    except:
                        data_sample_ls.append(0)
                # 判断当前日期是否为节假日,以及假期天数和当前所处假期阶段
                try:
                    holiday_list = holiday[key_list[1][:10].replace('-', '')]
                    data_sample_ls.append(1)
                    data_sample_ls.extend(holiday_list)
                except:
                    data_sample_ls.extend([0, 0, 0])
                # 判断是否是工作日
                if is_workday(this_time):
                    data_sample_ls.append(1)
                else:
                    data_sample_ls.append(0)
                # 判断是星期几
                day = datetime.date(this_time.year, this_time.month, this_time.day)
                data_sample_ls.append(day.isoweekday())
                # 计算当前时段号
                minute = this_time.hour * 60 + this_time.minute + this_time.second / 60
                if minute == 0:
                    data_sample_ls.append(1)
                else:
                    data_sample_ls.append(math.ceil(minute / 5))
                # add month
                data_sample_ls.append(this_time.month)
                # 最后添加本时刻流量
                data_sample_ls.append(data[key])
                # 保存下来每一条数据
                data_sample.append(data_sample_ls)
            save_name = './4.data_check/cangestion/flow_sample_data.csv'

        else:
            # 得到门架车型流量的对应字典
            data = dbf.get_disc_from_document(path, [2, 3, 0, 4, 5], encoding='utf-8', key_for_N=False, key_length=4,
                                              ifIndex=False, key_for_N_type='list', sign='_')
            # 遍历各门架车型，计算每条记录的训练数据
            data_sample = []  # 用于保存训练数据
            for key in data.keys():
                data_sample_ls = []  # 用于保存单独一条的数据
                key_list = key.split('_')
                if key_list[2] <= '2021-08-01 04:00:00':
                    continue
                data_sample_ls.extend(key_list)
                # 将前4个小时的所有流量数据获取到
                this_time = datetime.datetime.strptime(key_list[2], '%Y-%m-%d %H:%M:%S')
                for i in range(48):
                    new_time = this_time + datetime.timedelta(minutes=(5 * (i + 1) * (-1)))
                    new_key = key_list[0] + '_' + key_list[1] + '_' + datetime.datetime.strftime(new_time,
                                                                                                 '%Y-%m-%d %H:%M:%S') + '_' + \
                              key_list[3]
                    try:
                        new_flow = data[new_key]
                        data_sample_ls.append(new_flow)
                    except:
                        data_sample_ls.append(0)
                # 判断当前日期是否为节假日,以及假期天数和当前所处假期阶段
                try:
                    holiday_list = holiday[key_list[2][:10].replace('-', '')]
                    data_sample_ls.append(1)
                    data_sample_ls.extend(holiday_list)
                except:
                    data_sample_ls.extend([0, 0, 0])
                # 判断是否是工作日
                if is_workday(this_time):
                    data_sample_ls.append(1)
                else:
                    data_sample_ls.append(0)
                # 判断是星期几
                day = datetime.date(this_time.year, this_time.month, this_time.day)
                data_sample_ls.append(day.isoweekday())
                # 计算当前时段号
                minute = this_time.hour * 60 + this_time.minute + this_time.second / 60
                if minute == 0:
                    data_sample_ls.append(1)
                else:
                    data_sample_ls.append(math.ceil(minute / 5))
                # add month
                data_sample_ls.append(this_time.month)
                # 最后添加本时刻流量
                data_sample_ls.append(data[key])
                # 保存下来每一条数据
                data_sample.append(data_sample_ls)
            if treat_type == 'enStation':
                save_name = './4.data_check/enStation_flow/flow_sample_data_enStation.csv'

            elif treat_type == 'exStation':
                save_name = './4.data_check/exStation_flow/flow_sample_data_exStation.csv'

    with open(save_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_sample)


'''
    创建时间：2022/9/14
    完成时间：2022/9/14
    功能：基于神网模型进行一段时间门架或者收费站流量的预测
    关键词：流量预测、神网模型、一段时间
    修改时间：
'''


def compute_future_flow_of_gantry_station(compute_no, key_name, this_time, basic_data=None, treat_type='gantry'):
    """
    基于神网模型进行一段时间门架或者收费站流量的预测
    :param now_flow_dict:
    :param str this_time: 当前时刻
    :param str key_name: 对象的ID
    :param str treat_type: 选择预测对象，"gantry"为针对门架的，"station"为针对收费站，"all"为门架和收费站均计算
    :param int compute_no: 需要预测的未来时间段数
    :param list basic_data: 模型预测的起始流量数据
    :return:
    """
    gantry_flow = {}  # 保存门架预测流量数据
    station_flow = {}  # 保存收费站预测流量数据
    vehicle_types = [['CAR1_FLOW'], ['CAR2_FLOW'], ['CAR3_FLOW'], ['CAR4_FLOW'], ['TRUCK1_FLOW'], ['TRUCK2_FLOW'],
                     ['TRUCK3_FLOW'], ['TRUCK4_FLOW'], ['TRUCK5_FLOW'], ['TRUCK6_FLOW']]
    vehicle_types_int = [1, 2, 3, 4, 11, 12, 13, 14, 15, 16]

    # 获取上一个收费站的上站预测模型参数
    if treat_type == 'gantry':
        basic_path = './Network_Parameter_old/gantry_model/' + key_name
    elif treat_type == 'station_in':
        basic_path = './Network_Parameter_old/station_model/in_model/' + key_name
    elif treat_type == 'station_out':
        basic_path = './Network_Parameter_old/station_model/out_model/' + key_name
    try:
        with open(basic_path + '/initial.pkl', 'rb') as file:
            data_initial = pickle.loads(file.read())
    except:
        # dop.save_data_of_important([[key_name]], './error/error.csv', 'check_save', [0], [0])
        return []
    # 获取神网模型对象，并进行赋参
    neural_network = neuralNetwork.neuralNetwork_2(data_initial['inputnodes'], data_initial['outputnodes'],
                                                   data_initial['hnodes_num'],
                                                   data_initial['hnodes_num_list'], data_initial['learningrate'],
                                                   data_initial['activation_type'])
    neural_network.load_paremeter(basic_path)

    for i in range(compute_no):
        # 获取时间参数
        time_data = get_feature_of_flow_calculate(this_time, if_data=True)

        num = 0  #
        flow = []
        for k, vehicle_type in enumerate(vehicle_types):
            if treat_type == 'gantry':
                vehicle_type_list = [vehicle_types_int[k]]
                vehicle_type_list.extend(basic_data[key_name + '_' + vehicle_type[0]])
                vehicle_type_list.extend(time_data)
                if k == 0:
                    max_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/gantry_model/' + key_name + '/max.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                    min_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/gantry_model/' + key_name + '/min.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                for j, key in enumerate(max_list.keys()):
                    if min_list[key] == '0':
                        min_list[key] = 0
                    else:
                        min_list[key] = float(min_list[key])
                    vehicle_type_list[j] = (float(vehicle_type_list[j]) - float(min_list[key])) / (
                            float(max_list[key]) - float(min_list[key])) * 0.99 + 0.1
                flow = neural_network.query(vehicle_type_list)

            elif treat_type == 'station_in':
                # try:
                #     sample_data = basic_data[key_name + '_' + vehicle_type[0]]
                # except:
                #
                vehicle_type_list = [vehicle_types_int[k]]
                vehicle_type_list.extend(basic_data[key_name + '_' + vehicle_type[0]])
                vehicle_type_list.extend(time_data)
                if k == 0:
                    max_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/station_model/in_model/' + key_name + '/max.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                    min_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/station_model/in_model/' + key_name + '/min.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                for j, key in enumerate(max_list.keys()):
                    if (float(max_list[key]) - float(min_list[key])) == 0:
                        vehicle_type_list[j] = (vehicle_type_list[j] - float(min_list[key])) * 0.99 + 0.1
                    else:
                        vehicle_type_list[j] = (vehicle_type_list[j] - float(min_list[key])) / (
                                float(max_list[key]) - float(min_list[key])) * 0.99 + 0.1

                # get
                flow = neural_network.query(vehicle_type_list)

            elif treat_type == 'station_out':
                vehicle_type_list = [vehicle_types_int[k]]
                vehicle_type_list.extend(basic_data[key_name + '_' + vehicle_type[0]])
                vehicle_type_list.extend(time_data)
                if k == 0:
                    max_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/station_model/out_model/' + key_name + '/max.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                    min_list = dbf.get_disc_from_document(
                        './Network_Parameter_old/station_model/out_model/' + key_name + '/min.csv',
                        [0, 1], encoding='utf-8', ifIndex=False, ifNoCol=True)
                for j, key in enumerate(max_list.keys()):
                    if min_list[key] == '0':
                        min_list[key] = 0
                    else:
                        min_list[key] = float(min_list[key])
                    vehicle_type_list[j] = (vehicle_type_list[j] - float(min_list[key])) / (
                            float(max_list[key]) - float(min_list[key])) * 0.99 + 0.1
                flow = neural_network.query(vehicle_type_list)

            # 将得到的9位数转换为0和1的值
            flow = [0 if i[0] < 0.5 else 1 for i in flow]
            # 将9位0和1的数组转为字符串，用于十进制转化
            pre_result_str = ''
            for j in flow:
                pre_result_str = pre_result_str + str(j)

            if treat_type == 'gantry':
                ls = basic_data[key_name + '_' + vehicle_type[0]]
                ls.pop(0)
                ls.append(int(pre_result_str, 2))
                basic_data[key_name + '_' + vehicle_type[0]] = ls
            # num += int(pre_result_str, 2)
            if treat_type == 'gantry':
                gantry_flow[this_time + '_' + str(vehicle_type[0])] = int(pre_result_str, 2)
            else:
                station_flow[this_time + '_' + str(vehicle_type[0])] = int(pre_result_str, 2)

        # 更新最新数据
        this_time = datetime.datetime.strftime(datetime.datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S') +
                                               datetime.timedelta(minutes=5), '%Y-%m-%d %H:%M:%S')

    if treat_type == 'gantry':
        return gantry_flow
    elif treat_type == 'station_in' or treat_type == 'station_out':
        return station_flow
    else:
        return gantry_flow, station_flow


'''
    创建时间：2022/9/14
    完成时间：2022/9/14
    功能：判断路段拥堵的等级(根据车道各拥堵判定算法结果进行)
    关键词：拥堵等级，拥堵判定
    修改时间：No.1 2022-10-11,
'''


@dbf.timeIf_wrapper
def charge_congestion_level_by_result(congestion_charge_speed, congestion_charge_flow, congestion_charge_have,
                                      gantry_have_data, this_time):
    """
    判断路段拥堵的等级(根据车道各拥堵判定算法结果进行)
    :param this_time:
    :param gantry_have_data:
    :param congestion_charge_speed:
    :param congestion_charge_flow: 根据道路输入输出曲线匹配得到的拥堵判定结果
    :param congestion_charge_have: 根据道路承载量得到的拥堵判定结果
    :return:
    """

    vehicle_types = [['1'], ['2'], ['3'], ['4'], ['11'], ['12'], ['13'], ['14'], ['15'], ['16']]
    level_data = []  # 保存各路段拥堵相关信息
    for key in congestion_charge_speed.keys():
        gantry_have_data_new = {}
        # 如果速度判定为拥堵情况
        if congestion_charge_speed[key] == 1:
            # 判断是否是立交处的门架
            try:
                for i, vtype in enumerate(vehicle_types):
                    gantry_have_data_new[vtype[0]] = gantry_have_data[key + '_' + this_time][i]  # changed at 2023/01/29
            except:
                # 如果报错即为立交处门架，此时判定拥堵，默认一般等级
                level_data.append([key, 3, 2, length])
                continue
            # print('speed ' + key)
            # print(gantry_have_data_new)
            length = mcm.monte_carlo_main(gantry_have_data_new, [1, 0])
            try:
                # 如果承载量判定有相应结果，则进行相应拥堵等级赋值
                charge_have = congestion_charge_have[key]
                if charge_have == 2:
                    level_data.append([key, 5, 2, length])
                elif charge_have == 1:
                    level_data.append([key, 4, 2, length])
                elif length > 600:  # bug
                    level_data.append([key, 3, 2, length])
                else:
                    level_data.append([key, 0, 0, length])

            except:
                level_data.append([key, 4, 2, length])
        # 如果速度判定为非拥堵情况
        else:
            try:
                # 如果曲线拟合判定有相应结果
                charge_flow = congestion_charge_flow[key]
                if charge_flow == 1:
                    for i, vtype in enumerate(vehicle_types):
                        try:
                            gantry_have_data_new[vtype[0]] = gantry_have_data[key + '_' + this_time][i]
                        except:
                            gantry_have_data_new[vtype[0]] = 0
                    # print('flow ' + key)
                    # print(gantry_have_data_new)
                    length = mcm.monte_carlo_main(gantry_have_data_new, [1, 0])
                    try:
                        charge_have = congestion_charge_have[key]
                        if charge_have == 2:
                            level_data.append([key, 5, 2, length])
                        elif charge_have == 1:
                            level_data.append([key, 4, 2, length])
                        elif length > 600:  # bug
                            level_data.append([key, 3, 2, length])
                        else:
                            level_data.append([key, 0, 0, length])
                    except:
                        level_data.append([key, 4, 2, length])
                else:
                    try:
                        charge_have = congestion_charge_have[key]
                        if charge_have == 2:
                            level_data.append([key, 3, 1, 0])
                        elif charge_have == 1:
                            level_data.append([key, 2, 1, 0])
                        else:
                            level_data.append([key, 0, 0, 0])
                    except:
                        level_data.append([key, 4, 2, 0])
            except:
                # 如果曲线拟合判定没有相应结果，默认赋值为非拥堵
                level_data.append([key, 0, 0, 0])

    return level_data


'''
    创建时间：2022/10/08
    完成时间：2022/10/08
    功能：将所有的速度数据，按照时间和车型进行平均计算
    关键词：速度，平均
    修改时间：
'''


def compute_avg_data_of_speed():
    """
    将所有的速度数据，按照时间和车型进行平均计算
    :return:
    """
    # 得到速度的字典
    gantry_service_list = dbf.get_disc_from_document('./4.statistic_data/basic_data/speed_origin_data_time.csv',
                                                     [0, 2, 1, 3], encoding='gbk', key_for_N=True, key_length=3,
                                                     ifIndex=False, key_for_N_type='list', sign='_')
    # gantry_service_list = dbf.get_disc_from_document(data,
    #                                                  [0, 2, 1, 3], key_for_N=True, key_length=3,
    #                                                  ifIndex=False, key_for_N_type='list', sign='_', input_type='list')

    # 进行平均计算
    data_result = dbf.compute_dict_by_group(gantry_service_list, [], 'avg', '')
    dbf.basic_save_dict_data(data_result, './4.statistic_data/basic_data/speed_origin_data_avg.csv', True, True)


"""
ls
"""


def get_last_time_inout_data(path, time_no, this_time):
    data_dict = {}
    with open(path) as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if i > 0:
                try:
                    data_dict[row[1] + '_in'].append([row[0] + '_in', float(row[3])])
                except:
                    data_dict[row[1] + '_in'] = [[row[0] + '_in', float(row[3])]]
                try:
                    data_dict[row[1] + '_out'].append([row[0] + '_out', float(row[4])])
                except:
                    data_dict[row[1] + '_out'] = [[row[0] + '_out', float(row[4])]]
    data_time_dict = {}
    for i in range(time_no):
        t = datetime.datetime.strftime(datetime.datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S') +
                                       datetime.timedelta(minutes=5 * (time_no - i - 1) * (-1)),
                                       '%Y-%m-%d %H:%M:%S')
        for j in range(len(data_dict[t + '_in'])):
            try:
                data_time_dict[data_dict[t + '_in'][j][0]].append(data_dict[t + '_in'][j][1])
            except:
                data_time_dict[data_dict[t + '_in'][j][0]] = [data_dict[t + '_in'][j][1]]
        for j in range(len(data_dict[t + '_out'])):
            try:
                data_time_dict[data_dict[t + '_out'][j][0]].append(data_dict[t + '_out'][j][1])
            except:
                data_time_dict[data_dict[t + '_out'][j][0]] = [data_dict[t + '_out'][j][1]]
    return data_time_dict


"""
ls
"""


def get_last_flow_data(data, this_time):
    data_dict = {}
    for i in range(len(data)):
        data_dict[data[i][0]] = data[i][2:51]
    return data_dict


'''
    创建时间：2023/2/8
    完成时间：2023/2/8
    功能：针对门架和收费站原始数据进行特征转换
    关键词：
    修改时间：
'''


@dbf.timeIf_wrapper
def get_feature_of_flow(data, target='gantrys'):
    """
    针对门架和收费站原始数据进行特征转换
    :param DataFrame data: 输入数据
    :param target:
    :return:
    """
    columns = ['gantry_time', "CAR1_FLOW", "CAR2_FLOW", "CAR3_FLOW", "CAR4_FLOW", "TRUCK1_FLOW", "TRUCK2_FLOW",
               "TRUCK3_FLOW", "TRUCK4_FLOW", "TRUCK5_FLOW", "TRUCK6_FLOW"]
    if target == 'gantrys':
        data['TIME_POINT_ORDER'] = data['TIME_POINT_ORDER'].astype(str)
        data['gantry_time'] = data['INTERVAL_ID'] + '_' + data['TIME_POINT_ORDER']
        data = data[columns]
        data = data.set_index('gantry_time', drop=True).unstack()
        data = data.reset_index()
        data.columns = ['VEHICLE_TYPE', 'gantry_time', 'FLOW']
        data['INTERVAL_ID'] = data['gantry_time'].map(lambda x: x.split('_')[0])
        data = data[['VEHICLE_TYPE', 'INTERVAL_ID', 'FLOW']]
    else:
        data[['TIME_POINT_ORDER', 'DIRECTION', 'CLOSEST_GANTRY_ID']] = data[
            ['TIME_POINT_ORDER', 'DIRECTION', 'CLOSEST_GANTRY_ID']].astype(str)
        data['gantry_time'] = data['STATION_ID'] + '_' + data['TIME_POINT_ORDER'] + '_' + data['DIRECTION'] + '_' + \
                              data['CLOSEST_GANTRY_ID']
        data = data[columns]
        data = data.set_index('gantry_time', drop=True).unstack()
        data = data.reset_index()
        data.columns = ['VEHICLE_TYPE', 'gantry_time', 'FLOW']
        data['STATION_ID'] = data['gantry_time'].map(lambda x: x.split('_')[0])
        data['DIRECTION'] = data['gantry_time'].map(lambda x: x.split('_')[2])
        data['CLOSEST_GANTRY_ID'] = data['gantry_time'].map(lambda x: x.split('_')[3])
        data = data[['VEHICLE_TYPE', 'STATION_ID', 'DIRECTION', 'CLOSEST_GANTRY_ID', 'FLOW']]

    return data


'''
    创建时间：2023/2/14
    完成时间：2023/2/14
    功能：针对收费站流量数据进行数据补全，并将由于门架误识别造成的数据进行合并
    关键词：
    修改时间：
'''


@dbf.timeIf_wrapper
def get_full_data_of_station_flow(data, next_gantry_dict, last_gantry_dict, visual_gantry):
    """
    针对收费站流量数据进行数据补全，并将由于门架误识别造成的数据进行合并
    :param data: 补全目标数据
    :param next_gantry_dict: 收费站下一个门架对应字典
    :param last_gantry_dict:收费站上一个门架对应字典
    :param visual_gantry: 虚拟门架列表
    :return:
    """
    # 将待补全数据转化为字典
    columns = list(data.columns.values)
    columns.pop(1)
    times = list(set(data['TIME_POINT_ORDER'].values))
    data_dict = dbf.get_disc_from_document(data.values, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                           ifIndex=False, encoding='utf-8', key_length=4, length=[14, 4, 1, 16],
                                           sign='_', input_type='list')
    result_data = []  # 保存补全后结果数据
    # 遍历所有的时间段
    for j in range(len(times)):
        # 遍历所有的收费站对应数据
        for key in next_gantry_dict.keys():
            if len(key) < 10:
                continue
            direct = '1'
            visual_data = [0 for i in range(10)]
            for gantry in next_gantry_dict[key]:
                feature_visual = [key, times[j], direct, '']
                # 判断是否是虚拟门架
                if gantry in visual_gantry:
                    try:
                        visual_data = np.array(visual_data) + np.array(data_dict[key + '_' + str(times[j]) + '_1_' +
                                                                                 gantry])
                    except:
                        pass
                else:
                    feature = [key, times[j], direct, gantry[:-2]]
                    data_ls = [0 for i in range(10)]
                    for i in range(2):
                        try:
                            data_ls = np.array(data_ls) + np.array(data_dict[key + '_' + str(times[j]) + '_1_' +
                                                                             gantry[:-2] + str(i + 1) + '0'])
                        except:
                            pass
                    feature.extend(data_ls)
                    result_data.append(feature)
            feature_visual.extend(visual_data)
            result_data.append(feature_visual)

        # 遍历所有的收费站对应数据
        for key in last_gantry_dict.keys():
            if key == 'None':
                continue
            direct = '2'
            visual_data = [0 for i in range(10)]
            for gantry in last_gantry_dict[key]:
                feature_visual = [key, times[j], direct, '']
                # 判断是否是虚拟门架
                if gantry in visual_gantry:
                    try:
                        visual_data = np.array(visual_data) + np.array(data_dict[key + '_' + str(times[j]) + '_2_' +
                                                                                 gantry])
                    except:
                        pass
                else:
                    feature = [key, times[j], direct, gantry[:-2]]
                    data_ls = [0 for i in range(10)]
                    for i in range(2):
                        try:
                            data_ls = np.array(data_ls) + np.array(data_dict[key + '_' + str(times[j]) + '_2_' +
                                                                             gantry[:-2] + str(i + 1) + '0'])
                        except:
                            pass
                    feature.extend(data_ls)
                    result_data.append(feature)
            feature_visual.extend(visual_data)
            result_data.append(feature_visual)
    result_data = pd.DataFrame(result_data, columns=columns)
    return result_data
