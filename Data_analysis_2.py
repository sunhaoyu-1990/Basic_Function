import datetime
import math

import numpy as np
import Document_process as dop
from data import Data_Basic_Function as dbf
import Parameter_Comput as pc
import Keyword_and_Parameter as kp
import csv

'''
    创建时间：2021/11/21
    修改时间：2021/11/21
    内容：完成对车辆稽查各数据的分析,并进行相应可视化数据的输出
'''


'''
    创建时间:2022/6/28
    完成时间:2022/6/28
    功能: 得到某车所有记录的标准路径，并对比出缺失点位，累计每个缺失点位的数量
'''


def get_loss_gantry_by_plate(vehicle_plate):
    """
    得到某车所有记录的标准路径，并对比出缺失点位，累计每个缺失点位的数量
    :param vehicle_plate: 目标车牌号
    :return:
    """
    # 获取该车辆路径所在文件的地址
    case_data_vehicle_path = kp.get_parameter_with_keyword('case_data_vehicle_path')
    case_gantry_result_path = case_data_vehicle_path + vehicle_plate + '/gantry_result_' + vehicle_plate + '.csv'
    # 获取标准路径的正向字典数据
    standard_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv', ['ENROADNODEID', 'EXROADNODEID'],
                                               encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取标准路径的反向字典数据
    standard_back_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                                    ['EXROADNODEID', 'ENROADNODEID'],
                                                    encoding='gbk', key_for_N=True, key_for_N_type='list')

    # 循环该车辆的所有路径的标准路径和缺失点位统计数据
    standard_gantry_list = []  # 用于保存完整路径
    loss_gantry_num = {}  # 用于存储每个门架及其缺失次数
    loss_gantry = []  # 用于存储每个门架及其缺失次数并保存
    with open(case_gantry_result_path) as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if i == 0:
                # 获取收费单元所在字段的下标
                index = dbf.get_index_of_list(row, ['收费单元路径', '本省入口站id', '本省出口站id'])
            else:
                gantry_list_in = row[index[0]].split('|')
                gantry_list = row[index[0]].split('|')
                # 获得标准路径
                gantry_whole_list = dbf.get_standard_data(gantry_list_in, standard_dict, standard_back_dict,
                                                          row[index[1]], row[index[2]])
                standard_gantry_list.append(gantry_whole_list)
                # 将各门架ID进行处理，只保留前14位
                gantry_whole_list = [gantry[:14] for gantry in gantry_whole_list]
                gantry_list = [gantry[:14] for gantry in gantry_list]
                for gantry in gantry_whole_list:
                    if type(gantry) == str:
                        try:
                            loss_gantry_num[gantry] -= 1
                        except:
                            loss_gantry_num[gantry] = -1
                for gantry in gantry_list:
                    try:
                        loss_gantry_num[gantry] += 1
                    except:
                        loss_gantry_num[gantry] = 1

    for key in loss_gantry_num.keys():
        # 如果门架数值为0，说明未缺失
        if loss_gantry_num[key] == 0:
            pass
        # 如果门架数值小于0，说明缺失
        elif loss_gantry_num[key] < 0:
            loss_gantry.append([key, loss_gantry_num[key]])
        # 如果门架数值大于0，说明标准路径出现问题
        else:
            print('标准路径异常')

    gantry_save_path = case_data_vehicle_path + vehicle_plate + '/standard_gantry_data.csv'
    loss_save_path = case_data_vehicle_path + vehicle_plate + '/loss_gantry_data.csv'

    with open(gantry_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(standard_gantry_list)
    with open(loss_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(loss_gantry)


'''
    创建时间:2022/6/29
    完成时间:2022/6/29
    功能: 求取目标文件数据中的选定字段的五分位数和均值，可以设置筛选条件
    关键字：数据分布
    修改时间：No.1 2022/8/31，加入了直接对字典类型数据的分布参数计算
'''


def get_statistics_value_from_document(if_path=True, path='', features=[], key_value={}, feature_limit=[],
                                       treat_value='', save_name='', if_return=False):
    """
    求取目标文件数据中的选定字段的五分位数和均值，可以设置筛选条件
    :param save_name: 结果文件保存名称
    :param key_value: 如果if_path为FALSE，则直接对字典数据进行处理
    :param if_path: 决定进行哪种处理，TRUE则进行文件内容读取再进行分布参数计算，FALSE则直接进行分布参数计算
    :param treat_value: 如果需要对每个特征数值进行处理，1.这里可以输入特征字段名，则每个特征值除以该特征值，2.也可以输入数值，直接每一特征除以该值
    :param path: 目标文件地址
    :param features: 目标特征名称数组
    :param feature_limit: 各特征对应筛选条件数组
    :return:
    """
    if if_path:
        feature_value = {}
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    # 获取所有目标字段的下标
                    index = dbf.get_index_of_list(row, features)
                    # 如果treat_value不为’‘，则需要对每个特征数值进行处理，如果为字符串，则表明输入的为字段名称
                    if treat_value != '' and type(treat_value) == str:
                        treat_index = dbf.get_index_of_list(row, [treat_value])
                else:
                    for j in range(len(features)):
                        # 如果每个字段的限制条件内容为空，则不需要进行筛选，反之则进行筛选
                        if len(feature_limit) > 0:
                            # 如果进行筛选，则值与筛选值相同，则直接跳过
                            # bug:只能进行相等的筛选
                            if float(row[index[j]]) == float(feature_limit[j]):
                                continue
                        #
                        if treat_value != '' and type(treat_value) == str:
                            row_value = round(float(row[index[j]]) / float(row[treat_index[0]]), 3)
                        elif treat_value != '':
                            row_value = round(float(row[index[j]]) / treat_value, 3)
                        else:
                            row_value = float(row[index[j]])
                        try:
                            feature_value[features[j]].append(row_value)
                        except:
                            feature_value[features[j]] = [row_value]
    else:
        feature_value = key_value
    result = [['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'mean', 'mode', 'gantry', 'vehicle_type']]  # 用于保存所有特征的5分位值和均值
    for key in feature_value.keys():
        # 将每个特征的数组数据输入，得到其5分位值和均值
        data_list = dbf.get_statistics_value_of_list(feature_value[key], 'all', sortBySort=True)
        feature_value[key] = data_list
        # 判断key是否是拼接字符串，如果是则直接进行拆解添加
        if '_' in key:
            data_list.extend(key.split('_'))
        elif '-' in key:
            data_list.extend(key.split('-'))
        else:
            data_list.append(key)
        result.append(data_list)

    if save_name == '':
        save_name = kp.get_parameter_with_keyword('case_data_path') + 'statistic.csv'

    with open(save_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)

    if if_return:
        return feature_value


'''
    创建时间:2022/7/12
    完成时间:2022/7/12
    功能: 分别对门架各车型流量、门架各来源流量、收费站各车辆各方向上站和下站流量
'''


def compute_num_of_gantry_and_station(path, treat_type, seperate_time=5, start_date='', end_date=''):
    """
    分别对门架各车型流量、门架各来源流量、收费站各车辆各方向上站和下站流量
    :param seperate_time: 流量的处理周期时间
    :param start_date: 数据运算开始的日期设定
    :param path: 数据地址
    :param treat_type: 数据处理类型，gantry是计算门架数据，enStation为计算入口数据，exStation为计算出口数据
    :return:
    """
    # 判断设定的周期时间是否合理
    if 1440 % seperate_time > 0:
        print('设定的周期时间不合理！')
        return 0
    # 获取所有文件的地址
    paths = np.sort(path)
    if treat_type == 'gantry':
        path_list = dop.path_of_holder_document('./4.data_check/gantry_flow/')
        if len(path_list) != 0:
            key_value_type = dbf.get_disc_from_document('./4.data_check/gantry_flow/gantry_type_flow.csv',
                                                        [0, 1, 2, 3, 4], encoding='utf-8', key_length=4, ifIndex=False)
            key_value_class = dbf.get_disc_from_document('./4.data_check/gantry_flow/gantry_class_flow.csv',
                                                         [0, 1, 2, 3, 4], encoding='utf-8', key_length=4, ifIndex=False)
            key_value_from = dbf.get_disc_from_document('./4.data_check/gantry_flow/gantry_from_flow.csv',
                                                        [0, 1, 2, 3, 4], encoding='utf-8', key_length=4, ifIndex=False)
            key_value_enHex = dbf.get_disc_from_document('./4.data_check/gantry_flow/enHex_type_flow.csv',
                                                         [0, 1, 2, 3, 4, 5], encoding='utf-8', key_length=5,
                                                         ifIndex=False)
        else:
            key_value_type = {}  # 记录各门架各车型的流量
            key_value_class = {}  # 记录各门架各车种的流量
            key_value_from = {}  # 记录各门架来源的流量
            key_value_enHex = {}  # 记录各收费站各方向上的各车型的流量
    elif treat_type == 'enStation':
        key_value_type = {}  # 记录各收费站各车型的上站流量
        key_value_class = {}  # 记录各收费站各车种的上站流量
    elif treat_type == 'exStation':
        key_value_type = {}  # 记录各收费站各车型的下站流量
        key_value_class = {}  # 记录各收费站各车种的下站流量

    # 获取门架ID与其入口收费站的对应字典数据
    gantry_enStation = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv', ['id', 'enTollStation'],
                                                  encoding='utf-8')
    # 获取收费站ID和HEX码的对应字典数据
    station_HEX = dbf.get_disc_from_document('../Data_Origin/station_HEX.csv', ['id', 'NEWSTATIONHEX'],
                                             encoding='gbk')
    for k, path in enumerate(paths):
        if start_date != '' and path[-12:-4] < start_date:
            continue
        if end_date != '' and path[-12:-4] > end_date:
            continue
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:
                    if treat_type == 'gantry':
                        index = dbf.get_index_of_list(row, ['GANTRYID', 'TRANSTIME', 'ENTIME', 'VEHICLETYPE',
                                                             'VEHICLECLASS', 'ENTOLLSTATIONHEX'])
                    elif treat_type == 'enStation':
                        index = dbf.get_index_of_list(row, ['ENSTATIONHEX', 'ENTIME', 'VEHICLETYPE', 'VEHICLECLASS'])
                    elif treat_type == 'exStation':
                        index = dbf.get_index_of_list(row, ['EXSTATIONFX', 'EXTIME', 'EXVEHICLETYPE', 'EXVEHICLECLASS',
                                                             'EXGANTRYID'])
                else:
                    if treat_type == 'gantry':
                        if row[index[1]] == '' and len(row[index[5]]) < 2:
                            continue
                        # 计算该条记录时间到当天0点0分0秒时的时间间隔
                        try:
                            time_gap = datetime.datetime.strptime(row[index[1]][:19],
                                                                  '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                (row[index[1]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')
                        except:
                            print(row[index[1]])
                            continue
                        # 计算该条记录时间所处的周期
                        num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                        if num == 0:
                            num = 1
                        # 计算该时间点所属于的观察时间点
                        watch_time = datetime.datetime.strftime(
                            datetime.datetime.strptime((row[index[1]][:11] + '00:00:00'),
                                                       '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')
                        # 记录各门架的各车型流量
                        try:
                            key_value_type[
                                watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + row[index[3]]] += 1
                        except:
                            key_value_type[
                                watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + row[index[3]]] = 1
                        # 记录各门架的各车种流量
                        try:
                            key_value_class[
                                watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + row[index[4]]] += 1
                        except:
                            key_value_class[
                                watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + row[index[4]]] = 1
                        # 记录各门架的入口来源流量
                        if row[index[5]][:2] != '61':  # 如果入口在省外，统一设为outPrivince
                            HEX = 'outPrivince'
                        else:
                            HEX = row[index[5]]
                        try:
                            key_value_from[watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + HEX] += 1
                        except:
                            key_value_from[watch_time + '_' + str(num) + '_' + row[index[0]][:16] + '_' + HEX] = 1
                        # 记录各收费站各方向上的各车型的流量
                        # 1.获取该门架的入口收费站对应的HEX码
                        try:
                            enHEX = station_HEX[gantry_enStation[row[index[0]][:16]]]
                            # 2.获取门架经过时间与入口时间的时间差
                            time_gap = (datetime.datetime.strptime(row[index[1]][:19],
                                                                   '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                row[index[2]][:19], '%Y-%m-%d %H:%M:%S')).total_seconds() / 60.0
                        except:
                            # print(row[index[0]][:16])
                            continue

                        if row[index[5]] == enHEX and time_gap < 20:
                            # 计算该条记录入口时间到当天0点0分0秒时的时间间隔
                            time_gap = datetime.datetime.strptime(row[index[2]][:19],
                                                                  '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                (row[index[2]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')
                            # 计算该条记录时间所处的周期
                            num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                            if num == 0:
                                num = 1
                            # 计算该时间点所属于的观察时间点
                            watch_time = datetime.datetime.strftime(
                                datetime.datetime.strptime((row[index[2]][:11] + '00:00:00'),
                                                           '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                    minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')

                            try:
                                key_value_enHex[
                                    watch_time + '_' + str(num) + '_' + row[index[5]] + '_' + row[index[0]][:16] + '_' +
                                    row[index[3]]] += 1
                            except:
                                key_value_enHex[
                                    watch_time + '_' + str(num) + '_' + row[index[5]] + '_' + row[index[0]][:16] + '_' +
                                    row[index[3]]] = 1
                        # if i > 5000:
                        #     break
                    elif treat_type == 'enStation':
                        # 计算该条记录时间到当天0点0分0秒时的时间间隔
                        time_gap = datetime.datetime.strptime(row[index[1]][:19],
                                                              '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                            (row[index[1]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')
                        # 计算该条记录时间所处的周期
                        num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                        if num == 0:
                            num = 1
                        # 计算该时间点所属于的观察时间点
                        watch_time = datetime.datetime.strftime(
                            datetime.datetime.strptime((row[index[1]][:11] + '00:00:00'),
                                                       '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')
                        # 记录各门架的各车型流量
                        try:
                            key_value_type[watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[2]]] += 1
                        except:
                            key_value_type[watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[2]]] = 1
                        # 记录各门架的各车种流量
                        try:
                            key_value_class[
                                watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[3]]] += 1
                        except:
                            key_value_class[watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[3]]] = 1
                    elif treat_type == 'exStation':
                        # 计算该条记录时间到当天0点0分0秒时的时间间隔
                        time_gap = datetime.datetime.strptime(row[index[1]][:19],
                                                              '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                            (row[index[1]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')
                        # 计算该条记录时间所处的周期
                        num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                        if num == 0:
                            num = 1
                        # 计算该时间点所属于的观察时间点
                        watch_time = datetime.datetime.strftime(
                            datetime.datetime.strptime((row[index[1]][:11] + '00:00:00'),
                                                       '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')
                        # 记录各门架的各车型流量
                        try:
                            key_value_type[
                                watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[4]][:16] + '_' +
                                row[index[2]]] += 1
                        except:
                            key_value_type[
                                watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[4]][:16] + '_' +
                                row[index[2]]] = 1
                        # 记录各门架的各车种流量
                        try:
                            key_value_class[
                                watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[4]][:16] + '_' +
                                row[index[3]]] += 1
                        except:
                            key_value_class[
                                watch_time + '_' + str(num) + '_' + row[index[0]] + '_' + row[index[4]][:16] + '_' +
                                row[index[3]]] = 1
        # if k >= 1:
        #     break
    # 根据的处理内容进行选择不同的保存地址
    if treat_type == 'gantry':
        save_name = './4.data_check/gantry_flow/'
        key_value_type_list = []  # 记录各门架各车型的流量
        # 对各结果字典数据转成数组
        for key in key_value_type.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_type[key])
            key_value_type_list.append(list_ls)
        with open(save_name + 'gantry_type_flow_' + start_date + '_' + end_date + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)

        key_value_type_list = []  # 记录各门架各车种的流量
        for key in key_value_class.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_class[key])
            key_value_type_list.append(list_ls)
        with open(save_name + 'gantry_class_flow_' + start_date + '_' + end_date + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)

        key_value_type_list = []  # 记录各门架来源的流量
        for key in key_value_from.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_from[key])
            key_value_type_list.append(list_ls)
        with open(save_name + 'gantry_from_flow_' + start_date + '_' + end_date + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)

        key_value_type_list = []  # 记录各收费站各方向上的各车型的流量
        for key in key_value_enHex.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_enHex[key])
            key_value_type_list.append(list_ls)
        # 进行保存
        with open(save_name + 'enHex_type_flow_' + start_date + '_' + end_date + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)

    elif treat_type == 'enStation':
        save_name = './4.data_check/enStation_flow/'
        key_value_type_list = []  # 记录各收费站各车型的上站流量
        key_value_class_list = []  # 记录各收费站各车种的上站流量
        # 对各结果字典数据转成数组
        for key in key_value_type.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_type[key])
            key_value_type_list.append(list_ls)
        for key in key_value_class.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_class[key])
            key_value_class_list.append(list_ls)
        # 进行保存
        with open(save_name + 'enStation_type_flow.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)
        with open(save_name + 'enStation_class_flow.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_class_list)
    elif treat_type == 'exStation':
        save_name = './4.data_check/exStation_flow/'
        key_value_type_list = []  # 记录各收费站各车型的下站流量
        key_value_class_list = []  # 记录各收费站各车种的下站流量
        # 对各结果字典数据转成数组
        for key in key_value_type.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_type[key])
            key_value_type_list.append(list_ls)
        for key in key_value_class.keys():
            list_ls = [x for x in key.split('_')]
            list_ls.append(key_value_class[key])
            key_value_class_list.append(list_ls)
        # 进行保存
        with open(save_name + 'exStation_type_flow.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_type_list)
        with open(save_name + 'exStation_class_flow.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(key_value_class_list)


'''
    创建时间:2022/7/21
    完成时间:2022/7/21
    功能: 计算收费站各方向的上下站流量（通过稽查数据）
'''


def compute_direction_num_of_station(start_time, end_time, seperate_time=5):
    """
    计算收费站各方向的上下站流量（通过稽查数据）
    :param start_time: 数据源时间起始节点
    :param end_time: 数据源时间截止节点
    :return:
    """
    # 获取全部的数据
    # paths = dop.path_of_holder_document(kp.get_parameter_with_keyword('result_data_many_path'), True)
    paths = dop.path_of_holder_document('./2.Middle_Data/202108/', True)
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202109/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202110/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202111/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202112/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202201/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202202/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202203/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202204/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202205/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202206/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202207/', True))
    paths.extend(dop.path_of_holder_document('./2.Middle_Data/202208/', True))

    station_in_num = {}  # 用于保存收费站各方向各车型的上站数量
    station_out_num = {}  # 用于保存收费站各方向各车型的下站数量

    # 获取各匹配数据的字典
    # 获取各收费单元对应路段长度的字典，用于判断各收费单元是否是主线门架
    # gantry_length = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv', ['id', 'length'], encoding='utf-8')

    # 获取收费单元的标准路径字典
    gantry_relation_list = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                                      ['ENROADNODEID', 'EXROADNODEID'],
                                                      encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取收费单元的逆向标准路径字典
    gantry_back_relation_list = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                                           ['EXROADNODEID', 'ENROADNODEID'],
                                                           encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 逐条进行分析合并
    for path in paths:
        date = path.rsplit('_', 1)[1][:-4]
        # date = path.rsplit('/', 1)[1][:-4]
        # 判断该文件时间是否在所需范围内
        if start_time <= date <= end_time:
            print(path)
            with open(path) as f:
                for i, row in enumerate(f):
                    row = row.split(',')
                    row[-1] = row[-1][:-1]
                    if i == 0:
                        # 获取到所需字段的索引
                        get_index = dbf.get_index_of_list(row,
                                                           kp.get_parameter_with_keyword('compute_station_num_middle'))
                    else:
                        if row[get_index[1]] != '':
                            # 每条进行收费单元串拆散
                            gantry_list = row[get_index[0]].split('|')

                            # 获取前后截取的收费单元的数量，bug：这里默认设置截取长度为5
                            if len(gantry_list) <= 5:
                                start = 0
                                end = len(gantry_list) - 1
                            else:
                                start = -5
                                end = 5

                            # 将开始和结尾附近的收费单元获取到，并去除方向代码，同时转换为集合
                            first_gantry = set([gantry[:-2] for gantry in gantry_list[:end]])
                            end_gantry = set([gantry[:-2] for gantry in gantry_list[start:]])

                            # 获取入口站ID下一个收费单元ID
                            try:
                                first_tollinterval = gantry_relation_list[row[get_index[1]]]
                            except:
                                first_tollinterval = []
                            # 判断入口站下一个收费单元是一个还是两个，如果为一个说明为虚拟门架不是主线门架
                            if len(first_tollinterval) == 1:
                                # 如果不是主线，则获取下下一个收费单元信息
                                first_tollinterval = set(
                                    [gantry[:-2] for gantry in gantry_relation_list[first_tollinterval[0]]])
                            elif len(first_tollinterval) > 1:
                                first_tollinterval = set(
                                    [gantry[:-2] for gantry in first_tollinterval])
                            else:
                                first_tollinterval = set([])

                            # 进行入口匹配
                            first_ls = first_tollinterval.intersection(first_gantry)
                            # 如果入口站下一个收费单元就是主线门架，匹配出所走的门架
                            if len(first_ls) != 0:
                                # 获取匹配到的门架ID
                                gantryID = first_ls.pop()
                                # 获取入口时间所属时段信息
                                time_gap = datetime.datetime.strptime(row[get_index[5]][:19],
                                                                      '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                    (row[get_index[5]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')

                                # 计算该条记录时间所处的周期
                                num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                                if num == 0:
                                    num = 1
                                # 计算该时间点所属于的观察时间点
                                watch_time = datetime.datetime.strftime(
                                    datetime.datetime.strptime((row[get_index[5]][:11] + '00:00:00'),
                                                               '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                        minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')
                                # 进行车型判断，如果有出口车型用出口车型，如果没有用入口车型
                                if row[get_index[7]] == '':
                                    vehicle_type = row[get_index[3]]
                                else:
                                    vehicle_type = row[get_index[7]]
                                # 保存入口信息
                                try:
                                    station_in_num[
                                        watch_time + '_' + str(num) + '_' + row[
                                            get_index[1]] + '_' + gantryID + '_' + vehicle_type] += 1
                                except:
                                    station_in_num[
                                        watch_time + '_' + str(num) + '_' + row[
                                            get_index[1]] + '_' + gantryID + '_' + vehicle_type] = 1
                            # else:
                            #     continue
                        if row[get_index[6]] != '' and row[get_index[7]] != '':
                            # 进行出口匹配
                            try:
                                end_tollinterval = gantry_back_relation_list[row[get_index[6]]]
                            except:
                                end_tollinterval = []
                            # 获取出口站ID上一个收费单元ID
                            # 判断出口站上一个收费单元是一个还是两个，如果为一个说明为虚拟门架不是主线门架
                            if len(end_tollinterval) == 1:
                                # 如果不是主线，则获取上上一个收费单元信息
                                end_tollinterval = set(
                                    [gantry[:-2] for gantry in gantry_back_relation_list[end_tollinterval[0]]])
                            elif len(end_tollinterval) > 1:
                                end_tollinterval = set(
                                    [gantry[:-2] for gantry in end_tollinterval])
                            else:
                                end_tollinterval = set([])

                            # 如果出口站上一个收费单元就是主线门架，匹配出所走的门架
                            end_ls = end_tollinterval.intersection(end_gantry)
                            if len(end_ls) != 0:
                                # 获取匹配到的门架ID
                                gantryID = end_ls.pop()
                                # 获取入口时间所属时段信息
                                time_gap = datetime.datetime.strptime(row[get_index[9]][:19],
                                                                      '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                                    (row[get_index[9]][:11] + '00:00:00'), '%Y-%m-%d %H:%M:%S')

                                # 计算该条记录时间所处的周期
                                num = math.ceil(float(time_gap.total_seconds()) / 60 / seperate_time)
                                if num == 0:
                                    num = 1
                                # 计算该时间点所属于的观察时间点
                                watch_time = datetime.datetime.strftime(
                                    datetime.datetime.strptime((row[get_index[9]][:11] + '00:00:00'),
                                                               '%Y-%m-%d %H:%M:%S') + datetime.timedelta(
                                        minutes=(seperate_time * (num - 1))), '%Y-%m-%d %H:%M:%S')
                                # 保存出口信息
                                try:
                                    station_out_num[
                                        watch_time + '_' + str(num) + '_' + row[
                                            get_index[6]] + '_' + gantryID + '_' + row[get_index[7]]] += 1
                                except:
                                    station_out_num[
                                        watch_time + '_' + str(num) + '_' + row[
                                            get_index[6]] + '_' + gantryID + '_' + row[get_index[7]]] = 1
                            # else:
                            #     continue
    # 本地保存数据
    station_list = []
    for key in station_out_num.keys():
        key_list = key.split('_')
        key_list.append(station_out_num[key])
        station_list.append(key_list)
    save_name = './4.data_check/exStation_flow/'
    with open(save_name + 'exStation_from_type_flow.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(station_list)
    station_list = []
    for key in station_in_num.keys():
        key_list = key.split('_')
        key_list.append(station_in_num[key])
        station_list.append(key_list)
    save_name = './4.data_check/enStation_flow/'
    with open(save_name + 'enStation_go_type_flow.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(station_list)


def get_station_flow(path, station_id, gantry_id, result_type='list'):
    """

    :param path:
    :param station_id:
    :return:
    """
    if result_type == 'list':
        data = []
    elif result_type == 'dict':
        data = {}
    with open(path) as f:
        for i, row in enumerate(f):
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if row[2] == station_id and row[3] == gantry_id:
                if result_type == 'list':
                    data.append(row)
                elif result_type == 'dict':
                    data[row[0] + '_' + row[4]] = row[5]
    return data


def get_pass_last_no_next_passid(start_time, end_time):
    """

    :param start_time:
    :param end_time:
    :return:
    """
    # gantrys = ["G003061003000120", "G003061003000210", "G003061003000110", "G003061003000310", "G003061003000220",
    #            "G003061003000410", "G003061003000320", "G003061003000510", "G003061003000420", "G003061003000520",
    #            "G003061003000710", "G003061003000720", "G003061003000810", "G003061003000820", "G003061003001310",
    #            "G003061003000610", "G003061003000620", "G003061003000910", "G003061003000920", "G003061003001010",
    #            "G003061003001020", "G003061003001120", "G003061003001110", "G003061003001220", "G003061003001210",
    #            "G003061003001320"]
    gantrys = ["G003061003000210", "G003061003000120", "G003061003000110", "G003061003000220",
               "G003061003000410", "G003061003000510", "G003061003000420", "G003061003000520",
               "G003061003000710", "G003061003000720", "G003061003000810", "G003061003000820", "G003061003001310",
               "G003061003000610", "G003061003000620", "G003061003000910", "G003061003000920", "G003061003001010",
               "G003061003001020", "G003061003001120", "G003061003001110", "G003061003001220", "G003061003001210",
               "G003061003001320"]
    # gantrys = ["G003061003000510", "G003061003000610", "G003061003000410", "G003061003001610", "G003061003001620"]
    origin_data = {}
    # 获取标准路径的正向字典数据
    standard_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv', ['ENROADNODEID', 'EXROADNODEID'],
                                               encoding='gbk', key_for_N=True, key_for_N_type='list')
    # 获取标准路径的反向字典数据
    standard_back_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
                                                    ['EXROADNODEID', 'ENROADNODEID'],
                                                    encoding='gbk', key_for_N=True, key_for_N_type='list')
    station_in_dict = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv',
                                                 ['id', 'enTollStation'],
                                                 encoding='utf-8', key_for_N=False, key_for_N_type='list')
    # station_out_dict = dbf.get_disc_from_document('../Data_Origin/tollinterval.csv',
    #                                              ['id', 'exTollStation'],
    #                                              encoding='utf-8', key_for_N=False, key_for_N_type='list')
    station_id = dbf.get_disc_from_document('../Data_Origin/station_HEX.csv',
                                            ['NEWSTATIONHEX', 'id'],
                                            encoding='gbk', key_for_N=False, key_for_N_type='list')

    # paths = dop.get_path_with_filter('./2.Middle_Data/202207', [-12, -4], '>=', start_time)
    # paths.extend(dop.get_path_with_filter('./2.Middle_Data/202208', [-12, -4], '<', end_time))
    # datas = []
    # for i, data_path in enumerate(paths):
    #     date = data_path[-12:-4]
    #     data = pd.read_csv(data_path)
    #     if '类型' in list(data.columns.values):
    #         data = data.drop(['类型'], axis=1)
    #     data[['入口ID', '出口ID', '入口HEX码']] = data[['入口ID', '出口ID', '入口HEX码']].fillna('')
    #     if i == (len(paths) - 1):
    #         origin_data = dt.compute_num_of_col(data, 'PASSID', ['middle_type'], key_data=origin_data,
    #                                             add_data=date)
    #     try:
    #         data = data.set_index('PASSID')
    #         datas.append(data)
    #     except:
    #         try:
    #             data = data.set_index('index')
    #             datas.append(data)
    #         except:
    #             data = data.set_index('Unnamed: 0')
    #             datas.append(data)
    # # 开始进行数据合并
    # data_combine = dt.concat_middle_data_noiden(datas)
    # data_combine = data_combine.reset_index()
    #
    # data_index = data_combine['PASSID'].values
    # data_content = data_combine[['收费单元路径', 'PASSID', '入口ID', '出口ID', '门架路径', '门架时间串', '出口车型', '入口车型']].values
    # data_combine = {}
    # for i in range(len(data_index)):
    #     data_combine[data_index[i]] = data_content[i]

    for gantry in gantrys:
        # get the last gantryid
        gantry_last = standard_back_dict[gantry]
        gantry_last = set(gantry_last).intersection(set(gantrys))
        if len(gantry_last) == 0:
            gantry_last = ''
        else:
            gantry_last = gantry_last.pop()

        # get the next gantryid
        gantry_next = standard_dict[gantry]
        gantry_next = set(gantry_next).intersection(set(gantrys))
        if len(gantry_next) == 0:
            gantry_next = ''
        else:
            gantry_next = gantry_next.pop()

        # get the gantry data path list
        paths = dop.path_of_holder_document('./1.Gantry_Data/202207/gantry_path/', True)
        paths.extend(dop.path_of_holder_document('./1.Gantry_Data/202208/gantry_path/', True))

        # get the last gantry flow
        if gantry_last == '':
            pass
        else:
            passid_dict_last = {}
            for path in paths:
                date = path.rsplit('_', 1)[1][:-4]
                # date = path.rsplit('/', 1)[1][:-4]
                # 判断该文件时间是否在所需范围内
                if start_time <= date <= end_time:
                    print(path)
                    with open(path) as f:
                        for i, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if i == 0:
                                get_index = dbf.get_index_of_list(row, ['TOLLINTERVALID', 'PASSID'])
                            else:
                                if gantry_last in row[get_index[0]]:
                                    try:
                                        passid_dict_last[row[get_index[1]]] += 1
                                    except:
                                        passid_dict_last[row[get_index[1]]] = 1
                            # if i > 1000:
                            #     break

        # get the last gantry passid flow
        if gantry_next == '':
            pass
        else:
            passid_dict_next = {}
            for path in paths:
                date = path.rsplit('_', 1)[1][:-4]
                # date = path.rsplit('/', 1)[1][:-4]
                # 判断该文件时间是否在所需范围内
                if start_time <= date <= end_time:
                    print(path)
                    with open(path) as f:
                        for i, row in enumerate(f):
                            row = row.split(',')
                            row[-1] = row[-1][:-1]
                            if i == 0:
                                get_index = dbf.get_index_of_list(row,
                                                                   ['TOLLINTERVALID', 'PASSID'])
                            else:
                                if gantry_next in row[get_index[0]]:
                                    try:
                                        passid_dict_next[row[get_index[1]]] += 1
                                    except:
                                        passid_dict_next[row[get_index[1]]] = 1
                            # if i > 1000:
                            #     break

        # get this gantry passid flow
        paths_list = []
        for path in paths:
            date = path.rsplit('_', 1)[1][:-4]
            # date = path.rsplit('/', 1)[1][:-4]
            # 判断该文件时间是否在所需范围内
            if start_time <= date <= end_time:
                paths_list.append(path)
                # print(path)
                # with open(path) as f:
                #     for i, row in enumerate(f):
                #         row = row.split(',')
                #         row[-1] = row[-1][:-1]
                #         if i == 0:
                #             get_index = dbf.get_indexs_of_list(row, ['TOLLINTERVALID', 'PASSID', 'GANTRYID', 'TRANSTIME'])
                #         else:
                #             if 'G003061003000820' in row[get_index[0]] and row[get_index[3]] > '2022-07-31 17:10:00':
                #                 passid_dict_2[row[get_index[1]]] = [row[get_index[0]], row[get_index[2]], row[get_index[3]]]
                # if i >1000:
                #     break

        passid_dict_this = pc.get_real_flow_with_time_revise(paths_list, True, 'dict', [gantry])

        # standard_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv', ['ENROADNODEID', 'EXROADNODEID'],
        #                                            encoding='gbk', key_for_N=True, key_for_N_type='list')
        # # 获取标准路径的反向字典数据
        # standard_back_dict = dbf.get_disc_from_document('../Data_Origin/tom_noderelation.csv',
        #                                                 ['EXROADNODEID', 'ENROADNODEID'],
        #                                                 encoding='gbk', key_for_N=True, key_for_N_type='list')
        # compare the passid with middle data

        # for key in passid_dict_this.keys():

        # num = 0
        # for path in paths:
        #     print('middle:' + path)
        #     with open(path) as f:
        #         for i, row in enumerate(f):
        #             row = row.split(',')
        #             row[-1] = row[-1][:-1]
        #             if i == 0:
        #                 get_index = dbf.get_indexs_of_list(row, ['收费单元路径', 'PASSID', '入口ID', '出口ID', '门架路径', '门架时间串', '出口车型', '入口车型'])
        #                 col_name = ['收费单元路径', 'PASSID', '入口ID', '出口ID', '门架路径', '门架时间串', '出口车型', '入口车型',
        #                             'GANTRYID', 'TRANSTIME', 'TOLLINTERVALID', 'VEHICLETYPE']
        #                 if gantry_last != '':
        #                     col_name.append('gantry_last')
        #                 if gantry_next != '':
        #                     col_name.append('gantry_next')
        #                 passid_list.append(col_name)
        #             else:
        #                 try:
        #                     b = passid_dict_this[row[get_index[1]]]
        #                     num += 1
        #                     list_ls = dbf.get_values_of_list(row, get_index)
        #                     list_ls.extend(b)
        #                     if gantry_last == '':
        #                         pass
        #                     else:
        #                         try:
        #                             a = passid_dict_last[row[get_index[1]]]
        #                             list_ls.append(1)
        #                         except:
        #                             list_ls.append(0)
        #                     if gantry_next == '':
        #                         pass
        #                     else:
        #                         try:
        #                             c = passid_dict_next[row[get_index[1]]]
        #                             list_ls.append(1)
        #                         except:
        #                             list_ls.append(0)
        #                     passid_list.append(list_ls)
        #                             # list_ls.append(dbf.get_standard_data(row[get_index[0]].split('|'), standard_dict, standard_back_dict, row[get_index[2]], row[get_index[3]]))
        #                 except:
        #                     pass

        # passid_list = [['收费单元路径', 'PASSID', '入口ID', '出口ID', '门架路径', '门架时间串', '出口车型', '入口车型', 'GANTRYID', 'TRANSTIME',
        #                 'TOLLINTERVALID', 'VEHICLETYPE']]
        passid_list = [['GANTRYID', 'TRANSTIME', 'TOLLINTERVALID', 'VEHICLETYPE', 'enHEX']]
        if gantry_last != '':
            passid_list[0].append('gantry_last')
        if gantry_next != '':
            passid_list[0].append('gantry_next')
        last_loss_num = 0
        for key in passid_dict_this.keys():
            b = passid_dict_this[key]
            # list_ls = data_combine[key]
            list_ls = []
            list_ls.extend(b)
            if gantry_last == '':
                pass
            else:
                # if list_ls[2] != station_in_dict[gantry]:
                try:
                    aa = station_id[list_ls[4]]
                    if station_id[list_ls[4]] != station_in_dict[gantry]:
                        try:
                            a = passid_dict_last[key]
                            list_ls.append(1)
                        except:
                            last_loss_num += 1
                            list_ls.append(0)
                    else:
                        list_ls.append(1)
                except:
                    list_ls.append(1)
            if gantry_next == '':
                pass
            else:
                try:
                    c = passid_dict_next[key]
                    list_ls.append(1)
                except:
                    # last_loss_num += 1
                    list_ls.append(0)
            passid_list.append(list_ls)
        print(gantry)
        print('total:{}, loss:{}'.format(len(passid_list), last_loss_num))
        with open('./4.data_check/abnormal_flow_data/' + gantry + '_flow.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(passid_list)


'''
    创建时间:2022/8/30
    完成时间:2022/8/30
    功能: 计算各收费单元间的各车型车速情况
'''


def get_speed_of_intervals(start_time, end_time, ifMonth=False):
    """
    计算所有收费单元间的各车型车速情况
    :param start_time: 所用数据的开始时间点
    :param end_time: 所用数据的截止时间点
    :return:
    """
    if ifMonth:
        month = ['202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204',
                 '202205', '202206', '202207', '202208']
    # 获取各收费单元间的标准长度
    gantry_service_list = dbf.get_disc_from_document('../Data_Origin/gantry_service.csv',
                                                     ['GANTRYID', 'BEFOREGANTRYID', 'DISTANCE'],
                                                     encoding='gbk', length=16, key_length=2)
    # 获得所有中间特征数据的地址
    fold_path, middle_paths = dop.get_all_file_from_dir(kp.get_parameter_with_keyword('middle_manyDay_back_path'))

    for m in month:
        path_list = []
        for path in middle_paths:
            if path[-12:-6] == m:
                path_list.append(path)

        paths_devide = dop.cut_dir_list(path_list, 3, sort=True)

        for kpath in paths_devide.keys():
            # 遍历所有中间特征数据，只针对时间点内的数据进行处理
            interval_speed = {}  # 用于保存各收费单元各车型车速的数据
            for path in paths_devide[kpath]:
                if ifMonth:
                    pass
                else:
                    # 判断是否在时间点范围内
                    if start_time > path[-12:-4] or path[-12:-4] > end_time:
                        continue
                # 如果在进行数据读取
                with open(path) as f:
                    print(path)
                    for j, row in enumerate(f):
                        row = row.split(',')
                        row[-1] = row[-1][:-1]
                        if j == 0:
                            # 将门架路径和门架时间串字段的索引获取到
                            index = dbf.get_index_of_list(row, ['门架路径', '门架时间串', '入口车型', '出口车型'])
                        else:
                            # 遍历每一个门架路径，并判断此门架和上一个门架是否是临近关系
                            # 获取门架路径值
                            gantry_value = dbf.get_values_of_list(row, [index[0]])
                            # 获取门架时间串值
                            time_value = dbf.get_values_of_list(row, [index[1]])
                            # 获取该车辆车型信息
                            if dbf.get_values_of_list(row, [index[3]])[0] == '':
                                vehicle_type = dbf.get_values_of_list(row, [index[2]])[0]
                            else:
                                vehicle_type = dbf.get_values_of_list(row, [index[3]])[0]
                            # 门架拆分
                            gantry_list = gantry_value[0].split('|')
                            time_list = time_value[0].split('|')
                            for i in range(len(gantry_list)):
                                if (i + 1) >= len(gantry_list):
                                    break
                                else:
                                    # 判断此门架和上一个门架是否是临近关系
                                    try:
                                        distance = gantry_service_list[gantry_list[i + 1] + '-' + gantry_list[i]]
                                        # 如果是临近关系，就进行速度计算并保存
                                        intervel = dbf.compute_intervel_of_two_time(time_list[i], time_list[i + 1],
                                                                                    'hour')
                                        speed = round(float(distance) / intervel, 2)
                                        if '0' < time_list[i + 1][-4] < '5':
                                            this_time_new = time_list[i + 1][:-4] + '0:00'
                                        else:
                                            this_time_new = time_list[i + 1][:-4] + '5:00'
                                        try:
                                            interval_speed[gantry_list[i + 1] + '_' + this_time_new + '_' + time_list[
                                                i + 1] + '_' + vehicle_type].append(speed)
                                        except:
                                            interval_speed[gantry_list[i + 1] + '_' + this_time_new + '_' + time_list[
                                                i + 1] + '_' + vehicle_type] = [speed]
                                    except:
                                        continue
            # 将字典类型转换为数组，进行保存
            dbf.basic_save_dict_data(interval_speed,
                                     './4.statistic_data/basic_data/speed_origin_data_time_total_' + m + '_' + str(
                                         kpath) + '.csv', True, True)

    # # 进行每个门架速度集合数据的计算和保存
    # feature_value = get_statistics_value_from_document(False, key_value=interval_speed, save_name='./4.statistic_data/basic_data/gantry_statistic_speed.csv', if_return=True)

    # # 计算各门架各车型的阈值速度
    # threshold_dict = get_threshold_from_data(feature_value, 'statistic')
    # dbf.basic_save_dict_data(threshold_dict, './4.statistic_data/basic_data/gantry_speed_threshold.csv')


'''
    创建时间:2022/8/31
    完成时间:2022/8/31
    功能: 根据大数据进行阈值求取
    关键字：阈值计算
    修改时间：
'''


def get_threshold_from_data(data, treat_type='statistic'):
    """
    根据大数据进行阈值求取
    :param treat_type: 阈值求取的方法，目前有：1.针对分布参数进行阈值计算；2.针对原始数据通过可信度计算阈值
    :param data: 为字典类型数据
    :return:
    """
    threshold_dict = {}  # 用于保存各目标的阈值
    if treat_type == 'statistic':
        # 针对分布参数进行阈值计算
        for key in data.keys():
            # 将四分之一分位的值设为阈值
            threshold_dict[key] = round(data[key][1] * 0.6, 1)

    else:
        # 针对原始数据进行阈值计算
        pass

    return threshold_dict


'''
    创建时间：2022/9/15
    完成时间：2022/9/15
    功能：进行
    关键词：拥堵等级，拥堵判定
    修改时间：
'''

'''
    创建时间：2022/9/19
    完成时间：2022/9/19
    功能：进行反馈数组内容的统计
    关键词：
    修改时间：
'''


def statistic_keyword(data, split_sign):
    """
    进行反馈数组内容的统计
    :param list data: 字符串数组
    :return:
    """
    data_dict = {}
    result_data = []
    for i in range(len(data)):
        data_list = data[i].split(split_sign)
        for j in range(len(data_list)):
            data_dict[data_list[j]] = 1
    for key in data_dict.keys():
        result_data.append([key])
    with open('./statistic.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)


'''
    创建时间：2022/10/21
    完成时间：2022/10/21
    功能：每种车型的车辆正常行驶时通过西汉高速全段的通行时间
    关键词：西汉高速、通行时间
    修改时间：
'''


def get_time_of_eachType_pass_XH(start_time, end_time):
    """
    每种车型的车辆正常行驶时通过西汉高速全段的通行时间
    :param start_time: 对象数据的开始时间
    :param end_time: 对象数据的结束时间
    :return:
    """
    # 获取各高速的起止点门架ID字典
    start_end = kp.get_parameter_with_keyword('start_end_interval')

    # 获取西汉高速的起止门架ID
    start_end_XH = start_end['XH']

    # 获取目标数据地址集
    paths = dop.path_of_holder_document('./3.check_result/partTime_result_data/', True)

    # 将符合时间范围的地址进行保留
    path_list = []
    for path in paths:
        if path[-12:-4] >= start_time and path[-12:-4] <= end_time:
            path_list.append(path)

    # 获取所有完成经过西宝高速的车辆信息
    data = get_data_some_col_have_save_value(path_list, ['门架路径'], [start_end_XH[0], start_end_XH[1]], [], ifSave=False,
                                             save_columns=['入口车型', '出口车型', '门架路径', '门架时间串'])

    # 计算每次行驶的西宝通行时间
    data_result = []
    for i in range(len(data)):
        if i > 0:
            if data[i][1] != '':
                vehicle_type = data[i][1]
            else:
                vehicle_type = data[i][0]
            gantry_list = data[i][2].split('|')
            gantry_list = [gantry[:-2] for gantry in gantry_list]
            index_list = dbf.get_index_of_list(gantry_list, start_end_XH)
            # 获取对应的时间点
            time_list = data[i][3].split('|')
            if index_list[0] > index_list[1]:
                # 此情况下，为西安方向
                time_inout = dbf.get_values_of_list(time_list, index_list)
                # 计算时间差
                time_gap = round(dbf.compute_intervel_of_two_time(time_inout[1], time_inout[0], 'hour'), 2)
                data_result.append([time_inout[1][:10], vehicle_type, 'to_XiAn', time_gap])
            else:
                # 此情况下，为汉中方向
                time_inout = dbf.get_values_of_list(time_list, index_list)
                # 计算时间差
                time_gap = round(dbf.compute_intervel_of_two_time(time_inout[0], time_inout[1], 'hour'), 2)
                data_result.append([time_inout[1][:10], vehicle_type, 'to_HanZhong', time_gap])

    # 按照车型、通行方向、通行时间进行保存
    with open('./4.statistic_data/vehicle_time_of_XH.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_result)


'''
    创建时间：2022/10/21
    完成时间：2022/10/21
    功能：计算一定时间内的各门架平均车速
    关键词：西汉高速、通行时间
    修改时间：
'''


def statistic_speed_flow_rate_with_time(statistic_time):
    """
    计算一定时间内的各门架平均车速
    :param statistic_time: 进行统计的采样单位
    :return:
    """
    speed_path = dop.path_of_holder_document('./4.statistic_data/basic_data/speed_basic_data/')
    speed_dict = {}
    # for path in speed_path:
    #     with open(path) as f:
    #         for i, row in enumerate(f):
    #             row = row.split(',')
    #             row[-1] = row[-1][:-1]
    #             if statistic_time == 'month':
    #                 try:
    #                     speed_dict[row[0] + '_' + row[2][:7] + '_' + row[1]].append(float(row[3]))
    #                 except:
    #                     speed_dict[row[0] + '_' + row[2][:7] + '_' + row[1]] = [float(row[3])]
    #             elif statistic_time == 'total':
    #                 try:
    #                     speed_dict[row[0] + '_' + row[1]].append(float(row[3]))
    #                 except:
    #                     speed_dict[row[0] + '_' + row[1]] = [float(row[3])]
    #
    # for key in speed_dict.keys():
    #     speed_dict[key] = round(sum(speed_dict[key]) / len(speed_dict[key]), 2)

    flow_path = dop.path_of_holder_document('./4.statistic_data/basic_data/flow_basic_data/', True)
    flow_dict = {}
    for path in flow_path:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if statistic_time == 'month':
                    try:
                        flow_dict[row[0] + '_' + row[1][:7] + '_' + row[2]] += float(row[3])
                    except:
                        flow_dict[row[0] + '_' + row[1][:7] + '_' + row[2]] = float(row[3])
                elif statistic_time == 'total':
                    try:
                        flow_dict[row[0] + '_' + row[2]] += float(row[3])
                    except:
                        flow_dict[row[0] + '_' + row[2]] = float(row[3])

    dbf.basic_save_dict_data(speed_dict, './4.statistic_data/statistic_result/20221025/speed_every_month.csv',
                             if_many_key=True)
    dbf.basic_save_dict_data(flow_dict, './4.statistic_data/statistic_result/20221025/flow_every_month.csv',
                             if_many_key=True)


'''
    创建时间：2022/10/24
    完成时间：2022/10/24
    功能：计算每一个时刻前N个时间段的各车型流量和某车型以上车辆的占比
    关键词：前N时间段、某车型以上、各车型流量
    修改时间：
'''


def get_preN_time_flow_and_rate(origin_path, pre_num, type_threshold):
    """
    计算每一个时刻前N个时间段的各车型流量和某车型以上车辆的占比
    :param origin_path: 待处理数据的地址
    :param pre_num: 具体处理前几个时间段的参数
    :param type_threshold: 统计车型占比的阈值，当车型大于这个阈值时，才进行占比统计
    :return:
    """
    data = dbf.get_disc_from_document(origin_path, [0, 1, 2, 3], encoding='utf-8', key_for_N=False, key_length=3,
                                      ifIndex=False, key_for_N_type='list', sign='_')
    # 计算各时刻的总流量
    data_total = dbf.compute_dict_by_group(data, [0, 1], 'sum', '_')

    # 计算各时刻的大车流量
    data_total_13 = dbf.compute_dict_by_group(data, [0, 1], 'sum', '_', [2], [type_threshold])

    # 遍历各门架车型，计算每条记录的训练数据
    data_rate = []  # 用于保存前几个时刻的大车占有率
    data_sample = []  # 用于保存训练数据
    for key in data.keys():
        data_sample_ls = []  # 用于保存单独一条的数据
        data_rate_ls = []  # 用于保存大车占有率单独一条的数据
        key_list = key.split('_')
        data_sample_ls.extend(key_list)
        data_rate_ls.extend(key_list)
        # 将前4个小时的所有流量数据获取到
        this_time = datetime.datetime.strptime(key_list[1], '%Y-%m-%d %H:%M:%S')
        for i in range(pre_num):
            new_time = this_time + datetime.timedelta(minutes=(5 * (i + 1) * (-1)))
            # 拼成前时刻的key值
            new_key = key_list[0] + '_' + datetime.datetime.strftime(new_time, '%Y-%m-%d %H:%M:%S') + '_' + \
                      key_list[2]
            new_rate_key = key_list[0] + '_' + datetime.datetime.strftime(new_time, '%Y-%m-%d %H:%M:%S')

            # 获取之前时刻的各车型流量数据
            try:
                new_flow = data[new_key]
                data_sample_ls.append(new_flow)
            except:
                data_sample_ls.append(0)

            # 获取之前时刻的总流量数据
            try:
                new_flow = data_total[new_rate_key]
                data_rate_ls.append(new_flow)
            except:
                data_rate_ls.append(0)

            # 获取之前时刻的大车流量数据
            try:
                new_flow = data_total_13[new_rate_key]
                data_rate_ls.append(new_flow)
            except:
                data_rate_ls.append(0)

            if data_rate_ls[-2] == 0:
                data_rate_ls.append(0)
            else:
                data_rate_ls.append(round(data_rate_ls[-1] / data_rate_ls[-2], 3))
        # 保存下来每一条数据
        data_rate.append(data_rate_ls)
        data_sample.append(data_sample_ls)

    # 计算某车型以上车辆的占比
    with open('./4.statistic_data/statistic_result/20221025/preTime_flow_total_202207.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_sample)
    with open('./4.statistic_data/statistic_result/20221025/preTime_flow_rate_202207.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_rate)


'''
    创建时间：2022/10/24
    完成时间：2022/10/24
    功能：进行速度的每一个时刻前N个时间段
    关键词：前N时间段、某车型以上、各车型流量
    修改时间：
'''


def get_preN_time_speed(origin_path, pre_num):
    """

    :param origin_path:
    :param pre_num:
    :return:
    """
    data = dbf.get_disc_from_document(origin_path, [0, 1, 2, 3], encoding='utf-8', key_for_N=False, key_length=3,
                                      ifIndex=False, key_for_N_type='list', sign='_')
    data_avg = dbf.compute_dict_by_group(data, [], 'avg', '')
    data_min = dbf.compute_dict_by_group(data, [], 'min', '')
    data_max = dbf.compute_dict_by_group(data, [], 'max', '')

    # 获取各车型的各时刻的速度平均值、最大值、最小值
    data = {}
    for key in data_avg.keys():
        data[key] = [data_avg[key], data_min[key], data_max[key]]

    data_sample = []  # 用于保存训练数据
    for key in data.keys():
        data_sample_ls = []  # 用于保存单独一条的数据
        data_rate_ls = []  # 用于保存大车占有率单独一条的数据
        key_list = key.split('_')
        data_sample_ls.extend(key_list)
        data_rate_ls.extend(key_list)
        # 将前4个小时的所有流量数据获取到
        this_time = datetime.datetime.strptime(key_list[1], '%Y-%m-%d %H:%M:%S')
        for i in range(pre_num):
            new_time = this_time + datetime.timedelta(minutes=(5 * (i + 1) * (-1)))
            # 拼成前时刻的key值
            new_key = key_list[0] + '_' + datetime.datetime.strftime(new_time, '%Y-%m-%d %H:%M:%S') + '_' + \
                      key_list[2]

            # 获取之前时刻的各车型流量数据
            try:
                new_flow = data[new_key]
                data_sample_ls.append(new_flow)
            except:
                data_sample_ls.append([0, 0, 0])

        data_sample.append(data_sample_ls)

    with open('./', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


'''
    创建时间：2022/10/24
    完成时间：2022/10/24
    功能：进行速度方差的每一个时刻前N个时间段
    关键词：前N时间段、速度方差
    修改时间：
'''


def get_preN_time_speed_variance(origin_path, pre_num):
    """
    进行速度方差的每一个时刻前N个时间段
    :param origin_path:
    :param pre_num:
    :return:
    """
    # 生成每个门架每个时刻对应的时间和速度数组
    data_time = dbf.get_disc_from_document(origin_path, [0, 1, 2], encoding='utf-8', key_for_N=False, key_length=2,
                                           ifIndex=False, key_for_N_type='list', sign='_')
    data_speed = dbf.get_disc_from_document(origin_path, [0, 1, 4], encoding='utf-8', key_for_N=False, key_length=2,
                                            ifIndex=False, key_for_N_type='list', sign='_')

    # 对每个门架每个时刻的时间和速度进行排序
    for key in data_speed.keys():
        # 通过时间将速度值进行排序
        dbf.basic_quick_sort(data_time[key], 0, len(data_time[key]), data_speed[key])

        # 进行速度差计算
        speed_list = []  # 用于保存每个门架每个时刻的速度差
        for i in range(len(data_speed[key])):
            if i > 0:
                speed_list.append(float(data_speed[key][i]) - float(data_speed[key][i - 1]))

        # 进行速度的方差计算
        data_speed[key] = np.var(speed_list)

    # 进行数据的保存
    dbf.basic_save_dict_data(data_speed)


'''
    创建时间：2022/10/25
    完成时间：2022/10/25
    功能：统计一段时间内的高疑似率车辆的总金额
    关键词：高疑似率、总金额
    修改时间：
'''


def statistic_money_of_high_core(start_time, end_time, core):
    """
    统计一段时间内的高疑似率车辆的总金额
    :param start_time: 目标数据开始时间
    :param end_time: 目标数据截止时间
    :param core: 设定阈值
    :return:
    """
    month = ['202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204',
             '202205', '202206', '202207', '202208']
    # 获取目标文件地址集
    paths = dop.path_of_holder_document('./3.check_result/partTime_result_data_gantry', True)

    # 遍历地址集，满足时间范围的进行读取操作，否则跳过
    money_list_statistic = []
    for m in month:
        path_list = []
        for path in paths:
            if path[-12:-6] == m:
                path_list.append(path)

        # 判断
        money_list = get_data_some_col_have_save_value(path_list, ['行驶风险评分'], [core], ['>='], ifIndex=False,
                                                       ifSave=False,
                                                       save_columns=['PASSID', '入口车型', '出口车型', 'pay_fee', '出口计费方式',
                                                                     '门架费用',
                                                                     'FEE'])
        # money_list = get_data_some_col_have_save_value(path_list, ['行驶风险评分'], [core], ['>='], ifIndex=False, ifSave=False)
        #
        money_dict_pay = {}
        money_dict_gantry = {}
        for i, mm in enumerate(money_list):
            if i > 0:
                if mm[2] != '':
                    vehicle_type = mm[2]
                else:
                    vehicle_type = mm[1]
                try:
                    money_dict_pay[vehicle_type].append(float(mm[3]))
                except:
                    money_dict_pay[vehicle_type] = [float(mm[3])]
                try:
                    money_dict_gantry[vehicle_type].append(float(mm[5]))
                except:
                    money_dict_gantry[vehicle_type] = [float(mm[5])]

        for key in money_dict_pay.keys():
            money_list_statistic.append(
                [m, key, sum(money_dict_pay[key]), sum(money_dict_gantry[key]), len(money_dict_gantry[key])])

    # 保存数据
    with open('./4.statistic_data/statistic_result/20221022/money_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(money_list)
    with open('./4.statistic_data/statistic_result/20221022/money_result_statistic.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(money_list_statistic)


'''
    创建时间：2022/10/25
    完成时间：2022/10/26
    功能：根据车牌和某一门架时间查询信息
    关键词：车牌、门架时间
    修改时间：
'''


def get_vehicle_data_with_plate_and_gantryTime(vehicle_plate, gantry_time):
    """
    根据车牌和某一门架时间查询信息
    :param vehicle_plate: 车牌号
    :param gantry_time: 经过门架时间
    :return:
    """
    result_data = []  # 用于保存所有车辆的稽查结果数据
    loss_data = []  # 用于保存所有车辆的缺失数据
    # 遍历所有的车牌号
    for plate in vehicle_plate:
        # 通过门架时间向后推5天的时间
        result_paths_list = []  # 保存退后5天的所有稽查结果文件地址
        loss_paths_list = []  # 保存退后5天的所有缺失数据文件地址
        for i in range(6):
            t = datetime.datetime.strftime(datetime.datetime.strptime(gantry_time, '%Y-%m-%d %H:%M:%S') +
                                           datetime.timedelta(days=5 * (6 - i - 1) * (-1)),
                                           '%Y-%m-%d %H:%M:%S')
            result_paths_list.append('.//' + t[:10].replace('-', '') + '.csv')
            loss_paths_list.append('.//' + t[:10].replace('-', '') + '.csv')

        # 查找所有时间的稽查结果文件与车牌匹配的记录
        result_data.extend(
            get_data_some_col_have_save_value(result_paths_list, ['车牌(全)'], [plate], ['='], ifSave=False))
        loss_data.extend(
            get_data_some_col_have_save_value(loss_paths_list, ['车牌(全)'], [plate], ['='], ifSave=False))

    with open('./4.statistic_data/statistic_result/20221026/abnormal_plate_result_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_data)
    with open('./4.statistic_data/statistic_result/20221026/abnormal_plate_loss_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(loss_data)


'''
    创建时间：2022/10/26
    完成时间：2022/10/26
    功能：获取未匹配到工单的高风险度车辆信息
    关键词：工单匹配、高风险度
    修改时间：
'''


def get_vehicle_highCore_without_check(start_time, end_time, core, JK_list_path):
    """
    获取未匹配到工单的高风险度车辆信息
    :param start_time: 目标数据开始时间
    :param end_time: 目标数据截止时间
    :param core: 设定阈值
    :return:
    """
    # 获取所有时间范围内的稽查结果地址
    paths = dop.path_of_holder_document('./3.check_result/partTime_result_data_gantry', True)

    # 遍历地址集，满足时间范围的进行读取操作，否则跳过
    path_list = []
    for path in paths:
        if start_time <= path[-12:-4] <= end_time:
            path_list.append(path)

    # 获取所有时间范围内的高风险车辆信息
    highCore_data = get_data_some_col_have_save_value(path_list, ['行驶风险评分'], [core], ['>='], ifIndex=False, ifSave=False)

    # 将高风险车辆信息转为字典类型
    highCore_dict = {}
    for i in range(len(highCore_data)):
        if i > 0:
            try:
                highCore_dict[highCore_data[i][1]].append(highCore_data[i])
            except:
                highCore_dict[highCore_data[i][1]] = highCore_data[i]

    # 读取工单信息，并转为字典类型
    JK_list_dict = dbf.get_disc_from_document(JK_list_path, [''], key_for_N=True, ifIndex=True, key_for_N_type='list', sign='_')

    # 遍历高风险车牌
    for key in highCore_dict.keys():
        # 匹配工单车牌，如果匹配不上，保存所有记录信息
        try:
            inout_time = JK_list_dict[key]
        except:
            d = 1


'''
    创建时间:2022/6/13
    完成时间:2022/6/13
    功能:提取选定列中包含某些元素的数据（针对多文件）
    关键字：匹配数据
    重要程度：****
    修改内容：No.1 2022/9/15，增加了各字段值和目标值之间的比较符号compare_sign
'''


def get_data_some_col_have_save_value(paths, column_name, value_name, compare_sign=[], charge_type='and', save_tail=0,
                                      ifIndex=False, ifSave=True, save_columns=[], encoding='utf-8', save_combine=False):
    """
    提取选定列中包含某些元素的数据（针对多文件）
    :param ifIndex: 如果需要对判断的字段进行判断
    :param list compare_sign: 各字段值和目标值之间的比较符号
    :param list paths: 操作文件的地址
    :param list column_name: 进行操作的列名称
    :param list value_name: 进行判断的值
    :param charge_type: 参与判断的值的相互关系，and为同时包含，or为任一包含
    :return:
    """
    data = []  # 用于保存满足条件的数据
    col_save = 0  # 用于标识是否标题行已经保存
    for k, path in enumerate(paths):
        print(path)
        with open(path, encoding=encoding) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:  # 如果是第一行，获取到对比列的索引
                    if len(save_columns) != 0:
                        save_index = dbf.get_index_of_list(row, save_columns)
                    col_index = dbf.get_index_of_list(row, column_name)
                    if col_save == 0:  # 如果标题行还没有进行保存，则进行保存
                        if len(save_columns) != 0:
                            data.append(save_columns)
                        else:
                            data.append(row)
                        col_save += 1
                else:
                    charge_result = 0  # 判断是否满足条件的结果
                    # 如果是and，则所有值都有包含才可以进行保存
                    if charge_type == 'and':
                        for j in range(len(value_name)):
                            # 如果该值存在，则charge_result加1
                            if len(compare_sign) == 0:
                                if value_name[j] in row[col_index[0]]:
                                    charge_result += 1
                            else:
                                if compare_sign[j] == '>':
                                    if row[col_index[j]] > value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == '>=':
                                    if type(value_name[j]) == str:
                                        if row[col_index[j]] >= value_name[j]:
                                            charge_result += 1
                                    else:
                                        if float(row[col_index[j]]) >= value_name[j]:
                                            charge_result += 1
                                elif compare_sign[j] == '=':
                                    if row[col_index[j]] == value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == '<':
                                    if row[col_index[j]] < value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == '<=':
                                    if row[col_index[j]] <= value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == 'in':
                                    if row[col_index[j]] in value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == 'not in':
                                    if row[col_index[j]] not in value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == '<>':
                                    if row[col_index[j]] != value_name[j]:
                                        charge_result += 1
                                elif compare_sign[j] == 'dict_y':
                                    try:
                                        a = value_name[j][row[col_index[j]]]
                                        charge_result += 1
                                    except:
                                        pass
                                elif compare_sign[j] == 'dict_n':
                                    try:
                                        a = value_name[j][row[col_index[j]]]
                                    except:
                                        charge_result += 1

                        # 如果charge_result数值与value_name长度一致，表示所有值均有包含，即进行保存
                        if charge_result == len(value_name):
                            if ifIndex:
                                gantry_list = row[col_index[0]].split('|')
                                gantry_list = [x[:-2] for x in gantry_list]
                                if gantry_list.index(value_name[0]) < gantry_list.index(value_name[1]):
                                    if len(save_columns) == 0:
                                        data.append(row)
                                    else:
                                        data.append(dbf.get_values_of_list(row, save_index))
                            else:
                                if len(save_columns) == 0:
                                    if save_combine:
                                        data.append(row)
                                    else:
                                        data.extend(row)
                                else:
                                    if save_combine:
                                        data.append(dbf.get_values_of_list(row, save_index))
                                    else:
                                        data.extend(dbf.get_values_of_list(row, save_index))
                    # 如果不是and，则所有值有任一包含，即进行保存
                    else:
                        for j in value_name:
                            # 如果该值存在，则charge_result加1
                            if value_name[j] in row[col_index[0]]:
                                charge_result += 1
                        # 如果charge_result数值大于0，表示所有值有任一有包含，即进行保存
                        if charge_result > 0:
                            data.append(row)

    if ifSave:
        # 保存过滤后的数据
        # 获取数据保存的地址
        have_some_value_path = kp.get_parameter_with_keyword('have_some_value_path')
        if save_tail != 0:
            have_some_value_path = have_some_value_path.rsplit('.', 1)[0] + str(save_tail) + '.csv'
        with open(have_some_value_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    else:
        return data


if __name__ == '__main__':
    # statistic_feature('./3.loss_data/November/loss_origin_data/loss_data_202111pro.csv', [['time', '门架端口类型']], 'time_duan')
    # statistic_feature('./3.loss_data/November/loss_origin_data/loss_data_202111_only_in.csv', ['入口车型'], '1')

    # process the loss_data,that get the only_in/only_out/none/pro/mistake
    # ls_check_loss_data_with_pro('./3.loss_data/November/loss_origin_data/loss_data_202111.csv', 'origin')

    # process the loss_data_for_show,that get the only_in/only_out/none/pro/mistake
    # ls_check_loss_data_with_pro('./3.loss_data/November/loss_data_for_show/loss_data_202111_for_show.csv', 'show')

    # statistic for pro data
    # statistic_feature('./3.short_data/November/pro_data.csv', ['门架端口类型'], './4.data_check/November/statistic_data/', '门架端口类型')
    # statistic for whole data
    # statistic_feature('./3.short_data/November/whole_data.csv', [['出口车型', '出口通行介质']], './4.data_check/November/statistic_data/',
    #                   '出口车型_出口通行介质')

    # data2 = pd.read_csv('./3.short_data/pro_data.csv')
    # data1 = pd.read_csv('./3.short_data/whole_data.csv').drop(['Unnamed: 0'], axis=1)
    # result = path_of_whole_data_noiden(data1, data2)
    # result.to_csv('./3.short_data/all_path_of_whole_pro.csv')

    # compare_abnormal_and_normal()['川E49550', '京AMS127', '京AJU952', '宁B18626', '宁AK7993', '晋M57459', '鄂AZK797']
    # dd = pd.read_csv('./4.data_check/November/filter_data_for_GPS/filter_propass_Upath_data_time_out.csv')
    # dd['time'] = dd['门架时间串'].map(lambda x: x.split('|')[0][:10].replace('-', ''))
    # veh = dd['车牌'].values
    # time = dd['time'].values
    # test_result_with_origin_data(veh, './3.loss_data/abnormal_origin_data_not_match.csv',
    #                              time, 'gantry')

    # vehicle_list = ['冀AC2996','冀AC2996','冀AC2996','冀CC0317','冀CC0317','冀T35845','冀T35845','冀T35845','川B6BE62','川F80VR8','川F80VR8','晋A9EX62','晋A9EX62','晋A9EX62','晋C2787K','晋C84611','晋C84611','晋JRY229','晋JRY229','晋JRY229','晋JRY229','豫A515TR','豫A515TR','豫PC725C','豫PQ7862','豫PQ7862','豫PQ7862','贵JS1428','贵JS1428','鲁GX0G92','鲁GX0G92','鲁GX0G92','鲁GY9B25','鲁GY9B25','鲁GY9B25','鲁Q068DW','鲁Q068DW','鲁Q068DW']
    # vehicle_list = ['陕AU2S70','陕AV1M60','陕AX20P1','陕AXW822','陕BZ2078','陕D6998R','陕D9206X','陕HD6671','陕J488G5','陕J7P951','陕JZY168','陕K117G3','陕KB973M','陕KE697T','陕U66B15','陕URT185','陕UWR836']
    # time_list = ['20211120','20211119','20211117','20211115','20211119','20211122','20211117','20211119','20211119','20211120','20211121','20211119','20211118','20211119','20211120','20211117','20211120']
    # test_result_with_origin_data(['陕AR5536'], './4.data_check/November/data_for_check/abnormal_gantry_data_陕AR5536.csv', '', 'exit', result='check_result')
    # test_result_with_origin_data(['020000140102070064380820211117101155'], './4.data_check/November/data_for_check/abnormal_gantry_data_陕EF9772.csv',
    #                              ['20211117'], 'gantry_origin')
    # test_result_with_origin_data(
    #     ['川H83171_0', '川H83171_0', '川H83171_0'],
    #     './4.data_check/November/data_for_check/abnormal_gantry_data_陕EF9772.csv',
    #     ['20211119', '20211120', '20211121'], 'identify')

    # filter_data_with_feature('./3.check_result/November/inprovince/check_result_202111.csv', './3.short_data/November/whole_data.csv',
    #                          ['门架数占比'], [0.0], ['出口车牌', '门架路径', '门架时间串', '门架类型串', '出口车型'])

    # get the data we need to compare GPS ,the data from the whole data
    # filter_data_with_feature_new(None, None, ['出口车牌', '门架路径', '门架时间串', '门架类型串', '出口车型', 'SGROUP', '门架费用', 'pay_fee', '行驶时间'], all_get=True)

    # statistic_feature('./3.check_result/inprovince/check_result_202108_2_TOLLINTERVALID.csv', ['行驶风险判定'], 'danger')

    # get_information_from_Document('./1.Exit_Data/November/path_20211119.csv')

    #
    # get_all_data_of_vehicle(['陕J78076'])

    # get the data we need that pass the province
    # filter_data_with_feature_new(None, None, ['PASSID', '车牌', '门架路径', '门架时间串', '门架车型', '门架车种'])

    # get the data we need that pass the province,2022/2/9
    # filter_data_with_feature_new(None, None, ['PASSID', '车牌', '入口车牌(颜色)', '出口车牌(颜色)', '入口通行介质', '出口通行介质', '门架路径', '门架时间串', '入口车型', '出口车型', '门架车型', '入口ID', '入口时间', '出口ID', '出口时间'], in_path='./3.short_data/November/whole_data.csv', out_path='./4.data_check/November/filter_data_for_GPS/filter_vehicletype_16_14.csv')
    # filter_data_with_feature_new(None, None, 'all',
    #                              in_path='./3.short_data/November/whole_data.csv',
    #                              out_path='./4.data_check/November/filter_data_for_GPS/filter_vehicletype_96h.csv')
    # filter_data_with_feature_new(None, None, 'all',
    #                              in_path='./3.check_result/November/inprovince/check_result_202111_0210.csv',
    #                              out_path='./4.data_check/November/filter_data_for_GPS/filter_vehicletype_96h.csv')
    # vehicle = ["陕A0X7L1","陕A6HL97","陕UPC793","苏AZ2602","苏GF0566","浙A7U267","蒙KH0197","陕F29G56"]
    # filter_data_with_feature_new(None, vehicle, ['PASSID', '车牌', '门架路径', '门架时间串', '出口车型', '出口车种', 'pay_fee'], in_path='./3.short_data/November/whole_data.csv', out_path='./4.data_check/November/filter_data_for_GPS/filter_abnormal_different.csv')

    # check the enter data, 2022/2/11
    # simulate_check_data_after_modify_code('./1.Enter_Data/November/old_20220223/Enter_path/enter_path_20211108.csv', './1.Enter_Data/November/enter_path_20211108.csv')
    # simulate_check_data_after_modify_code('./1.Exit_Data/November/old_20220223/exit_path/exit_path_20211128.csv',
    #                                       './1.Exit_Data/November/exit_path_20211128.csv')
    # simulate_check_data_after_modify_code('./1.Gantry_Data/November/old_20220223/gantry/gantry_20211101.csv',
    #                                       './1.Gantry_Data/November/gantry_20211101.csv')
    # check the middle data
    # simulate_check_data_after_modify_code('./2.Middle_Data/November/main_middledata_20211118_2.csv',
    #                                       './2.Middle_Data/November/main_middledata_20211118.csv')
    # check the whole data
    # simulate_check_data_after_modify_code('./3.short_data/November/whole_data_0313.csv',
    #                                       './3.short_data/November/whole_data.csv')

    # check the loss data
    # simulate_check_data_after_modify_code('./3.loss_data/November/loss_origin_data/loss_data_202111_0415.csv',
    #                                       './3.loss_data/November/loss_origin_data/loss_data_202111_0425.csv')

    # check the LT_middle data
    # simulate_check_data_after_modify_code('./2.oneDay/back_data/LT_middle/2021-06-01_0530.csv',
    #                                       './2.oneDay/back_data/LT_middle/2021-06-01.csv', index_col='车牌(全)')
    # check the loss data
    # simulate_check_data_after_modify_code('./3.check_result/20211105 (copy).csv',
    #                                       './3.check_result/20211105.csv')

    # check the whole feature data
    # simulate_check_data_after_modify_code('./3.whole_data_and_feature/November/20220210/combine_data_2021110.csv',
    #                                       './3.whole_data_and_feature/November/combine_data_2021110.csv')

    # check the check result data
    # simulate_check_data_after_modify_code('./3.check_result/November/inprovince/check_result_202111.csv',
    #                                       './3.check_result/November/inprovince/check_result_202111_0415.csv')

    # check the check result data
    # simulate_check_data_after_modify_code('./1.Gantry_Data/202108/gantry_path/path_20210802.csv',
    #                                       './1.Gantry_Data/202108/gantry_path/path_20210802 (copy).csv')

    # data_whole = pd.read_csv('./3.short_data/November/whole_data.csv')
    # data_pro = pd.read_csv('./3.short_data/November/pro_data.csv')
    # data_station = pd.read_csv('../Data_Origin/station_OD.csv').drop(['Unnamed: 0'], axis=1)
    # data_gantry = pd.read_csv('../Data_Origin/gantry_OD_new.csv').drop(['Unnamed: 0'], axis=1)
    # data_result = analysis_for_OD(data_whole, data_pro, data_station, data_gantry)
    # data_result.to_csv('./4.data_check/November/data_OD.csv', encoding='gbk')

    # filter_data_ls()

    # get part of the data
    # get_part_of_data('./2.Middle_Data/November/main_middledata_20211101.csv')

    # data_whole_many_path = kp.get_parameter_with_keyword('data_whole_many_path')
    # cut_path = dop.cut_dir_list(data_whole_many_path, 9)
    # dop.run_with_multiple_threading(function=get_data_some_col_have_save_value,
    #                                 parameters=[(cut_path[i], ['收费单元路径'], ['G065W61003000720', 'G002261001000920'], 'and', i) for i in range(1, (len(cut_path) + 1))])
    # data_whole_many_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('data_whole_many_path'), True)
    # get_data_some_col_have_save_value(data_whole_many_path, ['收费单元路径'], ['G0005610050011', 'G0022610010019'], save_tail=1)

    # LT_result_data_many_path = kp.get_parameter_with_keyword('LT_result_data_many_path')
    # data = statistics_all_features(LT_result_data_many_path, feature='行驶风险评分', if_continuous=True, cut_type='cut', cut_num=20, if_max=False, MinMax=[0, 100])
    # with open('./4.data_check/score_distribute_LT.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)
    #
    # gantry_result_data_many_path = kp.get_parameter_with_keyword('gantry_result_data_many_path')
    # data = statistics_all_features(gantry_result_data_many_path, feature='行驶风险评分', if_continuous=True, cut_type='cut',
    #                                cut_num=20, if_max=False, MinMax=[0, 100])
    # with open('./4.data_check/score_distribute_gantry.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)

    # LT_result_data_many_path = kp.get_parameter_with_keyword('LT_oneDay_wash_back_path')
    # data = statistics_all_features(LT_result_data_many_path, feature=['出口省', '查验结果'])
    # with open('./4.data_check/frequency_cheat.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)

    # case_data_vehicle_path = kp.get_parameter_with_keyword('case_data_vehicle_path')
    # case_gantry_result_path = case_data_vehicle_path + '陕D3J879_0/gantry_result_陕D3J879_0.csv'
    # data = statistics_all_features(case_gantry_result_path, feature='超时门架区间串', if_continuous=False, value_split_sign='|')
    # # data = []
    # with open(case_data_vehicle_path + '陕D3J879_0/out_time_gantry_陕D3J879_0.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data)

    # portray_statistics_abnormal get
    # get_statistics_value_from_document('./4.poratry_data/history_data/vehicle_portray_total.csv',
    #                                    ['全程无门架记录频次', '路径缺失频次', 'U型行驶频次', 'J型行驶频次', '往复型行驶频次', '出口缺失频次', '费用异常频次', '最小费额计费频次', '出口轴数大于计费车型频次', '出入口车型不一致频次', '出入口车牌不一致频次', '行驶时长超过3天的频次'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], treat_value='总行驶次数')

    # portray_statistics_total get
    # get_statistics_value_from_document('./4.poratry_data/history_data/vehicle_portray_total.csv',
    #                                    ['全程无门架记录频次', '路径缺失频次', 'U型行驶频次', 'J型行驶频次', '往复型行驶频次', '出口缺失频次', '费用异常频次',
    #                                     '最小费额计费频次', '出口轴数大于计费车型频次', '出入口车型不一致频次', '出入口车牌不一致频次', '行驶时长超过3天的频次'],
    #                                    treat_value='总行驶次数')

    # get_loss_gantry_by_plate('陕D3J879_0')

    # ['陕AX20J0_0', '陕D3J879_0']
    # type = ['gantry_result', 'LT_result', 'gantry_portary', 'LT_portary', 'loss', 'LT_OD', 'LT_middle', 'whole_OD', 'whole_time']
    # for i in range(len(type)):
    #     get_data_with_vehicle_passid_time(['陕D3J879_0'], type[i], 'vehicle_plate', save_name='陕D3J879_0')
    #
    # gantry_result_data_many_path = dop.path_of_holder_document(kp.get_parameter_with_keyword('gantry_result_data_many_path'), True)
    #
    # max = 0
    # max_row = ''
    # for i, path in enumerate(gantry_result_data_many_path):
    #     with open(path) as f:
    #         for j, row in enumerate(f):
    #             row = row.split(',')
    #             row[-1] = row[-1][:-1]
    #             if j == 0:
    #                 index = dbf.get_indexs_of_list(row, ['行驶风险评分', '车牌(全)'])
    #             else:
    #                 if float(row[index[0]]) > max and '警' not in row[index[1]] and '陕000' not in row[index[1]]:
    #                     max = float(row[index[0]])
    #                     max_row = row
    #     if i > 10:
    #         break
    #
    # print(max_row)
    # fold_list, file_list = dop.get_all_file_from_dir('./1.Gantry_Data')
    # file_list_new = []
    # for i in range(len(file_list)):
    #     if 'path' not in file_list[i]:
    #         file_list_new.append(file_list[i])
    # compute_num_of_gantry_and_station(file_list_new, 'gantry', 5, '20211101', '20211130')

    # get_pass_last_no_next_passid('20220730', '20220808')

    # compute_direction_num_of_station('20210801', '20220818')

    get_speed_of_intervals('20220501', '20220814', ifMonth=True)

    # result_data = dop.get_data_in_document('./算法代码80.txt', 'def', ' ', '(', True)

    # statistic_keyword(result_data, '_')

    # statistic_money_of_high_core('20220701', '20220701', 80)
