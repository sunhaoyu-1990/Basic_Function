# coding=gbk
import datetime
import time

import pandas as pd

import GetData as gd
import Document_process as dop
import Keyword_and_Parameter as kp
from data import Data_Basic_Function as dbf

'''
    文档创建时间：2022/05/08
    1.收集昨日的绿通和收费数据，进行本地保存，保存地址通过Keyword_and_Parameter关键字管理库管理
    2.实现各结果数据的数据库上传
'''

'''
    创建时间：2022/05/08
    完成时间：2022/05/08
    功能：进行昨日的绿通和收费数据，进行本地保存
    修改时间：
'''


def get_yesterday_data():
    # 获取部中心和收费中心数据库参数
    LT_department_parameter = kp.get_parameter_with_keyword('LT_department_mysql')
    vehicle_charge_parameter = kp.get_parameter_with_keyword('vehicle_charge_mysql')
    # 创建操作部中心数据库的mysql对象
    db_LT = gd.OperationMysql(LT_department_parameter['host'], LT_department_parameter['port'],
                              LT_department_parameter['user'], LT_department_parameter['passwd'],
                              LT_department_parameter['db'])
    # 创建操作收费中心数据库的mysql对象
    db_charge = gd.OperationMysql(vehicle_charge_parameter['host'], vehicle_charge_parameter['port'],
                                  vehicle_charge_parameter['user'], vehicle_charge_parameter['passwd'],
                                  vehicle_charge_parameter['db'])
    # 获取昨日数据的起止时间
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    yesterday_time = datetime.datetime.strptime(time_now, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=-1)
    start_time = yesterday_time.strftime("%Y-%m-%d %H:%M:%S")[:-8] + '00:00:00'
    end_time = yesterday_time.strftime("%Y-%m-%d %H:%M:%S")[:-8] + '23:59:59'

    # 获取绿通和收费数据的存储位置信息
    LT_path = kp.get_parameter_with_keyword('LT_oneDay_save_path')
    gantry_path = kp.get_parameter_with_keyword('gantry_oneDay_save_path')
    gantry_path_pro = kp.get_parameter_with_keyword('gantry_oneDay_save_path_pro')
    entry_path = kp.get_parameter_with_keyword('in_oneDay_save_path')
    exit_path = kp.get_parameter_with_keyword('out_oneDay_save_path')
    iden_path = kp.get_parameter_with_keyword('iden_oneDay_save_path')

    # 获取绿通和收费数据的备注存储位置信息
    LT_back_path = kp.get_parameter_with_keyword('LT_oneDay_back_path')
    gantry_back_path = kp.get_parameter_with_keyword('gantry_oneDay_back_path')
    gantry_back_path_pro = kp.get_parameter_with_keyword('gantry_oneDay_back_path_pro')
    entry_back_path = kp.get_parameter_with_keyword('in_oneDay_back_path')
    exit_back_path = kp.get_parameter_with_keyword('out_oneDay_back_path')
    iden_back_path = kp.get_parameter_with_keyword('iden_oneDay_back_path')

    # 获取绿通和收费数据的sql语句
    LT_sql = kp.get_parameter_with_keyword('LT_features')
    gantry_sql = kp.get_parameter_with_keyword('charge_gantry_features')
    gantry_sql_pro = kp.get_parameter_with_keyword('charge_gantry_features_pro')
    entry_sql = kp.get_parameter_with_keyword('charge_in_features')
    exit_sql = kp.get_parameter_with_keyword('charge_out_features')
    iden_sql = kp.get_parameter_with_keyword('charge_iden_features')

    # 分别获取起止时间的各自数据库数据并进行保存
    db_LT.get_data_in_for(start_time, end_time, LT_path, save_type='oneData', get_type='LT', sql_str=LT_sql,
                          back_path=LT_back_path)
    # 门架数据获取及保存
    db_charge.get_data_in_for(start_time, end_time, gantry_path, save_type='oneData', get_type='gantry',
                              sql_str=gantry_sql, back_path=gantry_back_path)
    # 省界门架数据获取及保存
    db_charge.get_data_in_for(start_time, end_time, gantry_path_pro, save_type='oneData', get_type='gantry',
                              sql_str=gantry_sql_pro, back_path=gantry_back_path_pro)
    # 入口数据获取及保存
    db_charge.get_data_in_for(start_time, end_time, exit_path, save_type='oneData', get_type='entry',
                              sql_str=exit_sql, back_path=exit_back_path)
    # 出口数据获取及保存
    db_charge.get_data_in_for(start_time, end_time, entry_path, save_type='oneData', get_type='exit',
                              sql_str=entry_sql, back_path=entry_back_path)


'''
    创建时间：2022/2/21
    完成时间：2022/2/21
    功能：将稽查算法各阶段的数据写入数据库
    修改时间：No.1
'''


def save_data_to_Mysql(write_type, upload_data):
    """
    将稽查算法各阶段的数据写入数据库
    :param write_type:写入数据的类型，middle代表为中间数据写入，timePart代表一段时间合并处理的数据写入，checkFeature代表稽查特征处理表写入，checkResult代表稽查结果写入
    :param upload_data:需要写入的数据
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
                     upload_data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'datetime', 'datetime', 'float',
                      'float', 'float', 'float', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'float', 'float', 'float', 'float', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'float', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'float', 'int', 'datetime', 'datetime'])
    # 如果write_type为timePart, 则针对一段时间的合并数据进行保存
    elif write_type == 'timePart':
        db.write_one('partTimeConcat_data_20220511',
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
                     upload_data,
                     ['varchar', 'varchar', 'varchar', 'varchar', 'tinyint', 'varchar', 'varchar', 'datetime', 'int',
                      'int', 'int', 'int', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar',
                      'datetime', 'int', 'int', 'int', 'int', 'int', 'int', 'varchar', 'varchar', 'int',
                      'varchar', 'varchar', 'int', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar',
                      'int', 'varchar', 'int', 'datetime', 'datetime', 'tinyint', 'varchar', 'tinyint', 'varchar',
                      'varchar', 'datetime', 'float', 'varchar', 'varchar', 'varchar', 'varchar', 'datetime', 'varchar',
                      'varchar', 'varchar', 'datetime'])
    # 如果write_type为checkFeature, 则针对稽查特征处理后的数据进行保存
    elif write_type == 'checkFeature':
        db.write_one("checkFeature_data_20220511",
                     ["PASSID", "shortPathFee", "shortPathLength", "shortPath", "shortPathNum", "ifVehiclePlateMatch",
                      "ifVehiclePlateSame", "ifPlateColorSame", "ifOBUPlateSame", "ifETCPlateSame", "ifVehicleTypeSame",
                      "ifOBUTypeLarger", "ifUseShortFee", "ifVeTypeLargerAxle", "gantryNumRate", "ifGantryPathWhole",
                      "gantryPathIntegrity", "gantryPathMatch", "ifGrantryFeeSame", "ifShortFeeSame", "maxOutRangeTime",
                      "ifTimeOutRange", "maxOutRangeGantry", "outRangeTimePath", "outRangeGantryPath",
                      "outRangeSpeedPath", "totalTime", "ifShortTimeAbnormal", "shortOutRangeTime", "pathType"],
                     upload_data,
                     ['varchar', 'int', 'int', 'varchar', 'int', 'tinyint', 'tinyint', 'tinyint', 'tinyint',
                      'tinyint', 'tinyint', 'tinyint', 'tinyint', 'tinyint', 'float', 'tinyint', 'int', 'int',
                      'tinyint', 'tinyint', 'int', 'tinyint', 'varchar', 'varchar', 'varchar', 'varchar', 'int',
                      'tinyint', 'int', 'varchar'])
    # 如果write_type为checkResult, 则针对稽查结果数据进行保存
    elif write_type == 'checkResult':
        db.write_one('checkResult_data_20220511',
                     ["PASSID", "GantryPathAbnormal", "ifVehiclePlateMatchCode", "ifVehiclePlateSameCode",
                      "ifPlateColorSameCode",
                      "ifOBUPlateSameCode", "ifETCPlateSameCode", "ifVehicleTypeSameCode", "ifExAxleLargerCode",
                      "ifOBUTypeLargerCode", "ifPassTimeLarger3DaysCode", "ifVehicleTypeSameCode_filter",
                      "ifPassTimeAbnormalCode", "ifFeeMatchCode", "ifPathTypeAbnormalCode", "combineCode",
                      "abnormalType", "abnormalCore"],
                     upload_data,
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
                     upload_data,
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
                     upload_data,
                     ['varchar', 'varchar', 'varchar', 'int', 'int', 'float', 'float', 'float', 'float', 'float',
                      'float', 'varchar',
                      'datetime', 'int', 'int', 'int', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'int',
                      'int',
                      'varchar', 'int', 'int', 'int', 'int', 'varchar', 'int', 'varchar', 'varchar', 'varchar', 'int',
                      'datetime', 'int', 'varchar'])

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'totaldata':
        totaldata_LT_table_name = kp.get_parameter_with_keyword('totaldata_LT_table_name')
        totaldata_LT_table_features = kp.get_parameter_with_keyword('totaldata_LT_table_features')
        totaldata_LT_table_type = kp.get_parameter_with_keyword('totaldata_LT_table_type')
        db.write_one(totaldata_LT_table_name, totaldata_LT_table_features, upload_data, totaldata_LT_table_type)
    # 绿通OD次数原始数据上传数据库
    elif write_type == 'LT_OD':
        LT_OD_table_name = kp.get_parameter_with_keyword('LT_OD_table_name')
        LT_OD_table_features = kp.get_parameter_with_keyword('LT_OD_table_features')
        LT_OD_table_type = kp.get_parameter_with_keyword('LT_OD_table_type')
        db.write_one(LT_OD_table_name, LT_OD_table_features, upload_data, LT_OD_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'inoutTime':
        inout_time_table_name = kp.get_parameter_with_keyword('inout_time_table_name')
        inout_time_table_features = kp.get_parameter_with_keyword('inout_time_table_features')
        inout_time_table_type = kp.get_parameter_with_keyword('inout_time_table_type')
        db.write_one(inout_time_table_name, inout_time_table_features, upload_data, inout_time_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'gantry_OD':
        gantry_OD_table_name = kp.get_parameter_with_keyword('gantry_OD_table_name')
        gantry_OD_table_features = kp.get_parameter_with_keyword('gantry_OD_table_features')
        gantry_OD_table_type = kp.get_parameter_with_keyword('gantry_OD_table_type')
        db.write_one(gantry_OD_table_name, gantry_OD_table_features, upload_data, gantry_OD_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'LT_treated_OD':
        LT_OD_treated_table_name = kp.get_parameter_with_keyword('LT_OD_treated_table_name')
        LT_OD_treated_table_features = kp.get_parameter_with_keyword('LT_OD_treated_table_features')
        LT_OD_treated_table_type = kp.get_parameter_with_keyword('LT_OD_treated_table_type')
        db.write_one(LT_OD_treated_table_name, LT_OD_treated_table_features, upload_data, LT_OD_treated_table_type)

    # portary_LT
    elif write_type == 'portary_LT':
        long_portray_LT_table_name = kp.get_parameter_with_keyword('long_portray_LT_table_name')
        long_portray_LT_table_features = kp.get_parameter_with_keyword('long_portray_LT_table_features')
        long_portray_LT_table_type = kp.get_parameter_with_keyword('long_portray_LT_table_type')
        db.write_one(long_portray_LT_table_name, long_portray_LT_table_features, upload_data,
                     long_portray_LT_table_type)

    # 如果write_type为checkTotal, 则针对稽查结果数据进行保存
    elif write_type == 'portary_gantry':
        long_portray_gantry_table_name = kp.get_parameter_with_keyword('long_portray_gantry_table_name')
        long_portray_gantry_table_feature = kp.get_parameter_with_keyword('long_portray_gantry_table_feature')
        long_portray_gantry_table_type = kp.get_parameter_with_keyword('long_portray_gantry_table_type')
        db.write_one(long_portray_gantry_table_name, long_portray_gantry_table_feature, upload_data,
                     long_portray_gantry_table_type)


'''
    创建时间：2022/05/08
    完成时间：2022/05/08
    功能：进行各数据上传数据库——未测试
    修改时间：No.1 2022/6/17, add the charge for 行驶风险评分
'''


def save_data_of_path(write_type, path='', path_type='fold', input_data=None):
    """
    将输入地址内的数据写入数据库
    :param path_type:
    :param input_data: 数组类型数据
    :param write_type: 写入数据的类型，middle代表为中间数据写入，timePart代表一段时间合并处理的数据写入，checkFeature代表稽查特征处理表写入，checkResult代表稽查结果写入
    :param path: 需要写入的数据所在的地址
    :return:
    """
    if path_type == 'fold':
        paths = dop.path_of_holder_document(path, True)
    elif path_type == 'files':
        paths = path
    else:
        paths = [path]
    for path in paths:
        # if '20220129' < path.rsplit('/', 1)[1][:-4].replace('-', '') < '20220208' or '20210929' <
        # path.rsplit('/', 1)[1][:-4].replace('-', '') < '20211009':
        #     continue
        if path.rsplit('/', 1)[1][:-4].replace('-', '') <= '20220709':
            continue
        print(path.rsplit('/', 1)[1][:-4].replace('-', ''))
        save_data = []
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:  # 2022/6/17 add
                    index = dbf.get_index_of_list(row, ['行驶风险评分'])
                else:
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
                        list_ls = row[:-6]
                        list_ls.extend(row[-5:])
                        save_data.append(list_ls)
                    if write_type == 'portary_gantry':
                        save_data.append(row[:-10])
                    if write_type == 'middle':
                        save_data.append(row)
                    elif write_type == 'checkFeature':
                        list_ls = [row[2]]
                        list_ls.extend(row[59:])
                        save_data.append(list_ls)
                    elif write_type == 'checkResult':
                        list_ls = [row[2]]
                        list_ls.extend(row[-17:])
                        save_data.append(list_ls)
                    elif write_type == 'loss':
                        list_ls = row[1:-9]
                        list_ls.append(row[-1])
                        list_ls.extend(row[-9:-1])
                        save_data.append(list_ls)
                    elif write_type == 'whole':
                        list_ls = row[1:-9]
                        list_ls.append(row[21])
                        list_ls.extend(row[-9:])
                        save_data.append(list_ls)
                    elif write_type == 'province':
                        list_ls = row[1:-9]
                        if row[21] != '':
                            list_ls.append(row[21])
                        else:
                            list_ls.append(row[10])
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
                        if row[index[0]] != '' and 100 >= float(row[index[0]]) > 10:
                            save_data.append(row)
        if write_type == 'middle' or write_type == 'checkFeature' or write_type == 'checkResult' \
                or write_type == 'checkTotal' or write_type == 'tollinterval' or write_type == 'totaldata' \
                or write_type == 'LT_OD' or write_type == 'inoutTime' or write_type == 'gantry_OD' \
                or write_type == 'LT_treated_OD' or write_type == 'portary_LT' or write_type == 'portary_gantry':
            save_data_to_Mysql(write_type, save_data)
        else:
            save_data_to_Mysql('timePart', save_data)


'''
    创建时间：2022/10/11
    完成时间：2022/10/11
    功能：进行数据库的数据修改
    关键字：mysql，数据库，数据修改
    修改时间：
'''


def update_data_of_mysql(db_name, table_name, change_feature, change_value, compare_feature, compare_value,
                         compare_sign):
    """
    进行数据库的数据修改
    :param str db_name: 数据库名称，用于创建数据库操作对象
    :param str table_name: 进行操作的数据表名称
    :param list change_feature: 需要进行修改的特征名称集
    :param list change_value: 各特征需要改的目标值
    :param list compare_feature: 锁定目标数据的比较特征集
    :param list compare_value: 锁定目标数据的各特征值
    :param list compare_sign: 锁定目标数据的特征和数值之间的比较符号
    :return:
    """
    # 获取数据库数据
    # 创建数据库连接
    database_parameter = kp.get_parameter_with_keyword(db_name)
    db = gd.OperationMysql(database_parameter["host"], database_parameter["port"], database_parameter["user"],
                           database_parameter["passwd"], database_parameter["db"])
    # 遍历所有要修改的值
    for k in range(len(change_value)):
        # 生成修改语句
        # 生成头语句
        sql_string = 'update ' + table_name + ' set '
        # 生成修改语句
        for i in range(len(change_feature)):
            if len(change_feature) == 1 or i == (len(change_feature) - 1):
                if type(change_value[k][i]) == str:
                    sql_string += change_feature[i] + '="' + change_value[k][i] + '"'
                else:
                    sql_string += change_feature[i] + '=' + str(change_value[k][i])
            else:
                if type(change_value[k][i]) == str:
                    sql_string += change_feature[i] + '="' + change_value[k][i] + '",'
                else:
                    sql_string += change_feature[i] + '=' + str(change_value[k][i]) + ','

        # 生成匹配语句
        where_string = ''
        for i in range(len(compare_feature[k])):
            if type(compare_value[k][i]) == str:
                where_string += compare_feature[k][i] + ' ' + compare_sign[k][i] + ' "' + compare_value[k][i] + '"'
            elif type(compare_value[k][i]) == list:
                for j in range(len(compare_value[k][i])):
                    if type(compare_value[k][i][j]) == str:
                        if j == 0:
                            where_string += compare_feature[k][i] + ' ' + compare_sign[k][i] + ' ("' + \
                                            compare_value[k][i][j] + '"'
                        else:
                            where_string += ', "' + str(compare_value[k][i][j]) + '"'
                    else:
                        if j == 0:
                            where_string += compare_feature[k][i] + ' ' + compare_sign[k][i] + ' (' + str(
                                compare_value[k][i][j])
                        else:
                            where_string += ', ' + str(compare_value[k][i][j])
                where_string += ')'
            else:
                where_string += compare_feature[k][i] + ' ' + compare_sign[k][i] + ' ' + str(compare_value[k][i])
            if i < (len(compare_feature[k]) - 1):
                where_string += ' and '
            else:
                pass
        if where_string == '':
            pass
        else:
            sql_string = " where " + where_string

        if k < (len(change_value) - 1):
            # 修改数据
            db.update_data(sql_string, False)
        else:
            db.update_data(sql_string, True)


'''
    创建时间：2022/09/02
    完成时间：2022/09/06
    功能：进行数据库的数据获取
    关键字：mysql，数据库，数据获取
    修改时间：No.1 2022/10/31, add the new parameter 'addSQL'
'''


@dbf.timeIf_wrapper
def load_data_from_mysql(db_name, table_name, feature_name, feature_value, compare_sign, get_feature='all',
                         return_type='list', addSQL='', sql_type='mysql'):
    """
    进行数据库的数据获取
    :param sql_type:
    :param addSQL:
    :param str db_name: database name
    :param str return_type: 数据库获取的数据的返回形式
    :param str/list get_feature: 需要获取到的特征
    :param str table_name: 进行查询的数据表名称
    :param list feature_name: 条件特征名称，数组类型
    :param list feature_value: 条件对比数值，数组类型
    :param list compare_sign: 条件对比方式符号，数组类型
    :return:
    """
    # 生成查询语句
    # 生成获取字段的语句
    if get_feature == 'all':
        feature_string = "*"
    else:
        feature_string = ''
        for i in range(len(get_feature)):
            if i < (len(get_feature) - 1):
                feature_string += get_feature[i] + ","
            else:
                feature_string += get_feature[i]
    # 生成查询where后的语句
    where_string = ''
    for i in range(len(feature_name)):
        if i < (len(feature_name) - 1):
            if type(feature_value[i]) == str:
                where_string += feature_name[i] + ' ' + compare_sign[i] + ' "' + feature_value[i] + '" and '
            elif type(feature_value[i]) == list:
                for j in range(len(feature_value[i])):
                    if j == 0:
                        where_string += feature_name[i] + ' ' + compare_sign[i] + ' (' + str(feature_value[i][j])
                    else:
                        where_string += ', ' + str(feature_value[i][j])
                where_string += ') and '
        else:
            if type(feature_value[i]) == str:
                where_string += feature_name[i] + ' ' + compare_sign[i] + ' "' + feature_value[i] + '"'
            elif type(feature_value[i]) == list:
                for j in range(len(feature_value[i])):
                    if j == 0:
                        where_string += feature_name[i] + ' ' + compare_sign[i] + ' ("' + str(feature_value[i][j]) + '"'
                    else:
                        where_string += ', "' + str(feature_value[i][j]) + '"'
                where_string += ')'
            else:
                where_string += feature_name[i] + ' ' + compare_sign[i] + ' ' + str(feature_value[i])
    if where_string == '':
        sql_string = "SELECT " + feature_string + " from " + table_name
    else:
        sql_string = "SELECT " + feature_string + " from " + table_name + " where " + where_string

    # 2022/10/31 update
    if addSQL != '':
        sql_string += ' and ' + addSQL

    # 获取数据库数据
    # 创建数据库连接
    database_parameter = kp.get_parameter_with_keyword(db_name)
    db = gd.OperationMysql(database_parameter["host"], database_parameter["port"], database_parameter["user"],
                           database_parameter["passwd"], database_parameter["db"], sql_type)

    # 查询数据
    output_data = db.load_data(sql_string)

    # 将数据转换为所需数据类型
    if return_type == 'list':
        return output_data


'''
    创建时间：2022/10/08
    完成时间：2022/10/08
    功能：进行数据集的数据写入
    关键字：mysql，数据库，数据写入
    修改时间：
'''


@dbf.timeIf_wrapper
def write_data_to_mysql(upload_data, db_name, table_name, feature_name, feature_type, mysql_type):
    """

    :param list upload_data: 需要上传的数据
    :param str db_name: 获取数据库连接信息的关键字
    :param str table_name: 获取数据表名称的关键字
    :param str feature_name: 获取数据表各字段名称的关键字
    :param str feature_type: 获取数据表各字段类型的关键字
    :param str mysql_type: 上传方式
    :return:
    """
    # 获取输入的数据库的相关连接参数
    database_parameter = kp.get_parameter_with_keyword(db_name)
    # 将连接参数代入，得到mysql对象
    db = gd.OperationMysql(database_parameter["host"], database_parameter["port"], database_parameter["user"],
                           database_parameter["passwd"], database_parameter["db"])
    # 得到上传的目标数据表名称
    long_portray_gantry_table_name = kp.get_parameter_with_keyword(table_name)
    # 得到上传的各特征字段名称
    long_portray_gantry_table_feature = kp.get_parameter_with_keyword(feature_name)
    # 得到上传的各特征类型
    long_portray_gantry_table_type = kp.get_parameter_with_keyword(feature_type)
    # 进行数据上传
    db.write_one(long_portray_gantry_table_name, long_portray_gantry_table_feature, upload_data,
                 long_portray_gantry_table_type, data_type=mysql_type, save_num=500)


if __name__ == '__main__':
    data = pd.read_csv('./Data_Origin/test_station.csv')
    # data = data.sort_values('time', ascending=False)
    # data = data
    # data = data.drop_duplicates(subset=['INTERVAL_ID', 'TIME_POINT'], keep='first')
    data = data.fillna(0)
    write_data_to_mysql(data.values, 'overspeed_mysql', 'station_flow_base_name', 'station_flow_base_features',
                        'station_flow_base_features_type', 'csv')
