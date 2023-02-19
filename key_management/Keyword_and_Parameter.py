"""
    -*- coding: UTF-8 -*-
    Copyright (c) 2023, Sun HaoYu. All rights reserved.
    function: 项目关键字管理
"""

import Data_Basic_Function as dbf

# 创建全局变量
Compare_Sign_Parameter_Class = object
Read_Document_Parameter_Class = object

'''
    创建时间：2023/02/16
    完成时间：2023/02/16
    功能：所有参数对象的母对象
    修改时间：
'''


class Parameter_Model(object):
    def get_value(self, key):
        try:
            return getattr(self, key)
        except:
            return None


'''
    创建时间：2023/02/16
    完成时间：2023/02/16
    功能：对比函数中转管理，根据输入的对比符号，进行函数的选择
    修改时间：
'''


class Compare_Sign_Parameter(Parameter_Model):
    def __init__(self):
        # 比对符号对应各函数的对应表
        self.contain = dbf.charge_inputs_contain
        self.equal = dbf.charge_inputs_equal
        self.bigger = dbf.charge_inputs_bigger
        self.biggerEqual = dbf.charge_inputs_bigger_and_equal


'''
    创建时间：2023/02/16
    完成时间：2023/02/16
    功能：读取函数中转管理，根据输入的读取类型，进行函数的选择
    修改时间：
'''


class Read_Document_Parameter(Parameter_Model):
    def __init__(self):
        # 比对符号对应各函数的对应表
        self.list = dbf.get_list_from_document
        self.list = dbf.get_dict_from_config


# 基础参数获取类
class initial_keyClass:
    Compare_Sign_Parameter_Class = Compare_Sign_Parameter()
    Read_Document_Parameter_Class = Read_Document_Parameter()


'''
    创建时间：2023/02/16
    完成时间：2023/02/16
    功能：项目关键参数类
    修改时间：
'''


class Parameter_Management(Parameter_Model):
    def __init__(self, config_path):
        global Compare_Sign_Parameter_Class, Read_Document_Parameter_Class
        Compare_Sign_Parameter_Class = vars(initial_keyClass)['Read_Document_Parameter_Class']
        Read_Document_Parameter_Class = vars(initial_keyClass)['Read_Document_Parameter_Class']
        # 系统关键参数读取
        config_dict = dbf.get_dict_from_document(config_path, [0, 1, 2], key_length=1, ifIndex=False,
                                                 key_for_N_type='list')
        # 遍历所有参数内容，进行赋值
        for key in config_dict.keys():
            # 判定关键字类型，如果是list或者dict，则进行文件读取并返回相应格式内容，再进行赋值
            if config_dict[key][1] == 'list' or config_dict[key][1] == 'dict':
                function = Read_Document_Parameter_Class.get_value(config_dict[key][2])
                read_data = function(config_path)
            else:
                read_data = config_dict[key][0]
            try:
                setattr(self, key, read_data)
            except:
                self.error = '输入参数有错误'

        # 绿通基础数据清洗出来的特征字典名称设置
        self.LT_basic_feature = ['车种', '预约开始运输时间', '预约结束运输时间', '预约起点收费站编号', '预约终点收费站编号',
                                 '预约用户编号', '车牌(全)', '查验时间', '车型', '厢型', '车主', '出口车道号', '班组标号',
                                 '入口重量', '出口重量', '重量', '入口站ID', '出口站ID', '验货人员编号', '复核人员编号', 'PASSID',
                                 '查验结果', '通行介质', '免收费用', '不合格原因', '免检标识', '最后修改人', '最后修改时间',
                                 '入口收费站', '入口地', '入口市', '入口省', '出口收费站', '出口地', '出口市', '出口省', '绿通品种',
                                 '绿通品种小分类', '绿通品种大分类', '入口时间', '入口地输出概率', '出口地输入概率', '车型与品种匹配度',
                                 '出入地绿通运输概率', '是否载重异常']

        self.analysis_feature = {''}

        self.analysis_value_feature = {''}
