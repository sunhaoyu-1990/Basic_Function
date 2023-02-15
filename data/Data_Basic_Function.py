"""
数据基础处理函数集
文档创建时间：2021/12/22
文档修改时间：
"""
import csv
import random
# import numpy as np
import datetime
from fitter import Fitter
from scipy.stats import *
import pandas as pd
import dtw
# import xlrd
import sys
from urllib.parse import unquote, quote  # 进行URL码的转换

sys.setrecursionlimit(100000)  # 设置递归函数的递归循环上限，这里设置为100000次


# import seaborn as sns
# from scipy.stats import *


def get_index_of_list(row, features):
    """
    获取数组中多个字段的索引值，并以数组形式返回
    类型：非底层函数，底层应用Python中数组的index函数
    :param row: 数组
    :param features: 需要获取索引的特征值数据
    :return:
    """
    columns_list = []  # 进行每个特征索引的添加
    # 如果特征为数组类型，则遍历得到每一个的下标
    if type(features) == list:
        for fea in features:
            try:
                columns_list.append(row.index(fea))  # 底层应用Python中数组的index函数，进行元素索引的获取
            except Exception as e:  # 如果该字段名称未匹配到索引，说明数据中没有该特征，直接跳过，并显示异常情况
                print(str(e))
                print("异常：未匹配到数据中的字段，该特征名称为 ", fea)
                continue
    # 如果特征为非数组类型，直接得到该值的下标
    else:
        try:
            columns_list.append(row.index(features))
        except Exception as e:  # 如果该字段名称未匹配到索引，说明数据中没有该特征，直接跳过，并显示异常情况
            print(str(e))
            print("异常：未匹配到数据中的字段，该特征名称为 ", features)

    return columns_list


def concat_values_of_list(row, index_list, concat_sign, if_all=False):
    """
    根据提供的数组，将所给的索引的元素进行合并
    :param if_all:是否对row的全部数据进行操作，TRUE为全数据操作，FALSE为只对index_list的元素操作
    :param concat_sign: 合并时的间隔符
    :param row: 原始数组
    :param index_list: 需要合并的元素索引
    :return:
    """
    # 如果只对index_list的数据进行操作
    if not if_all:
        concat_list = []  # 用于获取所有下标对应的数值

        for i, index in enumerate(index_list):
            if i == 0:
                concat_list.append(row.pop(index))
            if i > 0:
                # 如果上一个下标是在这一个的前面，则下标需减一，因为每拿出一个元素就在数组中删除该元素
                if index_list[i] > index_list[i - 1]:
                    concat_list.append(row.pop(index - 1))
                # 如果上一个下标是在这一个的后面，则下标不变
                else:
                    concat_list.append(row.pop(index))
    # 如果对row的全部数据进行操作
    else:
        concat_list = row
    concat_list_row = []  # 用于保存最后的结果
    concat_string = ''  # 用于拼接各元素，形成字符串
    # 循环遍历所有元素
    for i, con in enumerate(concat_list):
        # 如果是最后一个元素，拼接的结尾不需要加分隔符
        if i == len(concat_list) - 1:
            concat_string += str(con)
        # 如果不是最后一个元素，拼接的结尾需要加分隔符
        else:
            concat_string += str(con) + concat_sign
    concat_list_row.append(concat_string)
    return concat_list_row


def get_values_of_list(row, index_list, remark='None'):
    """
    根据输入的数组和索引数组，输出所需的数组内容，以数组形式输出
    :param remark: 每个元素的处理方式，None为无处理，strip为去除每个元素的首尾空格
    :param row: 原数组
    :param index_list: 所需索引数组
    :return:
    """
    values = []
    for index in index_list:
        # None为无处理
        if remark == 'None':
            values.append(row[index])
        # strip为去除每个元素的首尾空格
        elif remark == 'strip':
            values.append(row[index].strip())
    return values


def compute_with_disc(disc_list, key_list, function='add', key_add='', parameters=None, split_sign='|'):
    """
    1.根据所给的字典数据，将key_list对应的值进行相应运算
    2.也可以用字典中key对应的数据，替换key_list，function用“replace”
    类型：非底层函数，底层应用Python中字符串的split函数
    :param key_add:
    :param split_sign:
    :param function: 元素的处理方式
    :param parameters: function为retain时，元素在parameter内存在，才参与处理
    :param disc_list: 字典原始数据
    :param key_list: 关键字数据，可以为数组，也可以是字符串
    :return:
    """
    except_num = 0  # 用于计算无法匹配上字典内容的数据量
    # 如果function是add，则key_list中的值对应字典里的值进行相加，所以字典中对应的值应为数字
    if function == 'add':
        if type(key_list) == str:
            key_list = key_list.split('|')  # 如果是字符串，以间隔符进行拆分，bug点：间隔符写死
        sum = 0  # 用于计算所有的数字之后
        for key in key_list:
            try:
                sum += int(disc_list[key])
            except:
                except_num += 1
                sum += 0
        return sum
    # 如果function是replace，则则用key_list值对应字典里的值进行替换，所以字典中对应的值应为字符串
    elif function == 'replace':
        # 如果key_list为字符串，则先进行拆分成数组，再匹配，最后再合并成字符串进行返回
        if type(key_list) == str:
            key_list = key_list.split(split_sign)  # bug：拆分的分隔符固定为|，后续自动分析间隔内容，判断出间隔符
            key_str = ''  # 用于合并所有的字符串
            if split_sign == '；':
                split_sign = ';'
            for i, key in enumerate(key_list):
                # 如果不是最后一个元素，则拼接结尾加“|”
                if i != len(key_list) - 1:
                    try:
                        if key_add != '':
                            key_str += str(disc_list[key + key_add]) + split_sign
                        else:
                            key_str += str(disc_list[key]) + split_sign
                    except:  # 如果字典中没有对应的值，则补入none或者0值
                        except_num += 1
                        # key_str += 'none|'
                        key_str += '0' + split_sign  # bug：补入的内容为固定内容，之后可根据字典内容类型进行自动赋值
                # 如果为最后一个元素，则拼接结尾不加“|”
                else:
                    try:
                        if key_add != '':
                            key_str += disc_list[key + key_add]
                        else:
                            key_str += disc_list[key]
                    except:  # 如果字典中没有对应的值，则补入none或者0值
                        except_num += 1
                        # key_str += 'none'
                        key_str += '0'  # bug：补入的内容为固定内容，之后可根据字典内容类型进行自动赋值
            return key_str
        # 如果key_list为数组类型，则对每个元素直接匹配，并将替换后的数据返回
        else:
            for i, key in enumerate(key_list):
                try:
                    key_list[i] = disc_list[key]
                except:  # 如果字典中没有对应的值，则补入NAN
                    key_list[i] = np.NAN  # bug：补入的内容为固定内容，之后可根据字典内容类型进行自动赋值
            return key_list
    # 将key_list对应的字典的值进行判断，看是否在parameter里，如果在则直接保存key_list的值，否则该值不进行输出
    elif function == 'retain':
        result_list = []
        if type(key_list) == str:
            if key_list != '':
                key_list = key_list.split(split_sign)
                for key in key_list:
                    try:
                        # 判断字典的匹配值是否在parameter中，是的话直接保存key，否则不进行保存
                        if disc_list[key] in parameters:
                            result_list.append(key)
                    except:
                        result_list.append(key)
        else:
            if len(key_list) != 0:
                for key in key_list:
                    try:
                        if disc_list[key] in parameters:
                            result_list.append(key)
                    except:
                        result_list.append(key)
        return result_list


'''
    修改时间：No.1 2022/3/16，增加了输入参数ifIndex，增加了新功能，即当输入直接为下标时的处理
            No.2 2022/4/22，添加了备注
            No.3 2022/9/8, add feature filter
'''


def get_disc_from_document(path, features, encoding='gbk', key_for_N=False, length=0, key_length=1, ifIndex=True,
                           key_for_N_type='disc', value_com_type='list', sign='-', input_type='path',
                           filter_feature=None, filter_value=None, ifNoCol=False):
    """
    根据文件地址读取文件，并将所需的字段，以字典的对应形式返回
    :param filter_feature:
    :param ifNoCol:
    :param filter_value:
    :param str sign:
    :param str input_type: 输入数据的内容，如果为path，表示从文件中读取数据，如果为list，表示输入即为数组类型
    :param str value_com_type:
    :param str key_for_N_type:如果数据中有一个key对应多个值的情况，则多个值的集合格式，dict为集合，其他为数组
    :param bool ifIndex:输入的features是索引还是字典名称
    :param int key_length: key值需要拼接的字段数量
    :param int length:每个字段的截取长度值
    :param bool key_for_N:数据中是否有一个key对应多个值的情况，TRUE为存在，FALSE为不存在
    :param str encoding: 文件解码方式，默认为gbk
    :param str or list path: 文件地址，当input_type为'list'时，path为数组类型
    :param list features: 所需字段数组，为2维数组，第一位为key，第二位为value
    :return:
    """
    if input_type == 'path':
        file_type = path.rsplit('.', 1)[1]
        if file_type == 'csv' or file_type == 'txt':
            with open(path, encoding=encoding) as f:
                ganlist = {}
                for i, row in enumerate(f):
                    row = row.replace('"', '')
                    # 转换为数组
                    row = row.split(',')
                    # 去掉结尾的"\t"
                    row[-1] = row[-1][:-1]
                    if i == 0:
                        # 如果输入的features为字典名称，则通过文件中的字段名称，来确定索引
                        if ifIndex:
                            if filter_feature:  # add 2022/9/8
                                filter_index = get_index_of_list(row, filter_feature)
                            index_list = get_index_of_list(row, features)
                            continue
                        # 如果输入的features直接为索引，则直接采用
                        else:
                            if filter_feature:  # add 2022/9/8
                                filter_index = filter_feature
                            index_list = features
                            if ifNoCol:
                                pass
                            else:
                                continue

                    if filter_feature:  # add 2022/9/8
                        for j in range(len(filter_feature)):
                            if type(filter_value[j]) == list:
                                if row[filter_index[j]] in filter_value[j]:
                                    pass
                                else:
                                    continue
                            else:
                                if row[filter_index[j]] == filter_value[j]:
                                    pass
                                else:
                                    continue
                    # 判断key字段是否只需要一个字段值，如果不是，将各字段值拼成key
                    if key_length != 1:
                        # 创建key_string，用于将各字段拼成key
                        key_string = ''
                        # 判断是否截取每个字段的值
                        if length != 0:
                            # 进行循环拼接
                            for j in range(key_length):
                                # 最后一个元素之前的结尾拼接‘-’
                                if j < key_length - 1:
                                    key_string = key_string + row[index_list[j]][:length] + sign
                                else:
                                    key_string = key_string + row[index_list[j]][:length]
                        else:
                            for j in range(key_length):
                                if j < key_length - 1:
                                    key_string = key_string + row[index_list[j]] + sign
                                else:
                                    key_string = key_string + row[index_list[j]]
                    # 如果key只需要一个字段值
                    else:
                        if length != 0:
                            key_string = row[index_list[0]][:length]
                        else:
                            key_string = row[index_list[0]]
                    # 判断字段的value值是否是多个字段组成，如果是多个，则将其放入一个数组中
                    if (len(index_list) - key_length) >= 2:

                        if len(index_list) - key_length <= 1:
                            list_ls = row[index_list[-1]]
                        else:
                            if value_com_type == 'list':
                                list_ls = []
                                for j in range(key_length, len(index_list)):
                                    list_ls.append((row[index_list[j]]))
                            elif value_com_type == 'str':
                                list_ls = ''
                                for j in range(key_length, len(index_list)):
                                    if j != (len(index_list) - 1):
                                        list_ls += row[index_list[j]] + ','
                                    else:
                                        list_ls += row[index_list[j]]

                        ganlist[key_string] = list_ls

                    else:
                        # 判断是否有一个key对应多个值的情况，如果有可以通过key_for_N_type设置集合方式
                        if key_for_N:
                            try:
                                # 如果key_for_N_type为‘dict’，则将多个值以集合保存
                                if key_for_N_type == 'disc':
                                    ganlist[key_string].add(row[index_list[-1]])
                                # 如果key_for_N_type不是‘dict’，则将多个值以数组保存
                                elif key_for_N_type == 'list':
                                    ganlist[key_string].append(row[index_list[-1]])
                                elif key_for_N_type == 'add':
                                    ganlist[key_string] += float(row[index_list[-1]])
                            # 当添加失败时，先创建key-value
                            except:
                                if key_for_N_type == 'disc':
                                    ganlist[key_string] = {row[index_list[-1]]}
                                elif key_for_N_type == 'list':
                                    ganlist[key_string] = [row[index_list[-1]]]
                                elif key_for_N_type == 'add':
                                    ganlist[key_string] = float(row[index_list[-1]])
                        # 如果不是一个key对应多个值的情况
                        else:
                            ganlist[key_string] = row[index_list[-1]]
        elif file_type == 'xlsx':
            ganlist = {}
            data = pd.read_excel(path)
            columns = list(data.columns.values)
            # 如果输入的features为字典名称，则通过文件中的字段名称，来确定索引
            if ifIndex:
                data[features] = data[features].fillna('')
                index_list = get_index_of_list(columns, features)
            # 如果输入的features直接为索引，则直接采用
            else:
                index_list = features
            data = data.values
            for i in range(len(data)):
                # 判断key字段是否只需要一个字段值，如果不是，将各字段值拼成key
                if key_length != 1:
                    # 创建key_string，用于将各字段拼成key
                    key_string = ''
                    # 判断是否截取每个字段的值
                    if length != 0:
                        # 进行循环拼接
                        for j in range(key_length):
                            # 最后一个元素之前的结尾拼接‘-’
                            if j < key_length - 1:
                                key_string = key_string + data[i][index_list[j]][:length] + sign
                            else:
                                key_string = key_string + data[i][index_list[j]][:length]
                    else:
                        for j in range(key_length):
                            if j < key_length - 1:
                                key_string = key_string + data[i][index_list[j]] + sign
                            else:
                                key_string = key_string + data[i][index_list[j]]
                # 如果key只需要一个字段值
                else:
                    if length != 0:
                        key_string = data[i][index_list[0]][:length]
                    else:
                        key_string = data[i][index_list[0]]
                # 判断字段的value值是否是多个字段组成，如果是多个，则将其放入一个数组中
                if (len(index_list) - key_length) > 2:

                    if len(index_list) - key_length <= 1:
                        list_ls = data[i][index_list[-1]]
                    else:
                        list_ls = []
                        for j in range(key_length, len(index_list)):
                            list_ls.append((data[i][index_list[j]]))

                    ganlist[key_string] = list_ls

                else:
                    # 判断是否有一个key对应多个值的情况，如果有可以通过key_for_N_type设置集合方式
                    if key_for_N:
                        try:
                            # 如果key_for_N_type为‘dict’，则将多个值以集合保存
                            if key_for_N_type == 'disc':
                                ganlist[key_string].add(data[i][index_list[-1]])
                            # 如果key_for_N_type不是‘dict’，则将多个值以数组保存
                            elif key_for_N_type == 'list':
                                ganlist[key_string].append(data[i][index_list[-1]])
                            elif key_for_N_type == 'add':
                                ganlist[key_string] += float(data[i][index_list[-1]])
                        # 当添加失败时，先创建key-value
                        except:
                            if key_for_N_type == 'disc':
                                ganlist[key_string] = {data[i][index_list[-1]]}
                            elif key_for_N_type == 'list':
                                ganlist[key_string] = [data[i][index_list[-1]]]
                            elif key_for_N_type == 'add':
                                ganlist[key_string] = float(data[i][index_list[-1]])
                    # 如果不是一个key对应多个值的情况
                    else:
                        ganlist[key_string] = data[i][index_list[-1]]
    elif input_type == 'list':
        ganlist = {}
        data = path
        index_list = features  # 将特征索引进行赋值
        for i in range(len(data)):
            # 判断key字段是否只需要一个字段值，如果不是，将各字段值拼成key
            if key_length != 1:
                # 创建key_string，用于将各字段拼成key
                key_string = ''
                # 判断是否截取每个字段的值
                if length != 0:
                    # 进行循环拼接
                    for j in range(key_length):
                        # 最后一个元素之前的结尾拼接‘-’
                        if j < key_length - 1:
                            key_string = key_string + str(data[i][index_list[j]])[:length] + sign
                        else:
                            key_string = key_string + str(data[i][index_list[j]])[:length]
                else:
                    for j in range(key_length):
                        if j < key_length - 1:
                            key_string = key_string + str(data[i][index_list[j]]) + sign
                        else:
                            key_string = key_string + str(data[i][index_list[j]])
            # 如果key只需要一个字段值
            else:
                if length != 0:
                    key_string = str(data[i][index_list[0]])[:length]
                else:
                    key_string = str(data[i][index_list[0]])
            # 判断字段的value值是否是多个字段组成，如果是多个，则将其放入一个数组中
            if (len(index_list) - key_length) > 2:

                if len(index_list) - key_length <= 1:
                    list_ls = data[i][index_list[-1]]
                else:
                    list_ls = []
                    for j in range(key_length, len(index_list)):
                        list_ls.append((data[i][index_list[j]]))

                ganlist[key_string] = list_ls

            else:
                # 判断是否有一个key对应多个值的情况，如果有可以通过key_for_N_type设置集合方式
                if key_for_N:
                    try:
                        # 如果key_for_N_type为‘dict’，则将多个值以集合保存
                        if key_for_N_type == 'disc':
                            ganlist[key_string].add(data[i][index_list[-1]])
                        # 如果key_for_N_type不是‘dict’，则将多个值以数组保存
                        elif key_for_N_type == 'list':
                            ganlist[key_string].append(data[i][index_list[-1]])
                        elif key_for_N_type == 'add':
                            ganlist[key_string] += float(data[i][index_list[-1]])
                    # 当添加失败时，先创建key-value
                    except:
                        if key_for_N_type == 'disc':
                            ganlist[key_string] = {data[i][index_list[-1]]}
                        elif key_for_N_type == 'list':
                            ganlist[key_string] = [data[i][index_list[-1]]]
                        elif key_for_N_type == 'add':
                            ganlist[key_string] = float(data[i][index_list[-1]])
                # 如果不是一个key对应多个值的情况
                else:
                    ganlist[key_string] = data[i][index_list[-1]]
    return ganlist


'''
    修改时间：2022/02/09，新增加针对多维数组中的某一各元素进行排序的功能
'''


def basic_sort_list(lists, ascending=True, element=-1):
    """
    通过冒泡法，将数组进行排序,可进行正序也可进行倒序
    :param element: 如果需要对多维数组进行排序，且通过多维中的某一个元素进行排序，需输入该元素的下标，2022/02/09添加
    :param ascending: 正序和倒序特征值
    :param lists: 需排序数组
    :return:
    """
    count = len(lists)  # 获取数组的长度
    for i in range(0, count):  # 进行遍历排序
        if element == -1:  # 如果element为-1，说明是对一维数组进行排序
            for j in range(i + 1, count):
                if ascending:  # 如果ascending为TRUE，为正序
                    if lists[i] > lists[j]:
                        lists[i], lists[j] = lists[j], lists[i]
                else:  # 为从大到小排序
                    if lists[i] < lists[j]:
                        lists[i], lists[j] = lists[j], lists[i]
        else:  # element不为-1，说明是对多维数组进行排序
            for j in range(i + 1, count):
                if ascending:  # 如果ascending为TRUE，为正序
                    if lists[i][element] > lists[j][element]:  # 判断多维数组第element的元素的大小
                        lists[i], lists[j] = lists[j], lists[i]
                else:  # 为从大到小排序
                    if lists[i][element] < lists[j][element]:  # 判断多维数组第element的元素的大小
                        lists[i], lists[j] = lists[j], lists[i]
    return lists


'''
    创建时间：2022/2/9
    完成时间：2022/2/9
    功能：将数组中相邻两个相同的元素全部去除，达到去重的目的
    修改时间：
'''


def basic_delete_adjoin_same(lists, element=-1):
    """
    将数组中相邻两个相同的元素全部去除，达到去重的目的
    :param element:多维数组时，指定进行比较的元素的下标
    :param lists:数组数据
    :return:
    """
    drop_index_list = []  # 用于保存需要删除的元素的下标
    for i in range(0, len(lists) - 1):  # 遍历第一个元素到倒数第二个元素的下标
        if element == -1:  # 如果element为-1，说明是对一维数组进行去重
            if lists[i] == lists[i + 1]:  # 如果相邻两个元素相同，将后一个的下标进行保存
                drop_index_list.append(i + 1)
        else:  # element不为-1，说明是对多维数组进行操作
            if lists[i][element] == lists[i + 1][element]:  # 如果相邻两个数组的同下标元素相同，将后一个数组的下标进行保存
                drop_index_list.append(i + 1)

    for i in range(len(drop_index_list)):  # 遍历需要删除的元素下标数组，进行删除
        lists.pop(drop_index_list[len(drop_index_list) - i - 1])  # 进行倒序删除，即原始数组的元素从右往左删除，避免下标错乱

    return lists


'''
    创建时间：2022/2/9
    完成时间：2022/2/9
    功能：数组（一维或多维）去重
    修改时间：
'''


def basic_duplicate_remove(lists, element=[-1, -1]):
    """
    数组（一维或多维）去重
    :param lists: 一维或多维数组
    :param element: 双元素的数组，第一个为用于排序的列索引，第二个是判断是否相同的列索引
    :return:
    """
    # 将输入的数组进行排序，如果为多维数组，就针对某一维进行排序
    # list_sort = basic_sort_list(lists, element=element[0])  # 应用冒泡法
    basic_quick_sort(lists, 0, len(lists) - 1, element=element[0])  # 应用快速排序法
    # 将数组中相邻两个相同的元素全部去除
    list_result = basic_delete_adjoin_same(lists, element=element[1])
    return list_result


'''
    创建时间：2022/1/5
    完成时间：2022/1/5
    功能：计算输入的两个时间的间隔时间，输出单位可选
    修改时间：
'''


def compute_intervel_of_two_time(time_first, time_second, return_type='minite'):
    """
    计算输入的两个时间的间隔时间，输出单位可选
    :param return_type: 返回数据的单位，默认为分钟
    :param time_first: 时间字符串，前一个时间
    :param time_second: 时间字符串，后一个时间
    :return:
    """
    intervel = datetime.datetime.strptime(time_second, '%Y-%m-%d %H:%M:%S') - \
               datetime.datetime.strptime(time_first, '%Y-%m-%d %H:%M:%S')  # 计算每段路门架之间的间隔时间
    if return_type == 'minite':
        intervel = intervel.total_seconds() / 60.0
    elif return_type == 'hour':
        intervel = intervel.total_seconds() / 3600.0
    elif return_type == 'day':
        intervel = intervel.total_seconds() / 3600.0 / 24
    elif return_type == 'second':
        intervel = intervel.total_seconds()

    return intervel


'''
    创建时间：2022/1/11
    完成时间：2022/1/11
    功能：将输入数据list，转换为5分位值进行输出
    修改时间：No.1 2022/3/11，将数组的排序方法更新为快速排序法
'''


def get_statistics_value_of_list(list_value, return_type='all', remove_ab=False, sortBySort=False):
    """
    将输入数据list，转换为5分位值进行输出
    :param sortBySort:TRUE应用numpy的sort方法进行排序，FALSE应用自己的快速排序法进行排序
    :param remove_ab: 判断是否去掉异常数据
    :param return_type: 数据的返回形式，all为全部返回，statistics为五分位值，mean为均值
    :param list_value: 输入数据，可为字符串也可为数组
    :return: 将新生成的数组类型数据返回
    """
    # Q0存储最小值，Q1存储1/4值，Q2存储中位值，Q3存储3/4值，Q4存储最大值
    if type(list_value) == str:
        list_value = list_value.split(',')  # bug：分隔符固定为“，”
    list_value = [float(i) for i in list_value]
    if sortBySort:
        list_value = np.sort(list_value)
    else:
        basic_quick_sort(list_value, 0, len(list_value) - 1)  # 通过自编的快速排序法进行数组排序

    # 如果return_type为'all'或者'statistic'，则进行五分位值的计算:
    if return_type == 'all' or return_type == 'statistic':
        if len(list_value) == 1:  # 如果输入的数组只有一位，则五分位数都为这个数
            Q0 = float(list_value[0])
            Q1 = float(list_value[0])
            Q2 = float(list_value[0])
            Q3 = float(list_value[0])
            Q4 = float(list_value[0])
        elif len(list_value) == 2:  # 如果输入的数组为2位，则前3分位为第1个数，后为第2个数
            Q0 = float(list_value[0])
            Q1 = float(list_value[0])
            Q2 = float(list_value[0])
            Q3 = float(list_value[1])
            Q4 = float(list_value[1])
        elif len(list_value) == 3:  # 如果输入的数组为3位，则前2分位为第1个数，中位数为第二个数，后2分位为第3个数
            Q0 = float(list_value[0])
            Q1 = float(list_value[0])
            Q2 = float(list_value[1])
            Q3 = float(list_value[2])
            Q4 = float(list_value[2])
        elif len(list_value) == 4:  # 如果输入的数组为4位，则前1分位为第1个数，1/4位数和中位数为第二个数，3/4位数为第3个数，最大数为第4个数
            Q0 = float(list_value[0])
            Q1 = float(list_value[1])
            Q2 = float(list_value[1])
            Q3 = float(list_value[2])
            Q4 = float(list_value[3])
        elif len(list_value) == 5:  # 如果输入的数组为5位，则5个数分别对应5分位数
            Q0 = float(list_value[0])
            Q1 = float(list_value[1])
            Q2 = float(list_value[2])
            Q3 = float(list_value[3])
            Q4 = float(list_value[4])
        elif len(list_value) % 2 == 0:  # 如果输入的数组大于5位，且有偶数个，则中位数为中间两个数的均值
            Q0 = float(list_value[0])
            Q1 = float(list_value[int(len(list_value) / 4)])
            Q2 = (float(list_value[int(len(list_value) / 2) - 1]) + float(list_value[int(len(list_value) / 2)])) / 2
            Q3 = float(list_value[int(len(list_value) / 4) * 3])
            Q4 = float(list_value[len(list_value) - 1])
        else:  # 如果输入的数组大于5位，且有奇数个
            Q0 = float(list_value[0])
            Q1 = float(list_value[int(len(list_value) / 4)])
            Q2 = float(list_value[int(len(list_value) / 2)])
            Q3 = float(list_value[int(len(list_value) / 4) * 3])
            Q4 = float(list_value[len(list_value) - 1])

    # 如果返回的模式为all或者mean，则需要计算均值
    if return_type == 'all' or return_type == 'mean':
        mode = {}  # 用于计算均值
        # 获取均值
        sum_num = 0
        n = 0
        for value in list_value:
            if value > 0:  # bug：这里设置了下限，即大于0的数据，才可以参与均值计算
                n += 1
                sum_num += value
            # 2022/8/5添加，进行众数的计算
            try:
                mode[value] += 1
            except:
                mode[value] = 1
        max_mode_value, max_mode_index = get_topN_values_or_index(list(mode.values()), 1, list(mode.keys()), True)
        if n == 0:  # 如果没有数值参与均值计算，则均值为0
            avg = 0
        else:
            avg = round(sum_num / n, 2)
    # 如果为all，则返回全部的五分位值和均值
    if return_type == 'all':
        data_list = [Q0, Q1, Q2, Q3, Q4, avg, max_mode_index[0]]
    elif return_type == 'statistic':
        data_list = [Q0, Q1, Q2, Q3, Q4]
    # 如果为mean，则只计算均值
    elif return_type == 'mean':
        data_list = [avg]
    else:
        data_list = []

    return data_list


'''
    创建时间：2022/2/10
    完成时间：2022/2/10
    功能：将给定的数组以给定的符号相隔进行合并成字符串，并可限制数组中每一项的长度
'''


def list_to_string_with_sign(lists, sign, lens=""):
    """
    将给定的数组以给定的符号相隔进行合并成字符串，并可限制数组中每一项的长度
    :param lists: 需要合并的数组
    :param sign: 合并间隔符
    :param lens: 数组每一项的合并长度
    :return: 返回字符串，和原数组的长度
    待完善：截取长度是否可以根据某个特定字符进行
    """
    # 创建空字符串，用于将lists中的值进行串联
    path = ''

    # 将所有lists中的值用sign相隔，进行组合
    for i, gantry in enumerate(lists):
        # 如果是第一个元素，则在开头不加分隔符
        if i == 0:
            # 如果lens为‘’，代表不需要对lists中的值进行截取
            if lens == '':
                path = gantry
            # 否则对lists中的值进行长度截取
            else:
                path = gantry[:lens]
        # 其余元素在开头加分隔符
        else:
            if lens == '':
                path = path + sign + gantry
            else:
                path = path + sign + gantry[:lens]
    return path


'''
    创建时间：2022/2/13
    完成时间：2022/2/13
    功能：计算所给数组中某一个元素（或者该元素的部分内容）在数组中所有出现的下标位置
    修改时间：
'''


def get_all_index_of_same_element(lists, element_index, length):
    """
    计算所给数组中某一个元素（或者该元素的部分内容）在数组中所有出现的下标位置
    :param lists: 输入的数组
    :param element_index: 要进行比较的元素在数组中的下标位置
    :param length: 元素参与对比的部分长度
    :return: 下标位置数组
    """
    index_list = []  # 用于记录与指定元素内容一样的元素的下标
    # 循环遍历输入数组的所有元素
    for i, ele in enumerate(lists):
        if ele[:length] == lists[element_index][:length]:
            index_list.append(i)
    return index_list


'''
    创建时间：2022/2/13
    完成时间：2022/2/13
    功能：计算所给数组中所有元素（或者元素的部分内容）在数组出现的最多次数
    修改时间：
'''


def get_max_times_of_elements(lists, length):
    """
    计算所给数组中所有元素（或者元素的部分内容）在数组出现的最多次数
    :param lists: 输入的数组
    :param length: 元素参与对比的部分长度
    :return: 返回最大长度和最大长度门架中最后一个的下标
    """
    max_value = 0  # 用于记录数组中元素出现的最大次数
    last_index = 0  # 用于记录最高重复的门架中最后的一个的下标
    ele_times = {}  # 用于记录每个元素和其在数组中的出现次数，不返回，可在后续进行拓展
    # 循环遍历所有输入数组的元素
    for i, ele in enumerate(lists):
        try:
            ele_times[ele[:length]] += 1  # 每个元素出现，就在其对应的value加上1
        except:
            ele_times[ele[:length]] = 1  # 如果出现的元素为第一次，则给该元素的值赋值1
        if max_value <= ele_times[ele[:length]]:  # 如果该元素的出现次数大院max_value，就将该值赋给max_value
            max_value = ele_times[ele[:length]]
            last_index = i
    return max_value, last_index


'''
    创建时间：2022/2/13
    完成时间：2022/2/13
    功能：判断所给数组被所给下标分割的两部分比例是否大于所给的系数
    修改时间：No.1 2022/3/1，将left_num除以right_num，将parameter更改为float(1/parameter)
'''


def chargeIf_prop_of_two_part(lists, index_in, parameter):
    """
    判断所给数组被所给下标分割的两部分比例是否大于所给的系数
    :param index_in: 输入的断点的下标位置
    :param parameter: 进行对比的系数值
    :param lists: 输入的数组
    :return: 返回0或1,0代表比所给系数小，1代表比所给系数大
    """
    if index_in != 0:
        left_num = index_in  # 断点左侧数组元素的个数
        right_num = len(lists) - index_in  # 断点右侧数组元素的个数
        # 循环遍历所有输入数组的元素
        prop = left_num / right_num  # 计算右侧元素数量和左侧的比例，2022/3/1修改，将left_num除以right_num
        if_big = 0  # 用于判断比例是否大于所给的系数，默认为0
        if prop < float(1 / parameter):  # 判断比例是否大于所给的系数, 2022/3/1修改，将parameter更改为float(1/parameter)
            if_big = 1
    else:
        if_big = 1
    return if_big


'''
    创建时间：2022/2/13
    完成时间：2022/2/13
    功能：判断所给的路径是U型路径、J型路径、巡回路径还是正常路径
    修改时间：
'''


def service_charge_Upath_Jpath_Cyclepath(path_in):
    """
    判断所给的路径是U型路径、J型路径、巡回路径还是正常路径
    :param path_in: 输入的路径
    :return: 返回0/1/2/3,0代表正常，1代表U型路径，2代表J型路径，3代表巡回型路径
    """
    if path_in != '':
        path_in_list = path_in.split('|')  # 将路径字符串拆解为数组,bug：分隔符固定
        max_para, max_index = get_max_times_of_elements(path_in_list, 14)  # 计算路径中各路段门架出现的最大次数，门架ID的前14位是各路段门架的唯一码
        if max_para > 2:
            return 3  # 如果各路段门架出现的最大次数大于2次，说明有巡回情况
        elif max_para == 1:  # bug: 可能由于门架数据缺失，造成非正常路径判断为正常路径
            return 0  # 如果各路段门架出现的最大次数为1次，说明无路段重复行驶情况
        else:
            # 得到与出口门架ID重复的门架在数据中的下标
            # bug: 如果由于数据缺失造成出口门架ID在路径中没有找到重复的情况，将失效
            index_list = get_all_index_of_same_element(path_in_list, -1, 14)
            if len(index_list) == 0:
                index_list = get_all_index_of_same_element(path_in_list, max_index, 14)
            # 得到重复路径的距离与出口门架ID所在位置到入口的距离的比例是否大于2，是的话判断为U型路径，不是的话判断为J型路径
            if_para = chargeIf_prop_of_two_part(path_in_list, index_list[0], 2)
            if if_para == 1:  # 是的话判断为U型路径
                return 1
            else:  # 不是的话判断为J型路径
                return 2
    else:
        return 0


'''
    创建时间：2021/11/2
    完成时间：2021/11/2
    功能：获取两个字符串的最小相似步数
    修改时间：
'''


def get_similarity_of_two_string(first, second):
    """
    对比两个字符串的相似度
    :param first:字符串1
    :param second:字符串2
    :return: 返回量字符串的最小改变步长和最长字符串的长度，用于后续计算相似度
    """
    if len(first) == 0 and len(second) != 0:  # 如果第一个字符串为空, 则返回第二个字符串的长度
        return len(second), len(second)
    elif len(second) == 0 and len(first) != 0:  # 如果第二个字符串为空, 则返回第一个字符串的长度
        return len(first), len(first)
    elif len(second) == 0 and len(first) == 0:  # 如果两个字符串都为空, 则直接返回1,1，bug：该情况为特殊情况，具体的返回值需根据实际情况确定，目前是按相同返回
        return 0, 1
    elif len(first) > len(second):  # 判断两个字符串的长度，并进行调换，确保第二个字符串不比第一个短
        first, second = second, first

    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [list(range(second_length)) for x in range(first_length)]
    # print distance_matrix
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if first[i - 1] != second[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
            # print distance_matrix
    length = len(second)
    return distance_matrix[first_length - 1][second_length - 1], length


'''
    创建时间：2021/11/4
    完成时间：2021/11/4
    功能：数组的阈值转换，对比数组每个值与阈值的大小，进行划分，根据输入的值重新定义
    修改时间：
'''


def data_transform_by_threshold(lists, threshold, scope):
    """
    数组的阈值转换，对比数组每个值与阈值的大小，进行划分，根据输入的值重新定义
    如果threshold为数组，则判断list的值是否在其中
    :param lists: 需要处理的数组
    :param threshold: 阈值，或者为阈值数组
    :param scope: 满足阈值上下时赋的值
    :return:
    """

    result = []
    for i, li in enumerate(lists):
        if type(threshold) == list:
            if li in threshold:
                result.append(scope[1])
            else:
                result.append(scope[0])
        else:
            if li < threshold:
                result.append(scope[0])
            else:
                result.append(scope[1])
    return result


'''
    创建时间：2022/2/25
    完成时间：2022/2/25
    功能：计算两个字符串数组内所有字符串的相似度
    修改时间：
'''


def get_similarity_of_two_list(first_list, second_list):
    """
    计算两个字符串数组内所有字符串的相似度(偏差距离/长字符串长度)，如果有空值，则相似度无效，赋值为0
    :param first_list: 字符串列表
    :param second_list: 字符串列表
    :return:
    """
    similarity = []  # 用于存储每个字符串比较后的相似度，
    for i in range(len(first_list)):
        # 将两个字符串分别代入函数进行变化步长和总长度的运算
        si, length = get_similarity_of_two_string(first_list[i], second_list[i])
        si = 1 - (si / length)
        similarity.append(si)

    return similarity


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：根据输入的数组和偏移位数，返回偏移数据
    修改时间：
'''


def get_shift_list_of_list(lists, shift_num, index=-1, inout=False):
    """
    根据输入的数组和偏移位数，返回偏移数据
    :param lists: 输入数组
    :param shift_num: 要移的位移
    :param index: 如果lists是多维数组，index表示为进行位移的元素索引
    :param inout: 决定返回的方式，FALSE是只返回偏移后的数组，TRUE是将偏移后的数据添加到原数组后，进行返回
    :return:
    """
    # 如果输入的数组为一维数组的时候
    if index == -1:
        shift_list = []
        for i in range(len(lists)):
            if i + shift_num >= len(lists):
                shift_list.append(None)
            else:
                shift_list.append(lists[i + shift_num])
    # 如果输入的数组为多维数组的时候
    else:
        shift_list = []
        for i in range(len(lists)):
            # 如果偏移后的索引超出了数组的长度，得到的偏移值为None
            if i + shift_num >= len(lists):
                # 如果需要将偏移后的数据添加到输入数组后进行输出
                if inout:
                    shift_list.append(None)
                # 如果直接输出偏移后的数组
                else:
                    lists[i].append(None)
            else:
                if inout:
                    shift_list.append(lists[i + shift_num][index])
                else:
                    lists[i].append(lists[i + shift_num])

    if inout:
        return shift_list
    else:
        return lists


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：提取出数组中的前N个最大或者最小的数据及其索引值
    修改时间：
'''


def get_topN_values_or_index(value_list, top_num, index_list=[], if_index=False, treat_type='max'):
    """
    提取出数组中的前N个最大或者最小的数据及其索引值
    :param value_list: 输入的数据数组
    :param top_num: 输入提取出几个数据
    :param index_list: 输入的索引数组
    :param if_index: 决定是否输出索引内容，FALSE为不输出，TRUE为输出
    :param treat_type: 决定是按最大还是最小输出
    :return:
    """
    # 对输入的数组进行快速排序，默认为正排
    if if_index:  # 判断是否输出对应的其他数据值，如果需要，就将该数据也进行同步排序
        basic_quick_sort(value_list, 0, len(value_list) - 1, index_list)
    else:  # 如果不需要只对输入数组进行排序
        basic_quick_sort(value_list, 0, len(value_list) - 1)
    # 如果输出最大的几个
    if treat_type == 'max':
        if value_list[-1] == '0':  # bug:
            if if_index:
                return 0, 0
            else:
                return 0
        else:
            if if_index:
                return value_list[((-1) * top_num):], index_list[((-1) * top_num):]
            else:
                return value_list[((-1) * top_num):]
    # 如果输出最小的几个
    else:
        if if_index:
            return value_list[:top_num], index_list[:top_num]
        else:
            return value_list[:top_num]


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：快速排序递归函数
    修改时间：
'''


def basic_quick_sort(values, start, end, index_list=[], element=-1):
    """
    快速排序递归函数
    :param element:如果输入的values为多维数组时，element表示通过第几个元素进行排序
    :param index_list:输入的索引数组，可以根据values内容，同步排序索引数组
    :param values: 输入的数组
    :param start: 起始元素下标
    :param end: 截止元素下标
    :return:
    """
    # 终止判断，如果起止和结尾下标一样，排序结束
    if start >= end:
        return
    # 进行分区排序，返回分区的中断点
    try:
        end_new = basic_partition(values, start, end, index_list=index_list, element=element)
    except:
        return
    # 通过中断点将数组分成两部分，分别进行进一步的区间排序和再拆分
    basic_quick_sort(values, start, end_new - 1, index_list=index_list, element=element)
    basic_quick_sort(values, end_new + 1, end, index_list=index_list, element=element)


'''
    创建时间：2022/3/3
    完成时间：2022/3/3
    功能：快速排序递归函数的分区函数
    修改时间：
'''


def basic_partition(values, start, end, index_list, element):
    """
    快速排序递归函数的分区函数
    :param element:每个元素参与排序的
    :param index_list:
    :param values:
    :param start:
    :param end:
    :return:
    """
    if element != -1:
        pivot = values[end][element]
    else:
        pivot = values[end]
    i = start

    for j in range(start, end):
        if element != -1:
            if values[j][element] < pivot:
                values[i], values[j] = values[j], values[i]
                if len(index_list) > 0:
                    index_list[i], index_list[j] = index_list[j], index_list[i]
                i += 1
        else:
            if values[j] < pivot:
                values[i], values[j] = values[j], values[i]
                if len(index_list) > 0:
                    index_list[i], index_list[j] = index_list[j], index_list[i]
                i += 1

    values[i], values[end] = values[end], values[i]

    if len(index_list) > 0:
        index_list[i], index_list[end] = index_list[end], index_list[i]

    return i


'''
    创建时间：2022/3/6
    完成时间：2022/3/6
    功能：将数组类型变量转换为字典类型
    修改时间：No.1 2022/4/22，添加了备注
'''


def basic_list_to_disc(lists, addType=''):
    """
    将数组类型变量转换为字典类型，key为上一个值，value为下一个值
    :param addType:值的类型选择，list为数组型，disc为字典类型，其他为本来类型
    :param lists:输入数组
    :return:
    """
    # 创建结果字典，初始为空
    disc_new = {}
    # 循环遍历数组内容
    for i in range(len(lists) - 1):
        # 如果一个值可能对应多个值，可选择合并方式
        if addType == 'list':
            disc_new[lists[i]] = [lists[i + 1]]
        elif addType == 'disc':
            disc_new[lists[i]] = {lists[i + 1]}
        else:
            disc_new[lists[i]] = lists[i + 1]
    return disc_new


'''
    创建时间：2022/3/6
    完成时间：2022/3/6
    功能：对比两个字典，以判断disc_target字典内容是否满足disc_origin字典内关系
    修改时间：No1. 2022/4/22，添加备注
'''


def basic_compare_disc_match_other(disc_origin, disc_target, value_list=[]):
    """
    对比两个字典，以判断disc_target字典内容是否满足disc_origin字典内关系
    :param value_list:补充对比数据，数组类型，作用：当标准字典的内容不同于目标字典时，会继续与补充数据对比，判断是否在补充数据中存在
    :param disc_origin:标准字典，用于对比判断
    :param disc_target:目标字典，待进行判断的内容
    :return:0为完整，1为不完整
    """
    # 创建判断结果字段，初始为0，其中0为完整，1为不完整
    result = 0
    # 如果目标字典为空，直接判断为不完整
    if len(disc_target) == 0:
        return 1
    # 循环遍历目标字典的所有收费单元ID
    for i, key in enumerate(disc_target.keys()):
        # 如果需要用到补充对比数据，则value_list不为空
        if value_list:
            # bug：这里与补充数据的对比范围固定为了5
            start = i - 5
            end = i + 5
            # start的数值必须大于等于0
            if start < 0:
                start = 0
            # end的数值必须小于补充数据的总长度
            if end == len(value_list):
                end = len(value_list) - 1
        try:
            # 如果进行补充数据的对比
            if value_list:
                # 判断目标字典中该key对应的值和标准字典中的是否有交集
                if disc_target[key].intersection(disc_origin[key]):
                    continue
                # 判断标准字典中该key对应的值是否存在于补充数据中
                elif set(value_list[start:end]).intersection(disc_origin[key]):
                    continue
                # 以上都不成立，判断不完整
                else:
                    result = 1
                    break
            # 如果不进行补充数据的对比
            else:
                # 判断目标字典中该key对应的值和标准字典中的是否有交集
                if disc_target[key].intersection(disc_origin[key]):
                    continue
                # 以上都不成立，判断不完整
                else:
                    result = 1
                    break
        # 如果字典匹配不到key值，直接跳过
        except Exception as e:
            # print(str(e))
            continue

    return result


'''
    创建时间：2022/3/7
    完成时间：2022/3/7
    功能：计算输入数组各要素的关联度（支持度和置信度）
    修改时间：
'''


# def basic_apriori(data, min_support=0, min_confidence=0, K=1, key_name=''):
#     """
#     计算输入数组各要素的关联度（支持度和置信度），返回大于最小支持度和提升度的数据
#     :param key_name:
#     :param K:
#     :param data: 各组合的数组数据
#     :param min_support: 最小支持度
#     :param min_confidence: 最小置信度
#     :return:
#     """
#     list_num = {}
#     first_num = {}
#     second_num = {}
#     for i in range(len(data)):
#         first = data[i][:K]
#         second = data[i][K:]
#         first = list_to_string_with_sign(first, '-')
#         second = list_to_string_with_sign(second, '-')
#         try:
#             list_num[first + '_' + second] += 1
#         except:
#             list_num[first + '_' + second] = 1
#         try:
#             first_num[first] += 1
#         except:
#             first_num[first] = 1
#         try:
#             second_num[second] += 1
#         except:
#             second_num[second] = 1
#
#     support = []
#     # confidence = []
#     for i, key in enumerate(list_num.keys()):
#         if list_num[key] > min_confidence:
#             if key_name == '':
#                 support.append([key.split('_')[0], key.split('_')[1], list_num[key] / first_num[key.split('_')[0]],
#                                 list_num[key] / second_num[key.split('_')[1]],
#                                 (list_num[key] / first_num[key.split('_')[0]]) * (
#                                             list_num[key] / second_num[key.split('_')[1]])])
#             else:
#                 support.append(
#                     [key_name, key.split('_')[0], key.split('_')[1], list_num[key] / first_num[key.split('_')[0]],
#                      list_num[key] / second_num[key.split('_')[1]],
#                      (list_num[key] / first_num[key.split('_')[0]]) * (
#                              list_num[key] / second_num[key.split('_')[1]])])
#         if (list_num[key] / len(data)) > min_support:
#             if key_name == '':
#                 support.append([key.split('_')[0], key.split('_')[1], list_num[key] / first_num[key.split('_')[0]],
#                                 list_num[key] / second_num[key.split('_')[1]],
#                                 (list_num[key] / first_num[key.split('_')[0]]) * (
#                                             list_num[key] / second_num[key.split('_')[1]])])
#             else:
#                 support.append(
#                     [key_name, key.split('_')[0], key.split('_')[1], list_num[key] / first_num[key.split('_')[0]],
#                      list_num[key] / second_num[key.split('_')[1]],
#                      (list_num[key] / first_num[key.split('_')[0]]) * (
#                              list_num[key] / second_num[key.split('_')[1]])])
#     return support


def basic_apriori_new(data, min_support=0, min_confidence=0, K='ALL', key_name='', ifSort=True, if_inout=False):
    """
    计算输入数组各要素的关联度（支持度、置信度和提升度），返回大于最小支持度和提升度的数据
    :param if_inout:是否进行in、out标识，默认为FALSE
    :param ifSort:是否考虑次序因素，默认为FALSE，TRUE为是，FALSE为否
    :param K:控制计算的组合的元素数量，ALL为全部组合进行计算
    :param key_name:主索引名称，用于加在输出数据的第一列，默认为空
    :param data: 各组合的数组数据
    :param min_support: 最小支持度
    :param min_confidence: 最小置信度
    :return:
    """
    # 用于存储每个组合的出现次数
    list_num = {}
    # 用于保存全部的组合内容
    all_thing = []
    # 遍历得到所有的组合池
    for i in range(len(data)):
        single_thing = []  # 用于存储每个数组的所有可能性组合
        # 获取第i各数组的所有可能组合，并放入single_thing里
        get_all_feature_compose(data[i], K='ALL', result_compose=single_thing, ifSort=ifSort, if_inout=if_inout)
        # 去除每个数组组合中重复的部分
        all_thing.extend(list(set(single_thing)))
        # 补充该数组全元素组合的情况
        all_thing.append(list_to_string_with_sign(data[i], '-'))
    # 通过组合池，计算出每个组合的出现次数，用于后续各关联度的计算
    for i in range(len(all_thing)):
        # 将每个组合情况数组转换为字符串
        try:
            list_num[all_thing[i]] += 1
        except:
            list_num[all_thing[i]] = 1

    # 用于存储各组合的支持度
    support = []
    # 用于存储各组合的前置信度
    confidence_first = []
    # 用于存储各组合的后置信度
    confidence_second = []
    # 用于存储各组合的提升度
    upValue_first = []
    # 用于存储各组合的出现次数
    total_num = []
    # 循环遍历所有组合，进行运算
    for key in list_num.keys():
        # 如果不计算所有组合的关联度
        if K != 'ALL':
            # 进行过滤，该组合的元素数等于K，才继续进行运算
            if len(key.split('-')) != K:
                continue

        # 如果该特征的支持度满足最小支持度，即计算并保存支持度、置信度和提升度
        if (list_num[key] / len(data)) >= min_support:
            # 计算并保存支持度
            # 如果不需要保存上主索引值
            if key_name == '':
                # 保存该组合的出现次数
                total_num.append(list_num[key])
                # 计算支持度并保存
                support.append([key, list_num[key] / len(data)])
            # 如果需要保存上主索引值
            else:
                support.append([key_name, key, list_num[key] / len(data)])

            # 如果该组合的元素数量为2个及以上
            if len(key.rsplit('-', 1)) > 1:
                # 如果组合是加上了in和out的标识
                if if_inout:
                    if key_name == '':
                        # 前置信度为该组合次数 比上 该组合排除最后一个元素的组合的次数
                        confidence_first.append([key, list_num[key] / list_num[key.rsplit('-', 1)[0] + '_in']])
                    else:
                        confidence_first.append(
                            [key_name, key, list_num[key] / list_num[key.rsplit('-', 1)[0] + '_in']])

                    # 保存后置信度
                    # if sort
                    if key_name == '':
                        # 后置信度为该组合次数 比上 该组合排除最后一个元素的次数
                        confidence_second.append([key, list_num[key] / list_num[key.rsplit('-', 1)[1] + '_out']])
                    else:
                        confidence_second.append(
                            [key_name, key, list_num[key] / list_num[key.rsplit('-', 1)[1] + '_out']])

                    # 计算并保存提升度
                    if key_name == '':
                        upValue_first.append([key, (list_num[key] / list_num[key.rsplit('-', 1)[0] + '_in']) / (
                                list_num[key.rsplit('-', 1)[1] + '_out'] / len(data))])
                    else:
                        upValue_first.append(
                            [key_name, key, (list_num[key] / list_num[key.rsplit('-', 1)[0] + '_in']) / (
                                    list_num[key.rsplit('-', 1)[1] + '_out'] / len(data))])
                # 如果组合中没有in和out标识
                else:
                    if key_name == '':
                        confidence_first.append([key, list_num[key] / list_num[key.rsplit('-', 1)[0]]])
                    else:
                        confidence_first.append([key_name, key, list_num[key] / list_num[key.rsplit('-', 1)[0]]])

                    # 保存后置信度
                    # if sort
                    if key_name == '':
                        confidence_second.append([key, list_num[key] / list_num[key.rsplit('-', 1)[1]]])
                    else:
                        confidence_second.append([key_name, key, list_num[key] / list_num[key.rsplit('-', 1)[1]]])

                    # 计算并保存提升度
                    if key_name == '':
                        upValue_first.append([key, (list_num[key] / list_num[key.rsplit('-', 1)[0]]) / (
                                list_num[key.rsplit('-', 1)[1]] / len(data))])
                    else:
                        upValue_first.append([key_name, key, (list_num[key] / list_num[key.rsplit('-', 1)[0]]) / (
                                list_num[key.rsplit('-', 1)[1]] / len(data))])
            # 如果该组合的元素数量为1个，则置信度和提升度均为0
            else:
                # 保存前置信度
                if key_name == '':
                    confidence_first.append([key, 0])
                else:
                    confidence_first.append([key_name, key, 0])

                # 保存后置信度
                if key_name == '':
                    confidence_second.append([key, 0])
                else:
                    confidence_second.append([key_name, key, 0])

                # 计算并保存提升度
                if key_name == '':
                    upValue_first.append([key, 0])
                else:
                    upValue_first.append([key_name, key, 0])

    return support, confidence_first, confidence_second, upValue_first, total_num


'''
    创建时间：2022/3/8
    完成时间：2022/3/8
    功能：将输入的数组中的所有的数组进行排序
    修改时间：
'''


def basic_sort_lists(lists, ascending=True):
    """
    将输入的数组中的所有的数组进行排序
    :param lists: 输入的多维数组
    :param ascending: 决定排序的类型，TRUE为正序，FALSE为倒序
    :return:
    """
    for i in range(len(lists)):
        basic_quick_sort(lists[i], 0, len(lists[i]) - 1)
    return lists


'''
    创建时间：2022/3/8
    完成时间：2022/3/8
    功能：对输入的数据进行关联度分析，返回满足最小支持度和最小置信度的数据及相应的支持度、置信度和提升度
    修改时间：
'''


def process_apriori(data, remove_order=False, min_support=0, min_confidence=0, K='ALL', key_name='', ifSort=True,
                    if_inout=False):
    """
    对输入的数据进行关联度分析，返回满足最小支持度和最小置信度的数据及相应的支持度、置信度和提升度
    :param if_inout:
    :param ifSort:
    :param remove_order: 是否去除次序因素，如果为FALSE，即不去除次序，如果为TRUE即去除
    :param data: 输入的数据，数组类型
    :param min_support: 最小支持度
    :param min_confidence: 最小置信度
    :param K: 分析模式，K为数字时，即分析K种变量的各关联度，K为ALL时，输出所有可能性的关联度值
    :param key_name: 返回的数据每行加上前置条件
    :return:
    """
    # 如果需要去除次序影响
    if remove_order:
        ifSort = False
        # if_inout = True
        data = basic_sort_lists(data)

    # 获得输入数据的支持度、置信度和提升度
    support, confidence_first, confidence_second, upValue_first, total_num = basic_apriori_new(data,
                                                                                               min_support=min_support,
                                                                                               min_confidence=min_confidence,
                                                                                               K=K, key_name=key_name,
                                                                                               ifSort=ifSort,
                                                                                               if_inout=if_inout)

    return support, confidence_first, confidence_second, upValue_first, total_num


'''
    创建时间：2021/09/18
    完成时间：2021/09/22
    功能：将输入的特征进行各种组合后输出,组合的特征数仅为总特征数减一个
    修改时间：No.1 2022/3/8，增加了len(feature) == 2的情况
'''


def feature_compose(feature, ifSort=False, if_inout=False):
    """
    将输入的特征进行各种组合后输出,组合的特征数仅为总特征数减一个
    :param if_inout:是否进行in、out标识，默认为FALSE
    :param ifSort:是否考虑次序因素，默认为FALSE
    :param feature: 特征数组
    :return:
    problem:
    """
    # 用于保存每个组合的值
    values_list = []
    # 如果输入数据为字符串，转换为数组
    if type(feature) != list:
        feature = feature.split('-')
    # 如果输入数组元素数小于1，直接返回该数组
    if len(feature) < 2:
        return feature
    # 如果输入数组元素数等于2，范别返回这两个元素
    elif len(feature) == 2:
        # 如果进行in、out标识
        if if_inout:
            # 将前一个元素带上in的结尾进行标识
            values_list.append(feature[0] + '_in')
            # 将后一个元素带上out的结尾进行标识
            values_list.append(feature[1] + '_out')
        # 如果不考虑顺序因素
        else:
            # 如果两个元素相同，只返回一个元素
            if feature[0] == feature[1]:
                values_list.append(feature[0])
            # 如果两个元素不相同，则返回两个元素
            else:
                values_list.append(feature[0])
                values_list.append(feature[1])
    # 如果是多个元素的情况下，则得到所有的组合情况
    else:
        # 如果考虑次序的作用，则只获取排除首尾元素的组合情况
        if ifSort:
            values_list.append(list_to_string_with_sign(feature[1:], '-'))
            values_list.append(list_to_string_with_sign(feature[:-1], '-'))
        # 如果不考虑次序的作用，则获取排除各单一元素的情况
        else:
            for i in range(len(feature)):
                if i == 0:
                    # 获取去掉第一个元素后的组合
                    values_list.append(list_to_string_with_sign(feature[i + 1:], '-'))
                else:
                    # 获取去掉第i各元素后的组合
                    values_list.append(list_to_string_with_sign(feature[:i] + feature[i + 1:], '-'))

    return values_list


'''
    创建时间：2022/3/8
    完成时间：2022/3/8
    功能：获得输入数组所有的组合可能性
    修改时间：
'''


def get_all_feature_compose(feature, K='ALL', result_compose=[], ifSort=True, if_inout=False):
    """
    获得输入数组所有的组合可能性
    :param if_inout:是否进行in、out标识，默认为FALSE
    :param ifSort:是否考虑次序因素，默认为FALSE
    :param K: 处理参数，如果为ALL，就将数组所有的可能性返回
    :param result_compose: 用于每次递归结果的保存
    :param feature: 输入特征数组
    :return:
    problem:
    """
    # 如果输入数据为字符串，则进行转换
    if type(feature) != list:
        feature = feature.split('-')
    # 如果输入数组为一个元素，直接终止递归
    if len(feature) <= 1:
        return
    # 如果K为ALL，则返回所有的组合情况
    if K == 'ALL':
        # 获取该输入数组的去除单一元素后的所有组合
        new_compose = feature_compose(feature, ifSort=ifSort, if_inout=if_inout)
        # 将所有的组合，保存入结果数组里
        result_compose.extend(feature_compose(feature, ifSort=ifSort, if_inout=if_inout))
        # 通过递归，分别将本次得到的所有组合，继续进行拆分组合
        for i in range(len(new_compose)):
            get_all_feature_compose(new_compose[i], K=K, result_compose=result_compose, if_inout=if_inout)

    return result_compose


'''
    创建时间：2022/3/11
    完成时间：2022/3/11
    功能：对输入的数组进行异常值去除，有3西格玛、IQR和Z分数方法
    修改时间：
'''


def get_list_without_abnormal(list_value_in, process_type):
    """
    对输入的数组进行异常值去除，有3西格玛、IQR和Z分数方法
    :param list_value_in: 输入的异常值
    :param process_type: 异常值去除方法，非状态分布用IQR(IQR)，正态分布用3西格玛(3SIGMA)和Z分数法(ZSCORE)
    :return:
    """
    # 进行数组的五分位值和均值的计算
    value_list = get_statistics_value_of_list(list_value_in, return_type='all', sortBySort=True)
    list_value = []
    deal_index = []
    if process_type == 'IQR':  # 通过四分位距进行判断，不需要数据满足正态分布
        k = 1.5  # 中度异常系数
        # k = 3  # 极度异常系数

        # 计算正常值域的上下限
        down_line = value_list[1] - k * (value_list[3] - value_list[1])  # 值域下限
        up_line = value_list[3] + k * (value_list[3] - value_list[1])  # 值域上限

        # 进行判断，如果下限低于0，赋值为0,
        # bug：目前的下限为0，只是针对业务中午负值的数据集的
        # if down_line < 1:
        #     down_line = 1
        #
        # if up_line > 60:
        #     up_line = 60

        # 对数组进行筛选

        for i in range(len(list_value_in)):
            # 通过倒序进行筛查，如果不符合直接删除
            if down_line <= float(list_value_in[i]) <= up_line:
                deal_index.append(i)
            # elif float(list_value[i]) < 1500 or float(list_value[i]) > 60000:
            #     drop_index.append(i)
        for i in range(len(deal_index)):
            list_value.append(list_value_in[deal_index[i]])
        list_value_in = []

    elif process_type == '3SIGMA':  # 也叫标准分数（standard score）,一个给定分数距离平均数多少个标准差，需要数据满足正态分布
        k = 3  # 设定为3倍标准差，即数据点如果大于或者小于均值的3倍标准差，即判断为异常值

        # 计算正常值域的上下限
        up_line = value_list[-1] + k * np.std(list_value_in)  # 值域上限
        down_line = value_list[-1] - k * np.std(list_value_in)  # 值域下限

        # 对数组进行筛选
        for i in range(len(list_value_in)):
            # 通过倒序进行筛查，如果不符合直接删除
            if down_line <= list_value[len(list_value) - i - 1] <= up_line:
                deal_index.append(i)
        for i in range(len(deal_index)):
            list_value.append(list_value_in[deal_index[i]])

    elif process_type == 'ZSCORE':  # 也叫标准分数（standard score）,一个给定分数距离平均数多少个标准差，需要数据满足正态分布
        k = 2  # 为异常阈值，经验法则是使用2、2.5、3或3.5作为阈值，如果k等于3，效果等同于3sigma
        for i in range(len(list_value_in)):
            # 通过倒序进行筛查，如果不符合直接删除
            if ((list_value_in[i] - value_list[-1]) / np.std(list_value)) <= k:
                deal_index.append(i)
        for i in range(len(deal_index)):
            list_value.append(list_value_in[deal_index[i]])

    return list_value


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能：对输入的字符串中的URL码进行转换
    修改时间：
'''


def transform_URL_to_UTF(string, encoding='gbk'):
    """
    对输入的字符串中的URL码进行转换
    :param encoding:
    :param string: 输入的字符串，包含URL码
    :return: 输出字符串，为转换后UTF码
    """
    return unquote(string, encoding=encoding)


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能：判断输入数据的某一个是不是对应着某个内容
    修改时间：
'''


def charge_someone_of_data_same_someting(data, disc_data, index, value):
    """
    判断输入数据的某一个是不是对应着某个内容
    :param index: 需要进行匹配的部分数据的下标，1.如果是数组，index就是下标，2.如果是字符串，为开始下标到截止下标
    :param data:输入数据,可以是字符串、数组
    :param disc_data:匹配的字典
    :param value:进行对比的数值，可以是数组，即判断是否在这个数组内
    :return:
    """
    if type(data) == list:
        # 如果输入的数据为数组
        try:
            # 从对应字典中获取匹配的值
            target_value = disc_data[data[index]]
            if type(value) == str:
                if target_value == value:
                    # 如果匹配的值与给定的值一致，返回0
                    return 0
                else:
                    # 不一致，返回1
                    return 1
            elif type(value) == list:
                if target_value in value:
                    # 如果匹配的值与给定的值一致，返回0
                    return 0
                else:
                    # 不一致，返回1
                    return 1
        except:
            # 如果字典中没有匹配的值，返回2
            return 2
    elif type(data) == str:
        # 如果输入的数据为字符串
        try:
            # 截取所需的字符串，并从对应字典中获取匹配的值
            target_value = disc_data[data[index[0]:index[1]]]
            if type(value) == str:
                if target_value == value:
                    return 0
                else:
                    return 1
            elif type(value) == list:
                if target_value in value:
                    return 0
                else:
                    return 1
        except:
            return 2


'''
    创建时间：2022/11/8
    完成时间：2022/11/8
    功能：获取输入值到目标值的距离（移动步数）
    关键字：字典，标准路径，距离
    修改时间：
'''


def compute_step_between_input_and_target(input, standard_dict, target):
    """
    获取输入值到目标值的距离（移动步数）
    :param input: 输入值
    :param standard_dict: 字典
    :param target: 目标对比值
    :return:
    """
    step = 0  # 用于计算移动步长
    # 循环计算，直到匹配的值中包含目标值
    if type(input) != list:
        input = [input]
    while True:
        # 判断输入值是否等于目标值
        if input[0] == target:
            return step
        else:
            step += 1
            list_ls = []
            for i in range(len(input)):
                try:
                    list_ls.extend(standard_dict[input[i]])
                except:
                    # 如果匹配不到，返回错误
                    continue
                # 如果下一个
                if target in list_ls:
                    return step
            input = []
            input.extend(list_ls)

            # 如果搜索步长超过了15，进行推出，返回超限
            if step > 15:
                return 'out'


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能：获取输入数据的在路径（字典）中之后几个的内容
    关键字：递归函数，路径追踪
    修改时间：
'''


def get_next_n_from_disc(target_string, disc_data, num, result_path=[], index_list=[], split_path=[], treat_type='split'):
    """
    获取输入数据的在路径（字典）中之后几个的内容
    :param target_string: 起始数据
    :param disc_data: 目标字典数据
    :param num: 获取的个数
    :return:
    """
    if treat_type == 'split':
        return_data = []  #
        if type(target_string) == str:
            if len(result_path) != 0:
                return_data.extend(result_path[0])
            # 终止判定，如果到了递归次数上限num或者触发限制条件(bug:限制条件固定)
            if num < 1 or len(target_string) < 16:
                return_data.append(target_string)
                split_path.append(return_data)
                return

            # 未达到终止条件且target_string为字符串的话，则进行下一步追踪
            return_data.append(target_string)
            if len(result_path) == 0:
                result_path.append(return_data)
            else:
                result_path[0] = return_data
            try:
                target_string_new = disc_data[target_string]
            except:
                # 如果匹配不到直接返回
                return
            num += -1
            # 判断获取的target_string个数
            if len(target_string_new) == 1:
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path[:1], split_path=split_path, treat_type=treat_type)
            elif len(target_string_new) == 2:
                result_path.append(return_data)
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path[:1], split_path=split_path, treat_type=treat_type)
                get_next_n_from_disc(target_string_new[1], disc_data, num, result_path[1:], split_path=split_path, treat_type=treat_type)
            else:
                result_path.append(return_data)
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path[:1], split_path=split_path, treat_type=treat_type)
                get_next_n_from_disc(target_string_new[1:], disc_data, num, result_path[1:], split_path=split_path, treat_type=treat_type)

        # 判断是否target_string是数组
        else:
            return_data.extend(result_path[0])
            result_path.append(return_data)
            # 如果是数组且长度为2，则直接拆成两个进行下一步递归
            if len(target_string) == 2:
                get_next_n_from_disc(target_string[0], disc_data, num, result_path[:1], split_path=split_path, treat_type=treat_type)
                get_next_n_from_disc(target_string[1], disc_data, num, result_path[1:], split_path=split_path, treat_type=treat_type)
            # 如果长度大于2，则拆成出第一个和其他进行下一步递归
            else:
                get_next_n_from_disc(target_string[0], disc_data, num, result_path[:1], split_path=split_path, treat_type=treat_type)
                get_next_n_from_disc(target_string[1:], disc_data, num, result_path[1:], split_path=split_path, treat_type=treat_type)
    else:
        if type(target_string) == str:
            # 终止判定，如果到了递归次数上限num或者触发限制条件(bug:限制条件固定)
            if num < 1 or len(target_string) < 16:
                result_path.append(target_string)
                index_list.append(num)
                return

            # 未达到终止条件且target_string为字符串的话，则进行下一步追踪
            result_path.append(target_string)
            index_list.append(num)
            try:
                target_string_new = disc_data[target_string]
            except:
                # 如果匹配不到直接返回
                return
            num += -1
            # 判断获取的target_string个数
            if len(target_string_new) == 1:
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path, index_list, treat_type=treat_type)
            elif len(target_string_new) == 2:
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path, index_list, treat_type=treat_type)
                get_next_n_from_disc(target_string_new[1], disc_data, num, result_path, index_list, treat_type=treat_type)
            else:
                get_next_n_from_disc(target_string_new[0], disc_data, num, result_path, index_list, treat_type=treat_type)
                get_next_n_from_disc(target_string_new[1:], disc_data, num, result_path, index_list, treat_type=treat_type)

        # 判断是否target_string是数组
        else:
            # 如果是数组且长度为2，则直接拆成两个进行下一步递归
            if len(target_string) == 2:
                get_next_n_from_disc(target_string[0], disc_data, num, result_path, index_list, treat_type=treat_type)
                get_next_n_from_disc(target_string[1], disc_data, num, result_path, index_list, treat_type=treat_type)
            # 如果长度大于2，则拆成出第一个和其他进行下一步递归
            else:
                get_next_n_from_disc(target_string[0], disc_data, num, result_path, index_list, treat_type=treat_type)
                get_next_n_from_disc(target_string[1:], disc_data, num, result_path, index_list, treat_type=treat_type)


'''
    创建时间：2022/4/8
    完成时间：2022/4/8
    功能：获取与输入数据对应的字典中的值
    修改时间：
'''


def get_corespondance_from_disc(in_data, disc_data):
    """
    获取与输入数据对应的字典中的值
    :param in_data: 数组类型数据
    :param disc_data: 目标字典数据
    :return:
    """
    return_data = []
    for i in range(len(in_data)):
        try:
            disc_thing = disc_data[in_data[i]]
            return_data.append(disc_thing)
        except:
            return_data.append('')
    return return_data


'''
    创建时间：2022/4/12
    完成时间：2022/4/12
    功能：对数字进行填补，并生成字符串
    修改时间：
'''


def get_fill_string(data, value, sign, num):
    """
    对数字进行填补，并生成字符串
    :param data:输入的待填补的数字
    :param value:进行比较的数字
    :param sign:进行填补的内容，可以是数字、字符或符号
    :param num:填补的数量
    :return:
    """
    str_data = ''
    if data < value:
        for i in range(num):
            str_data = str(sign) + str(data)
    else:
        str_data = str(data)
    return str_data


'''
    创建时间：2022/4/21
    完成时间：2022/4/21
    功能：将输入的数组按一定的比例差分，并返回每一部分的所属的区间组合
    修改时间：No.1 2022/6/16,增加了按照输入的数值边界进行切割的代码
'''


def basic_cut_data(data, cut_num=0, cut_rate=[], bins=[], remove_num=-1, cut_type='qcut'):
    """
    将输入的数组按一定的比例差分，并返回每一部分的所属的区间组合
    :param data: 进行拆分的数据，数组类型
    :param cut_num: 进行平均切分，为数字类型，非必填
    :param cut_rate: 各区间切分比例，为数组类型，与cut_num必须填写一个
    :param bins: 各区间的命名，数组类型
    :param remove_num: 不进行考虑的数值，可以为数字或者数组
    :return:
    """
    data_new = []
    if remove_num != -1:
        for i in range(len(data)):
            if type(remove_num) == list:
                if data[i] not in remove_num:
                    data_new.append(data[i])
                else:
                    continue
            else:
                if data[i] != remove_num:
                    data_new.append(data[i])
                else:
                    continue
    else:
        data_new = data

    # 对输入数组内容进行去重，并进行排序
    set_data = list(set(data_new))
    basic_quick_sort(set_data, 0, len(set_data) - 1)
    # 进行各数值的次数计算
    count_num = {}
    for i in range(len(data_new)):
        try:
            count_num[data_new[i]] += 1
        except:
            count_num[data_new[i]] = 1

    # 判断是按照频次分割还是按照数值分割
    if cut_type == 'qcut':
        # 保存每个区间对应的值
        scope_bins = {}
        # 如果通过cut_num进行平均分
        if cut_num != 0:
            # 生成各区间的值数组
            if len(bins) == 0:
                bins = [i + 1 for i in range(cut_num)]
            # 如果分割后的余数超过了分割数量的一般，则每个区间的数量多一个，否则各区间数量不增加
            if (len(data_new) % cut_num) / float(cut_num) >= 0.5:
                cut_num_list = int(len(data_new) / cut_num) + 1
            else:
                cut_num_list = int(len(data_new) / cut_num)

            num_next = 0  # 用于累加当次合计次数
            num_last = 0  # 记录上一次累加合计次数
            now_scope = 0  # 记录当前所处的区间
            scope_num = 0
            # 循环遍历所有的变量值set_data
            for i in range(len(set_data)):
                num_next += count_num[set_data[i]]
                # 计算每个区间的上限数值
                threshold_num = cut_num_list * (now_scope + 1)
                # 如果上限数值大于总数或者循环到最后一个数值，就将上限设为总数
                if now_scope == (cut_num - 1):
                    threshold_num = len(data_new)
                if i != len(set_data) - 1:
                    # 如果当前数值的累加次数已经超过了该区间的上限，进行如下判断
                    if num_next >= threshold_num:
                        # 判断当前数值的累加和上一个数值的累加哪一个接近区间阈值
                        if (num_next - threshold_num) > (threshold_num - num_last):
                            if scope_num == 0:  # 如果该区间还没有数值，则将当前数值赋给该区间
                                scope_bins[set_data[i]] = bins[now_scope]
                            else:  # 如果该区间还没有数值，则将当前数值赋给下一区间
                                scope_bins[set_data[i]] = bins[now_scope + 1]
                        else:
                            scope_bins[set_data[i]] = bins[now_scope]
                        # 进入下一个区间
                        now_scope += 1
                        scope_num = 0
                    # 如果当前数值的累加次数没有超过该区间的上限，直接进行赋值
                    else:
                        scope_bins[set_data[i]] = bins[now_scope]
                        scope_num += 1  # 记录该区间的数值个数
                else:
                    scope_bins[set_data[i]] = bins[now_scope]
                num_last = num_next * 1

        else:
            # 生成各区间的值数组
            if len(bins) == 0:
                bins = [i + 1 for i in range(len(cut_rate))]
            num_next = 0  # 用于累加当次合计次数
            num_last = 0  # 记录上一次累加合计次数
            now_scope = 0  # 记录当前所处的区间
            scope_num = 0
            # 循环遍历所有的变量值set_data
            for i in range(len(set_data)):
                num_next += count_num[set_data[i]]
                # 计算每个区间的上限数值
                if now_scope == len(cut_rate) - 1:
                    threshold_num = len(data_new)
                else:
                    threshold_num = int(cut_rate[now_scope] * len(data_new))
                if i != len(set_data) - 1:
                    if num_next >= threshold_num:
                        if (num_next - threshold_num) > (threshold_num - num_last):
                            if scope_num == 0:  # 如果该区间还没有数值，则将当前数值赋给该区间
                                scope_bins[set_data[i]] = bins[now_scope]
                            else:
                                scope_bins[set_data[i]] = bins[now_scope + 1]
                        else:
                            scope_bins[set_data[i]] = bins[now_scope]
                        now_scope += 1
                        scope_num = 0
                    else:
                        scope_bins[set_data[i]] = bins[now_scope]
                        scope_num += 1  # 记录该区间的数值个数
                else:
                    scope_bins[set_data[i]] = bins[now_scope]
                num_last = num_next * 1

        # 保存每个数值所在的区间
        data_scope = []
        for i in range(len(data)):
            if remove_num != -1:
                if type(remove_num) == list:
                    if data[i] in remove_num:
                        data_scope.append(0)

                    else:
                        data_scope.append(scope_bins[data[i]])

                else:
                    if data[i] == remove_num:
                        data_scope.append(0)

                    else:
                        data_scope.append(scope_bins[data[i]])
            else:
                data_scope.append(scope_bins[data[i]])

        return data_scope

    else:
        # 对输入数组内容进行去重，并进行排序
        data_scope = {}
        index = 0
        for i in range(len(set_data)):
            if (index + 1) == len(bins):
                if set_data[i] > cut_rate[index]:
                    try:
                        data_scope[bins[index]] += count_num[set_data[i]]
                    except:
                        data_scope[bins[index]] = count_num[set_data[i]]
            else:
                if set_data[i] <= cut_rate[index + 1]:
                    try:
                        data_scope[bins[index]] += count_num[set_data[i]]
                    except:
                        data_scope[bins[index]] = count_num[set_data[i]]
                else:
                    index += 1
                    try:
                        data_scope[bins[index]] += count_num[set_data[i]]
                    except:
                        data_scope[bins[index]] = count_num[set_data[i]]
        if (index + 1) < len(bins):
            data_scope[bins[index + 1]] = 0

        return data_scope


'''
    创建时间：2022/5/24
    完成时间：2022/5/24
    功能：将字典类型的数据进行保存为文件
    关键字：字典保存，字典转数组
    修改时间：No.1 2022/8/30，添加了key的拆分和保存
'''


def basic_save_dict_data(dict_data, save_path, if_many_key=False, if_many_values=False, key_sign='_', save_index=-1):
    """
    将字典类型的数据进行保存为文件
    :param if_many_values: 判断value是否是数组
    :param key_sign: 字典的key进行拆分时的分隔符
    :param save_path: 保存地址
    :param save_index: 如果为数据，可以输入下标，进行选择性保存，-1为全部保存
    :param if_many_key: 字典的key是否需要拆分保存
    :param dict_data: 目标字典数据
    :return:
    """
    list_data = []  # 保存字典转换后的数据
    # 遍历所有的key-value进行转换
    for key in dict_data.keys():
        if if_many_key:
            key_list = key.split(key_sign)
        else:
            key_list = [key]
        if if_many_values:
            if save_index == -1:
                if type(dict_data[key]) == list:
                    # 如果保存全部数据
                    # for i in range(len(dict_data[key])):
                        # if i == 0:
                        #     key_list.append(dict_data[key][i])
                        # else:
                        #     key_list = key_list[:-1]
                        #     key_list.append(dict_data[key][i])
                    key_list.extend(dict_data[key])
                else:
                    key_list.append(dict_data[key])
                list_data.append(key_list)
            else:
                key_list.append(dict_data[key][save_index])
                list_data.append(key_list)
        else:
            key_list.append(dict_data[key])
            list_data.append(key_list)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list_data)


'''
    创建时间：2022/5/24
    完成时间：2022/5/24
    功能：MinMax
    修改时间：
'''


def basic_Min_Max(value, min, max):
    if value < min:
        return 0
    elif value > max:
        return 1
    else:
        return (value - min) / (max - min)


'''
    创建时间：2022/6/7
    完成时间：2022/6/7
    功能：
    修改时间：
'''


def get_fitter(data, start_rate=0.025, end_rate=0.975):
    """

    :param data:
    :param start_rate:
    :param end_rate:
    :return:
    """
    # 通过fitter对象，进行六种分布函数的拟合，拟合时长上限为100s
    f = Fitter(data, distributions=['gamma', 'rayleigh', 'norm', 'erlang', 'logistic', 'dgamma',
                                    'genlogistic', 'loglaplace', 'burr12', 'johnsonsu', 'dweibull'],
               timeout=100)
    # f = Fitter(disc_data[key], distributions=['gamma', 'rayleigh', 'norm', 'erlang', 'logistic'],
    #            timeout=100)
    # 进行拟合训练
    f.fit()
    # 得到拟合度最好的函数的参数
    key_parameter = f.get_best()
    # 得到该函数的名称
    key_f = list(key_parameter)[0]
    # 得到该函数的参数
    parameter = list(key_parameter[key_f].values())
    # 进行函数名称的判断，并进行相应函数的95%置信度下的临界值
    if key_f == 'gamma':
        start_value = gamma.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = gamma.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = gamma.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = gamma.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [gamma.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]
    elif key_f == 'rayleigh':
        start_value = rayleigh.ppf(start_rate, parameter[0], parameter[1])
        end_value = rayleigh.ppf(end_rate, parameter[0], parameter[1])

        start = rayleigh.ppf(0.001, parameter[0], parameter[1])
        end = rayleigh.ppf(0.999, parameter[0], parameter[1])
        x_value = np.linspace(start, end, 200)
        y_value = [rayleigh.pdf(x, parameter[0], parameter[1]) for x in x_value]
    elif key_f == 'norm':
        start_value = norm.ppf(start_rate, parameter[0], parameter[1])
        end_value = norm.ppf(end_rate, parameter[0], parameter[1])

        start = norm.ppf(0.001, parameter[0], parameter[1])
        end = norm.ppf(0.999, parameter[0], parameter[1])
        x_value = np.linspace(start, end, 200)
        y_value = [norm.pdf(x, parameter[0], parameter[1]) for x in x_value]
    elif key_f == 'erlang':
        start_value = erlang.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = erlang.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = erlang.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = erlang.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [erlang.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]
    elif key_f == 'logistic':
        start_value = logistic.ppf(start_rate, parameter[0], parameter[1])
        end_value = logistic.ppf(end_rate, parameter[0], parameter[1])

        start = logistic.ppf(0.001, parameter[0], parameter[1])
        end = logistic.ppf(0.999, parameter[0], parameter[1])
        x_value = np.linspace(start, end, 200)
        y_value = [logistic.pdf(x, parameter[0], parameter[1]) for x in x_value]
    elif key_f == 'dgamma':
        start_value = dgamma.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = dgamma.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = dgamma.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = dgamma.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [dgamma.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]

    elif key_f == 'genlogistic':
        start_value = genlogistic.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = genlogistic.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = genlogistic.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = genlogistic.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [genlogistic.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]

    elif key_f == 'loglaplace':
        start_value = loglaplace.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = loglaplace.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = loglaplace.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = loglaplace.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [loglaplace.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]

    elif key_f == 'burr12':
        start_value = burr12.ppf(start_rate, parameter[0], parameter[1], parameter[2], parameter[3])
        end_value = burr12.ppf(end_rate, parameter[0], parameter[1], parameter[2], parameter[3])

        start = burr12.ppf(0.001, parameter[0], parameter[1], parameter[2], parameter[3])
        end = burr12.ppf(0.999, parameter[0], parameter[1], parameter[2], parameter[3])
        x_value = np.linspace(start, end, 200)
        y_value = [burr12.pdf(x, parameter[0], parameter[1], parameter[2], parameter[3]) for x in x_value]

    elif key_f == 'johnsonsu':
        start_value = johnsonsu.ppf(start_rate, parameter[0], parameter[1], parameter[2], parameter[3])
        end_value = johnsonsu.ppf(end_rate, parameter[0], parameter[1], parameter[2], parameter[3])

        start = johnsonsu.ppf(0.001, parameter[0], parameter[1], parameter[2], parameter[3])
        end = johnsonsu.ppf(0.999, parameter[0], parameter[1], parameter[2], parameter[3])
        x_value = np.linspace(start, end, 200)
        y_value = [johnsonsu.pdf(x, parameter[0], parameter[1], parameter[2], parameter[3]) for x in x_value]

    elif key_f == 'dweibull':
        start_value = dweibull.ppf(start_rate, parameter[0], parameter[1], parameter[2])
        end_value = dweibull.ppf(end_rate, parameter[0], parameter[1], parameter[2])

        start = dweibull.ppf(0.001, parameter[0], parameter[1], parameter[2])
        end = dweibull.ppf(0.999, parameter[0], parameter[1], parameter[2])
        x_value = np.linspace(start, end, 200)
        y_value = [dweibull.pdf(x, parameter[0], parameter[1], parameter[2]) for x in x_value]

    return start_value, end_value, x_value, y_value


'''
    创建时间：2022/6/16
    完成时间：2022/6/16
    功能：从文件中提取出指定列的值
    修改时间：
'''


def get_value_from_path(paths, feature, if_max=False):
    """
    从文件中提取出指定列的值
    :param paths: 地址集
    :param feature: 特征名称
    :param if_max: 是否反馈最大最小值
    :return:
    """
    feature_data = []
    if if_max:
        max = 0
        min = 0
    for k, path in enumerate(paths):
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:  # 如果是第一行，获取到对比列的索引
                    col_index = get_index_of_list(row, [feature])
                else:
                    if row[col_index[0]] == '':
                        continue
                    if if_max:
                        if max < float(row[col_index[0]]):
                            max = float(row[col_index[0]])
                        elif min > float(row[col_index[0]]):
                            min = float(row[col_index[0]])
                    feature_data.append(float(row[col_index[0]]))

    if if_max:
        return feature_data, max, min
    else:
        return feature_data


'''
    创建时间：2022/6/23
    完成时间：2022/6/23
    功能：根据提供的数组和标准字典，获取到标准数组组合
    修改时间：2022/6/30, change the start_station and end_station from first tollerate to station id
'''


def get_standard_data(origin_list, standard_dict, standard_back_dict, start_station, end_station):
    """
    根据提供的数组和标准字典，获取到标准数组组合
    :param end_station:
    :param start_station:
    :param origin_list:
    :param standard_dict:
    :return:
    """
    standard_list = []  # 用于保存最终的标准顺序内容
    stop = True  # 循环的终止字段
    now_value = 0  # 为当前指针指向的元素内容
    step = 0  # 用于记录当前的执行步数
    end_step = 4  # 记录达到最后的要素但是原始数组里边还有数据剩余，则再进行往下追寻的步数
    end_list = []  # 用于保存最后的几个临时节点
    while stop:
        # bug：这里与补充数据的对比范围固定为了6
        start = 0
        end = 6
        # end的数值必须小于补充数据的总长度
        if end >= len(origin_list):
            end = len(origin_list) - 1

        if step == 0:
            # standard_list.append(start_station)
            # try:
            #     # 判断目标字典中该key对应的值和标准字典中的是否有交集
            #     if {start_station}.intersection(set(origin_list[start:end])):
            #         origin_list.remove(start_station)
            # # 如果字典匹配不到key值，直接跳过
            # except Exception as e:
            #     print('error_1:', str(e))
            #     continue
            step += 1
            now_value = start_station

        elif now_value != end_station and len(origin_list) > 0:
            try:
                if type(now_value) == str:
                    try:
                        standard = standard_dict[now_value]
                    except:
                        now_value = origin_list.pop(0)
                        continue
                else:
                    standard = []
                    for i in range(len(now_value)):
                        try:
                            standard.extend(standard_dict[now_value[i]])
                        except:
                            continue
                    if len(standard) == 0:
                        now_value = origin_list.pop(0)
                        continue
                if_add = False  # 用来确定是否搜索到并已添加
                if end < 6:
                    origin_list_charge = origin_list[start:]
                else:
                    origin_list_charge = origin_list[start:end]
                for value in standard:
                    if value == origin_list[0] or value[:-2] == origin_list[0][:-2]:
                        standard_list.append(value)
                        origin_list.pop(0)
                        if_add = True
                        now_value = value
                        break
                    elif {value}.intersection(set(origin_list_charge)):
                        standard_list.append(value)
                        origin_list.remove(value)
                        if_add = True
                        now_value = value
                        break
                    elif {value[:-2]}.intersection(set([x[:-2] for x in origin_list_charge])):
                        standard_list.append(value)
                        if value[-2:] == '10':
                            value_new = value[:-2] + '20'
                        else:
                            value_new = value[:-2] + '10'
                        origin_list.remove(value_new)
                        if_add = True
                        now_value = value
                        break

                if if_add:
                    continue
                else:
                    standard_list.append(standard)
                    now_value = standard
            except Exception as e:
                print('error_2:', str(e))
        elif now_value != end_station and len(origin_list) <= 0:
            try:
                if type(now_value) == str:
                    try:
                        standard = standard_dict[now_value]
                    except:
                        now_value = origin_list.pop(0)
                        continue
                else:
                    standard = []
                    for i in range(len(now_value)):
                        try:
                            standard.extend(standard_dict[now_value[i]])
                        except:
                            continue

                if_add = False  # 用来确定是否搜索到并已添加
                if len(standard) == 0:
                    if_add = True
                    now_value = end_station
                else:
                    for value in standard:
                        if value == end_station:
                            if_add = True
                            now_value = value
                            break

                if if_add:
                    continue
                else:
                    if len(standard) == 1:
                        standard_list.append(standard[0])
                        now_value = standard[0]
                    else:
                        standard_list.append(standard)
                        now_value = standard
            except Exception as e:
                print('error_2:', str(e))

        elif (now_value == end_station or end_station in now_value) and len(origin_list) > 0:

            for i in range(end_step):
                try:
                    if type(now_value) == str:
                        standard = standard_dict[now_value]
                    else:
                        standard = []
                        for j in range(len(now_value)):
                            standard.extend(standard_dict[now_value[j]])
                    if_add = False  # 用来确定是否搜索到并已添加
                    if end < 6:
                        origin_list_charge = origin_list[start:]
                    else:
                        origin_list_charge = origin_list[start:end]
                    for value in standard:
                        if value == origin_list[0] or value[:-2] == origin_list[0][:-2]:
                            standard_list.extend(end_list)
                            end_list = []  # 清空临时数组
                            standard_list.append(value)
                            origin_list.pop(0)
                            if_add = True
                            now_value = value
                            end_step = 4
                            break
                        elif {value}.intersection(set(origin_list_charge)):
                            standard_list.extend(end_list)
                            end_list = []  # 清空临时数组
                            standard_list.append(value)
                            origin_list.remove(value)
                            if_add = True
                            now_value = value
                            end_step = 4
                            break
                        elif {value[:-2]}.intersection(set([x[:-2] for x in origin_list_charge])):
                            standard_list.extend(end_list)
                            end_list = []  # 清空临时数组
                            standard_list.append(value)
                            if value[-2:] == '10':
                                value = value[:-2] + '20'
                            else:
                                value = value[:-2] + '10'
                            origin_list.remove(value)
                            if_add = True
                            now_value = value
                            end_step = 4
                            break
                    if if_add:
                        i = 0
                        break
                    else:
                        end_list.append(standard)
                except Exception as e:
                    print('error_3:', str(e))
            if i == end_step - 1:
                break
        elif (now_value == end_station or end_station in now_value) and len(origin_list) == 0:
            break

    for i in range(len(standard_list)):
        if len(standard_list) - 1 - i == 0:
            break
        else:
            last_one = standard_list[len(standard_list) - 2 - i]
            now_one = standard_list[len(standard_list) - 1 - i]
            if type(last_one) == str:
                continue

            last_standard = []
            if type(now_one) == str:
                last_standard.extend(standard_back_dict[now_one])
            else:
                for k in range(len(now_one)):
                    last_standard.extend(standard_back_dict[now_one[k]])

            ls_list = []
            for j in range(len(last_one)):
                if last_one[j] in last_standard:
                    ls_list.append(last_one[j])
            if len(ls_list) == 0:
                print('error')
            elif len(ls_list) == 1:
                standard_list[len(standard_list) - 2 - i] = ls_list[0]
            else:
                standard_list[len(standard_list) - 2 - i] = ls_list

    return standard_list


'''
    创建时间：2022/8/1
    完成时间：2022/8/1
    功能：根据提供的文件地址和查找值，提供所需的结果，如统计数量、获取所需字段值
    修改时间：
'''


def get_result_from_data_with_colValue(paths, col_name, col_value, charge_thing='in', treat_type='count'):
    """
    根据提供的文件地址和查找值，提供所需的结果，如统计数量、获取所需字段值
    :param str charge_thing: 比较单位
    :param list paths: 文件路径
    :param list col_name: 匹配的字段名称
    :param list col_value: 匹配的值
    :param str treat_type: 处理方式，count为统计满足要求的数量
    :return:
    """
    if treat_type == 'count':
        data_result = {}
    for path in paths:
        print(path)
        with open(path) as f:
            for i, row in enumerate(f):
                row = row.split(',')
                row[-1] = row[-1][:-1]
                if i == 0:  # 如果是第一行，获取到对比列的索引
                    col_index = get_index_of_list(row, col_name)
                else:
                    for j, col in enumerate(col_index):
                        if treat_type == 'count':
                            if charge_thing == 'in':
                                if col_value[j] in row[col]:
                                    try:
                                        data_result[col_name[j] + '_' + str(col_value[j])] += 1
                                    except:
                                        data_result[col_name[j] + '_' + str(col_value[j])] = 1

    if treat_type == 'count':
        result = []
        for i in range(len(col_value)):
            try:
                result.append(data_result[col_name[i] + '_' + str(col_value[i])])
            except:
                result.append(0)
        return result


'''
    创建时间：2022/9/6
    完成时间：2022/9/6
    功能：进行字典类型数据的内部汇总计算
    关键字：字典、字典求和、字典汇总计算
    修改时间：No.1 2022/10/08, add avg compute
            No.2 2022/10/24，增加阈值过滤内容
'''


def compute_dict_by_group(data, index, treat_type, split_sign, filter_index=[], filter_value=[]):
    """
    进行字典类型数据的内部汇总计算
    :param filter_value: 过滤阈值，只有比这个阈值大的才参与运算
    :param filter_index: 过滤进行运算的目标下标，对应字典key分割后的下标
    :param str split_sign: key值分割符号
    :param dict data: 原始字典数据
    :param list index: 需要进行groupby的字典key中的内容下标
    :param str treat_type: 汇总处理的方式
    :return:
    """
    # 汇总计算结果字典
    if treat_type != 'total_sum':
        result_dict = {}
    else:
        result_dict = 0
    # 遍历字典的key值
    for key in data.keys():
        if split_sign == '':
            new_key = key
        else:
            # 将关键字进行拆分
            key_list = key.split(split_sign)

            # 如果过滤参数不为空，则进行判断，2022/10/24
            if len(filter_index) != 0:
                ifCompute = True
                for i in range(len(filter_index)):
                    if float(key_list[filter_index[i]]) >= filter_value[i]:
                        pass
                    else:
                        ifCompute = False
                        break
                if ifCompute:
                    pass
                else:
                    continue

            # 将groupby的内容组合起来
            new_key = ''
            for i in range(len(index)):
                if i < (len(index) - 1):
                    new_key += key_list[index[i]] + '_'
                else:
                    new_key += key_list[index[i]]
        # 进行汇总运算
        if treat_type == 'count':
            try:
                result_dict[new_key] += 1
            except:
                result_dict[new_key] = 1
        elif treat_type == 'sum':
            try:
                result_dict[new_key] += float(data[key])
            except:
                result_dict[new_key] = float(data[key])
        elif treat_type == 'avg':
            list_ls = [float(value) for value in data[key]]
            result_dict[new_key] = round(sum(list_ls) / len(list_ls), 2)
        elif treat_type == 'total_sum':
            result_dict += data[key]
        elif treat_type == 'min':
            list_ls = [float(value) for value in data[key]]
            result_dict[new_key] = min(list_ls)
        elif treat_type == 'max':
            list_ls = [float(value) for value in data[key]]
            result_dict[new_key] = max(list_ls)

    return result_dict


'''
    创建时间：2022/9/1
    完成时间：2022/9/5
    功能：两列数据相似度计算，应用基于形状的计算：Hausdorff(豪斯多夫距离)、Fréchet(弗雷歇距离)；基于分段的计算：OWD(单向距离)、
         LIP(多线位置距离)；基于点的计算：CS(向量空间余弦相似度)、LCSS(最长公共子序列)、DTW(动态时间归整算法)、ED(欧几里得距离)
    关键字：相似度、距离计算
    修改时间：
'''


def compute_distance_of_two_list(treat_type, list1, list2, parameter):
    """
    两列数据相似度计算，应用基于形状的计算：Hausdorff(豪斯多夫距离)、Fréchet(弗雷歇距离)；基于分段的计算：OWD(单向距离)、
    LIP(多线位置距离)；基于点的计算：CS(向量空间余弦相似度)、LCSS(最长公共子序列)、DTW(动态时间归整算法)、ED(欧几里得距离)
    :param dict parameter: 计算所需的参数集合
    :param str treat_type: 计算方式
    :param list list1: 数据数组1
    :param list list2: 数据数组2
    :return:
    """
    # Hausdorff算法：相近点最大距离计算
    if treat_type == 'Hausdorff':
        dtw.dtw(list1, list2)

    # DTW(动态时间归整算法)
    elif treat_type == 'DTW':
        # 计算得到两条数据的相似度值
        dist, cost, acc, path = dtw.dtw(list1, list2, parameter['dist'], parameter['warp'], parameter['w'],
                                        parameter['s'])

        return dist

    # Fréchet(弗雷歇距离)算法

    # OWD(单向距离)算法

    # LIP(多线位置距离)算法

    # CS(向量空间余弦相似度)算法

    # LCSS(最长公共子序列)算法

    # ED(欧几里得距离)算法


'''
    创建时间：2022/9/26
    完成时间：2022/9/26
    功能：根据提供的数据内容和相应概率，随机得到一个值并返回
    关键字：概率、随机
    修改时间：
'''


def get_feature_random(features, random_rate=[], random_type='', feature_type='str'):
    """
    根据提供的数据内容和相应概率，随机得到一个值并返回
    :param feature_type: 输出的随机数的类型，str为字符串型，是从features中选择输出的，int和float为数值型，为直接输出随机数，
    此时features为随机数生成器的输入参数
    :param random_type: 随机数生成器的类型
    :param list features: 进行随机返回的特征集
    :param list random_rate: 与特征集各特征对应的出现概率值，总和为1
    :return:
    """
    if feature_type == 'str':
        # 将概率集的值转换成从0到1的累加值
        random_rate_new = [sum(random_rate[:(i+1)]) if i != (len(random_rate) - 1) else sum(random_rate[:]) for i in range(len(random_rate))]
        # 获取0到1之间的随机数
        random_num = random.random()
        # 根据随机数落到的区间，来返回对应的特征值
        for i in range(len(random_rate_new)):
            if i == 0:
                if 0 <= random_num <= random_rate_new[i]:
                    result = features[i]
                    break
            else:
                if random_rate_new[i - 1] < random_num <= random_rate_new[i]:
                    result = features[i]
                    break
    else:
        # 如果要生成正态分布的随机数
        if random_type == 'normality':
            # 输入正态分布的参数，并输出随机数
            result = random.normalvariate(mu=features[0], sigma=features[1])
        # 如果为正常随机数
        elif random_type == 'random':
            # 生成指定范围内的随机整数
            if feature_type == 'int':
                result = random.randint(features[0], features[1])
            # 生成指定范围内的随机小数
            elif feature_type == 'float':
                result = random.uniform(features[0], features[1])

    # 返回结果
    return result


'''
    创建时间：2023/2/7
    完成时间：2023/2/7
    功能：根据提供的字符串和要拼接的下标，进行内容拼接
    关键字：
    修改时间：
'''


def gather_string_with_list(data, index_list, sign):
    """
    根据提供的字符串和要拼接的下标，进行内容拼接
    :param sign: 拼接的连接符
    :param data: 输入的合并数组
    :param index_list: 进行合并的元素的下标
    :return:
    """
    result_string = ''
    for i in range(len(index_list)):
        if i == 0:
            result_string += str(data[index_list[i]])
        else:
            result_string += sign + str(data[index_list[i]])

    return result_string



if __name__ == '__main__':
    x = [0, 1, 1, 1, 2, 2, 8, 7, 2, 3]
    y = [1, 1, 1, 2, 2, 8, 6, 3, 3, 2]
    parameter = {'dist': 'abs', 'warp': 1, 'w': 2, 's': 0.6}
    dist = compute_distance_of_two_list('DTW', x, y, parameter)
    print(dist)
