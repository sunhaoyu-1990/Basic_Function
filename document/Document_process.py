import csv

import pandas as pd
from data import Data_Basic_Function as dbf
import os

'''
本代码用于所有文件处理函数及过程
目前实现功能：1.指定文件的数据插入；2.文件读取；3.文件内容合并；4.文件内各列内容排序；5.读取path下，同一类的文件
'''


class Document_Action(object):
    """"""

    def __init__(self, path=None):
        """
        :param path: 文件地址
        """
        if path is not None:
            self.path = path
            # 获取文件的类型
            self.type = path.split('.')[-1]

    def load_path(self, path=None, parse_time=None):
        """
        :param path:文件地址
        :param parse_time:时间转换列
        :return: 返回读取到的文档内容，类型为DataFrame，若报错，返回None
        问题点：还无法读取全部类型数据
        """
        if path is not None:
            self.path = path
            self.type = path.split('.')[-1]
        try:
            if self.type == 'csv' or self.type == 'txt':
                try:
                    data = pd.read_csv(self.path, parse_dates=parse_time).drop(['Unnamed: 0'], axis=1)
                except:
                    data = pd.read_csv(self.path, parse_dates=parse_time)
            elif self.type == 'h5':
                data = pd.read_hdf(self.path, parse_dates=parse_time)
            elif self.type == 'xlsx':
                data = pd.read_excel(self.path, parse_dates=parse_time)
            elif self.type == 'json':
                data = pd.read_json(self.path, parse_dates=parse_time)
            else:
                with open(self.path, 'r') as file:
                    data = file.read()
        except:
            data = None

        return data

    def load_more_Document(self, name, path=None):
        """
        读取目录下同一类名字的多个文件
        :param name: 文件相同名称
        :param path: 新的目录地址，不输入，为默认地址
        :return:数据集，以数组形式返回
        """
        if path is not None:
            self.path = path
        static_path = self.path
        list_dir = os.listdir(self.path)
        data_list = []
        try:  # 循环文件名列表，
            for dir in list_dir:
                if dir.find(name) >= 0:
                    path1 = static_path + '/' + dir
                    data = self.load_path(path1)
                    data_list.append(data)
            return data_list
        except:
            return None

    def cut_(self):
        """
        去掉
        :return:
        """

    # 存储文档
    def Save_Document(self, data, path=None):
        """
        存储DataFrame类型的数据
        :param data:
        :param path: 保存位置
        :return: 无返回
        """
        if path is not None:
            self.type = path.split('.')[-1]
            self.path = path
        try:
            # 如果地址的结尾为CSV或TXT
            if self.type == 'csv' or self.type == 'txt':
                data.to_csv(self.path)
            elif self.type == 'h5':
                data.to_hdf(self.path)
            elif self.type == 'xlsx':
                data.to_excel(self.path)
            elif self.type == 'json':
                data.to_json(self.path)
            else:
                with open(self.path, 'w') as file:
                    file.write(data)
        except:
            print('保存失败')


'''
    创建时间：2021/11/24
    完成时间：2021/11/24
    功能: 根据输入的路径,获取所有该路径下的文件名,并组合成每个文件的路径名,输出路径数组
    修改时间：No.1 2022/6/16, 增加了replace_path输入参数，进行地址查找过程中，同时进行地址根目录替换
'''


def path_of_holder_document(path, sort=False, replace_path=''):
    """
    输入文件夹地址，返回文件夹下所有文件的完整地址数组
    :param path:
    :param sort: 是否对查询到的目录进行排序
    :param replace_path: 是否对查询到的地址的根目录进行替换，文件夹转移时使用，为空字符串时表示不进行，不为空时为替换的地址
    :param paths:文件夹地址
    :return:
    """
    paths_whole = os.listdir(path)
    replace_paths = []
    if sort:
        paths_whole = dbf.basic_sort_list(paths_whole)
    for i in range(len(paths_whole)):
        if '/' != path[-1]:
            paths_whole[i] = path + '/' + paths_whole[i]
            if replace_path != '':
                replace_paths.append(replace_path + '/' + paths_whole[i])
        else:
            paths_whole[i] = path + paths_whole[i]
            if replace_path != '':
                replace_paths.append(replace_path + paths_whole[i])
    if replace_path != '':
        return paths_whole, replace_paths
    else:
        return paths_whole


def cut_dir_list(path, cut_num, sort=False):
    """
    将提供的地址下的所有文件地址数组读取出来，并进行拆分，拆分个数为cut_num
    :param sort: 读取的文件地址内容是否排序
    :param path: 文件夹地址
    :param cut_num: 拆分个数
    :return:
    """
    if type(path) == list:
        paths_list = path
    else:
        paths_list = path_of_holder_document(path, sort)  # 读取地址下的所有文件地址，并以数组形式返回
    path_all = {}
    for i in range(cut_num):
        path_all[i + 1] = []  # 建立多个key-value，用于存放拆分后的地址
    for i in range(len(paths_list)):
        path_all[(i % cut_num) + 1].append(paths_list[i])
    return path_all


'''
    创建时间：2022/05/08
    完成时间：2022/05/08
    功能：根据输入的地址读取文件，对所需的特征所有数据进行相应的处理后，进行返回
    修改时间：2022/05/09, 新增了transform为delete的情况，即获取除了某个元素之外的内容——未测试
'''


def get_key_of_columns_with_transform(path, features, transform, parameters):
    """
    根据输入的地址读取文件，对所需的特征所有数据进行相应的处理后，进行返回
    :param path: 文件所在地址
    :param features: 所需要的特征名数组
    :param transform: 对应各特征的处理方式，可为数组，数组中每一个元素对个一个特征的处理方式
    :param parameters: 处理是所需要的参数
    :return:
    """
    vehicle = []
    columns_list = []
    with open(path) as f:
        for j, row in enumerate(f):
            row = row.replace('"', '')
            row = row.split(',')
            row[-1] = row[-1][:-1]
            if j == 0:
                columns_list = dbf.get_index_of_list(row, features)
            else:
                for i, col in enumerate(columns_list):
                    if transform[i] == 'left':
                        vehicle.append(row[col][:parameters[i]])
                    if transform[i] == 'right':
                        vehicle.append(row[col][parameters[i]:])
                    if transform[i] == 'delete':
                        str_ls = row[col][:parameters[i]]
                        str_ls += row[col][parameters[i] + 1:]
                        vehicle.append(str_ls)
    return vehicle


'''
    创建时间：2022/06/16
    完成时间：2022/06/16
    功能：获取一个文件夹下的所有文件
    修改时间：
'''


def get_all_file_from_dir(dir_path):
    """
    获取一个文件夹下的所有文件
    :param dir_path: 目标文件夹地址
    :return:
    """
    file_path = []  # 用于保存所有文件的地址
    fold_path = []  # 保存所有文件夹的相对地址
    # 获取该路径下的所有文件路径
    paths = path_of_holder_document(dir_path)
    paths_part = os.listdir(dir_path)
    # 将该路径下的文件取出
    for i in range(len(paths)):
        if os.path.isfile(paths[i]):
            file_path.append(paths[i])
        else:
            fold_path.append(paths_part[i])
            p1, p2 = get_all_file_from_dir(paths[i])
            fold_path.extend(p1)
            file_path.extend(p2)
    return fold_path, file_path


'''
    创建时间：2022/08/12
    完成时间：2022/08/12
    功能：load data use with
    修改时间：
'''


def load_data_with_path(path):
    """
    load data use with
    :param path:
    :return:
    """
    print(path)
    with open(path) as f:
        return f


'''
    创建时间：2022/08/12
    完成时间：2022/08/12
    功能：path filter
    修改时间：
'''


def get_path_with_filter(path, cut_length, filter_sign, filter_value):
    """
    path filter
    :param cut_length:
    :param path:
    :param filter_sign:
    :param filter_value:
    :return:
    """
    paths = path_of_holder_document(path, True)
    path_list = []
    for pa in paths:
        if filter_sign == '<':
            if pa[cut_length[0]:cut_length[1]] < filter_value:
                path_list.append(pa)
        elif filter_sign == '>':
            if pa[cut_length[0]:cut_length[1]] > filter_value:
                path_list.append(pa)
        elif filter_sign == '>=':
            if pa[cut_length[0]:cut_length[1]] >= filter_value:
                path_list.append(pa)
        elif filter_sign == '<=':
            if pa[cut_length[0]:cut_length[1]] <= filter_value:
                path_list.append(pa)
        elif filter_sign == '=':
            if pa[cut_length[0]:cut_length[1]] == filter_value:
                path_list.append(pa)
        elif filter_sign == '<>':
            if filter_value[0] < pa[cut_length[0]:cut_length[1]] < filter_value[1]:
                path_list.append(pa)

    return path_list


'''
    创建时间：2023/02/07
    完成时间：2023/02/07
    功能：对报错的关键信息进行保存
    修改时间：
'''


def save_data_of_important(data, save_path, save_type, charge_features=None, upload_features = None):
    """
    对报错的关键信息进行保存
    :param upload_features: 需要更新的数据字段
    :param charge_features: 查找的数据字段
    :param save_type: 数据保存形式，check_save为检查原有文件并更新，save为直接保存
    :param save_path: 存储地址
    :param data: 存储目标数据
    :return:
    """
    if save_type == 'check_save':
        length = len(charge_features)
        charge_features.extend(upload_features)
        # 获取原有数据
        origin_data = dbf.get_dict_from_document(save_path, charge_features, key_length=length, sign='_', ifIndex=False)
        # 遍历所有保存数据
        for i in range(len(data)):
            # 拼出对比内容
            charge_content = dbf.gather_string_with_list(data[i], charge_features[:length], '_')
            # 更新或者添加内容
            origin_data[charge_content] = dbf.get_values_of_list(data[i], upload_features)
            # 保存字典数据
            dbf.basic_save_dict_data(origin_data, save_path, length > 1, len(upload_features) > 1)

    elif save_type == 'save':
        # 直接保存数据
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

