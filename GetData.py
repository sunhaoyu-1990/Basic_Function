# 数据库操作算法包（主要是操作各类数据库的基本动作）
import csv
import pymysql
import cx_Oracle as cx
import pandas as pd


class OperationMysql:
    """
    Mysql数据库操作对象创建
    """
    def __init__(self, host, port, user, passwd, db_name, sql_type='mysql'):
        # 创建一个连接数据库的对象
        # 1.可以通过对象穿件时赋值数据库各参数；2.可以从已经存下的各数据库中选择各参数
        if sql_type == 'mysql':
            self.conn = pymysql.connect(
                # 部中心数据库
                # host='60.205.149.33',  # 连接的数据库服务器主机名
                # port=18607,  # 数据库端口号
                # user='cx',  # 数据库登录用户名
                # passwd='Lvtongcx2019',
                # db='freeway_appointment',

                # 陕西绿通车
                # host='60.205.149.33',  # 连接的数据库服务器主机名
                # port=3306,  # 数据库端口号
                # user='cx',  # 数据库登录用户名
                # passwd='cd2017',
                # db='lvtongche',

                # 收费中心
                # host='172.16.1.54',
                # port=3306,  # 数据库端口号
                # user='zhidu',  # 数据库登录用户名
                # passwd='sjfx2018',
                # db='road_infrastructure',  # 数据库名称

                # 测试数据库
                # host='192.168.0.95',
                # port=3306,  # 数据库端口号
                # user='root',  # 数据库登录用户名
                # passwd='123',
                # db='freeway_appointment',  # 数据库名称

                # 全路网稽查数据库
                # host='192.168.0.182',
                # port=3306,  # 数据库端口号
                # user='root',  # 数据库登录用户名
                # passwd='123456',
                # db='vehicle_check',  # 数据库名称

                host=host,
                port=port,  # 数据库端口号
                user=user,  # 数据库登录用户名
                passwd=passwd,
                db=db_name,  # 数据库名称

                charset='utf8',  # 连接编码
                cursorclass=pymysql.cursors.DictCursor
            )
            # 使用cursor()方法创建一个游标对象，用于操作数据库
            self.cur = self.conn.cursor()
        else:
            # self.conn = cx.connect(user, passwd, host + '/' + db_name, encoding='utf8')
            self.conn = cx.connect(user, passwd, host + ':' + str(port) + '/' + db_name)
            # 使用cursor()方法创建一个游标对象，用于操作数据库
            self.cur = self.conn.cursor()

    # 进行数据库的查询,2022/10/31 update
    def search_one(self, sql_string, ifEnd=True):
        result = []
        try:
            self.cur.execute(sql_string)
            result = self.cur.fetchall()  # 使用 fetchall()方法获取所有数据.只显示一行结果
        except Exception as e:
            print("search mysql string is:" + sql_string)
            print("search mysql error:" + str(e))
        finally:
            if ifEnd:
                self.cur.close()
        return result

    # 进行数据库的修改操作
    def update_data(self, sql_string, ifEnd=True):
        error = []  #
        # 开始进行数据库写入
        try:
            # 开始事务
            self.conn.begin()
            # 进行sql写入
            self.cur.execute(sql_string)
            # 提交事务
            self.conn.commit()
        except Exception as e:
            # 如果有错误就回滚事务.ap
            error.append([str(e), sql_string])
            print('error：', e)
        if ifEnd:  # 如果没有sql语句执行了，关闭cur
            self.cur.close()

        # 进行数据库执行报错的信息保存
        if len(error) != 0:
            with open('./4.data_check/November/data_for_check/error.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(error)

    # 自动生成上传语句进行数据表的写入
    def write_one(self, table_name, index_row, value_row, type_list, data_type='csv', save_num=500):
        """
        根据输入的数据表名称，插入列名称，插入列字段类型和插入数值，进行一条或多条数据的写入，如果save_type为csv，即从csv读取到的数据存入数据库，但此时没有了数据类型，均成了字符串，所以需要进行各数据类型的转换
        :param int save_num: 一次写入的数据量大小，为0时为一次性全部写入
        :param str data_type: 数据表存入模式，如果为csv则表示数据源是csv文件，需要进行格式转换，如果为code则表示为算法输出数据，直接转换为字符串
        :param list type_list: 插入列字段类型，用于对csv读取到的数据进行类型转换
        :param str table_name:需要插入的数据表名称
        :param list index_row:插入列名称
        :param list value_row:插入数值，为二维数组，如果插入的数值只有一条，进行单条写入，如果为多条，进行多条写入
        :return:
        """
        # 应用的SQL插入语句为insert into table_name(列1，...) values(值1，...),(值2，...)...
        index_string = '('  # 用于进行字符串拼接
        for i in range(len(index_row)):  # 遍历列名称数组，进行字符串拼接
            if i < len(index_row) - 1:  # 不是最后一个元素的时候，结尾拼接“，”
                index_string = index_string + str(index_row[i]) + ','
            else:  # 是最后一个元素的时候，结尾拼接“）”
                index_string = index_string + str(index_row[i]) + ')'

        value_string = ''  # 用于进行插入值的字符串拼接

        error = []

        for i in range(len(value_row)):  # 遍历所有要插入的数值内容
            if data_type == 'csv':  # 如果
                for j in range(len(value_row[i])):  # 遍历每一个数值数组内的元素，进行字符串拼接
                    if len(value_row[i]) == 1:
                        if value_row[i][j] != '' or value_row[i][j] != "":
                            if type_list[j] == 'varchar' or type_list[j] == 'datetime':
                                value_string = value_string + '("' + str(value_row[i][j]) + '")'
                            elif type_list[j] == 'float':
                                try:
                                    value_string = value_string + str(float(value_row[i][j])) + ')'
                                except:
                                    value_string = value_string + '0.0)'
                            elif type_list[j] == 'int' or type_list[j] == 'tinyint':
                                value_string = value_string + '(' + str(int(value_row[i][j])) + ')'
                        else:  # 如果该元素为空，就拼接None
                            value_string += '(NULL)'
                    elif j == 0:
                        if value_row[i][j] != '' or value_row[i][j] != "":
                            if type_list[j] == 'varchar' or type_list[j] == 'datetime':
                                value_string = value_string + '("' + str(value_row[i][j]) + '",'
                            elif type_list[j] == 'float':
                                try:
                                    value_string = value_string + str(float(value_row[i][j])) + ','
                                except:
                                    value_string = value_string + '0.0,'
                            elif type_list[j] == 'int' or type_list[j] == 'tinyint':
                                value_string = value_string + '(' + str(int(value_row[i][j])) + ','
                        else:  # 如果该元素为空，就拼接None
                            value_string += '(NULL,'
                    elif j < len(value_row[i]) - 1:  # 不是最后一个元素的时候，结尾拼接“，”
                        if value_row[i][j] != '' or value_row[i][j] != "":
                            if type_list[j] == 'varchar' or type_list[j] == 'datetime':
                                value_string = value_string + '"' + str(value_row[i][j]) + '",'
                            elif type_list[j] == 'float':
                                try:
                                    value_string = value_string + str(float(value_row[i][j])) + ','
                                except:
                                    value_string = value_string + '0.0,'
                            elif type_list[j] == 'int' or type_list[j] == 'tinyint':
                                try:
                                    value_string = value_string + str(int(float(value_row[i][j]))) + ','
                                except:
                                    value_string = value_string + '0,'
                        else:  # 如果该元素为空，就拼接None
                            value_string += 'NULL,'
                    else:  # 是最后一个元素的时候，结尾拼接“）”
                        if value_row[i][j] != '' or value_row[i][j] != "":
                            if type_list[j] == 'varchar' or type_list[j] == 'datetime':
                                value_string = value_string + '"' + str(value_row[i][j]) + '")'
                            elif type_list[j] == 'float':
                                try:
                                    value_string = value_string + str(float(value_row[i][j])) + ')'
                                except:
                                    value_string = value_string + '0.0)'
                            elif type_list[j] == 'int' or type_list[j] == 'tinyint':
                                if value_row[i][j] == '0.0':
                                    value_string = value_string +'0)'
                                else:
                                    value_string = value_string + str(int(float(value_row[i][j]))) + ')'
                        else:
                            value_string += 'NULL)'

            else:
                value_string = value_string + str(tuple(value_row[i])).replace("''", "NULL")

            if (i % save_num == 0 and i > 0) or i == len(value_row) - 1:
                # 进行sql插入语句的生成
                sql_string = 'insert into ' + table_name + ' ' + index_string + ' values' + value_string

                # 开始进行数据库写入
                try:
                    # 开始事务
                    self.conn.begin()
                    # 进行sql写入
                    self.cur.execute(sql_string)
                    # 提交事务
                    self.conn.commit()
                except Exception as e:
                    # 如果有错误就回滚事务.ap
                    error.append([str(e)])
                    print('error：', e)
                # finally:
                #     # 最后进行游标对象的关闭
                #     self.cur.close()
                value_string = ''
                print(i)

            elif i < len(value_row) - 1:  # 如果不是最后一个要插入的数值组的时候，结尾拼接“，”
                value_string += ','

        self.cur.close()

        with open('./error/error.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(error)

    # 设置所需数据的起止时间，实现数据库自动获取
    def get_data_in_for(self, beginTime, endTime, save_path, save_type='noOneDay', get_type='LT', sql_str='', back_path=''):
        """
        设置所需数据的起止时间，实现数据库自动获取
        :param get_type:
        :param save_type:
        :param save_path: 存储位置，到文件夹
        :param beginTime: 起始时间，格式为YYYYMMDD
        :param endTime: 截止时间，格式为YYYYMMDD
        :return: 自动保存到save_path，一个月一个文件
        """
        # 获取截止时间和起始时间的年份跨度
        year_gap = int(endTime[:4]) - int(beginTime[:4])
        # 将截止的月份与年份跨度相加，得到循环的截止数
        endNum = int(endTime[4:6]) + 12 * year_gap
        # 得到循环的起始数
        beginNum = int(beginTime[4:6])
        # 给每次循环的起止年份进行赋值
        yearj = int(beginTime[:4])
        yearjj = int(beginTime[:4])
        for i in range(beginNum, endNum):
            # 每次循环，重新赋值起止年份
            yearj += int((i - 1) / 12)
            # 每次循环，重新赋值起止月份
            j = i - 12 * int((i - 1) / 12)
            yearjj += int(i / 12)
            jj = (i + 1) - 12 * int(i / 12)

            # 获取Day数据
            if i != beginNum:
                startDay = '01'
            else:
                startDay = beginTime[6:]
            if i != endNum - 1:
                endDay = '01'
            else:
                endDay = endTime[6:]

                # 如果起始月份小于10，就补0，并转换为字符串
            if j < 10:
                j = '0' + str(j)
            else:
                j = str(j)
            # 如果截止月份小于10，就补0，并转换为字符串
            if jj < 10:
                jj = '0' + str(jj)
            else:
                jj = str(jj)

            # 获取全国从2021年9月到11月的数据
            if save_type == 'noOneDay':
                sql_str = "SELECT t1.`VEHICLE_CLASS`,t1.`APPOINTMENT_START_TRANSPORT_TIME`,t1.`APPOINTMENT_END_TRANSPORT_TIME`,t1.`APPOINTMENT_START_TOLLGATE_ID`,t1.`APPOINTMENT_END_TOLLGATE_ID`,t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`OWNER`,t1.`EXIT_LANE`,t1.`GROUP_ID`,t1.`EN_WEIGHT`,t1.`EX_WEIGHT`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`FEE`,t1.`REASON`,t1.`MEMO`,t1.`EXEMPTION`,t1.`UPDATE_BY`,t1.`UPDATE_TIME`,t1.`IS_VALID`,t1.`STATUS`,t2.`TYPE_ID` FROM `transport` t1,`transport_check_loaded_freight` t2 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t1.`CHECK_TIME`>='" + str(
                    yearj) + "-" + j + "-" + startDay + " 00:00:00' AND t1.`CHECK_TIME`<'" + str(
                    yearjj) + "-" + jj + "-" + endDay + " 00:00:00';"
                # 根据起止时间生成每次的保存文件名称
                save_path_new = save_path + '/' + str(yearj) + j + startDay + '-' + str(yearjj) + jj + endDay + '.csv'
            else:
                if get_type == 'LT':
                    sql_str = sql_str + "'" + beginTime + "' AND t1.`CHECK_TIME`<='" + endTime + "';"
                else:
                    sql_str = sql_str + "'" + beginTime + "' AND `TRANSTIME`<='" + endTime + "';"
                # 根据起止时间生成每次的保存文件名称
                save_path_new = save_path + beginTime[:9] + '.csv'
                # 根据起止时间生成每次的备份文件名称
                back_path_new = back_path + beginTime[:9] + '.csv'
            # 获取MySQL数据库数据
            # 获取pub_organ下所有收费站的数据
            if i == endNum - 1:  # 如果是最后一个，则在获取数据后，关闭游标
                ifEnd = True
            else:
                ifEnd = False
            data = self.search_one(sql_str, ifEnd=ifEnd)
            data = pd.DataFrame(data)
            data.to_csv(save_path_new)

    # 根据输入的sql语句，进行数据获取
    def load_data(self, sql_string):
        data = self.search_one(sql_string)
        data = pd.DataFrame(data)
        return data


if __name__ == '__main__':

    # 获取到从4月1日到现在所有苹果的运输数据（没有省份名称）
    sql = 'SELECT * FROM transport t LEFT JOIN transport_appointment_loaded_freight t2 ON t.`ID`=t2.`TRANSPORT_ID` WHERE t.`CHECK_RECEIVE_TIME`>"2021-04-01 00:00:00" AND t2.`TYPE_ID`="20101";'
    # 获取到从4月1日到现在所有苹果的运输数据（有省份名称）
    sql2 = 'SELECT * FROM transport t LEFT JOIN transport_appointment_loaded_freight t2 ON t.`ID`=t2.`TRANSPORT_ID` LEFT JOIN tollgate t3 ON t.`EXIT_TOLLGATE_ID`=t3.`ID` LEFT JOIN district d ON t3.`DISTRICT_ID`=d.`ID` WHERE t.`CHECK_RECEIVE_TIME`>"2021-04-01 00:00:00" AND t2.`TYPE_ID`="20101";'
    # 获取到从4月1日到现在所有苹果的运输数据（去除掉同省内运输的数据）
    sql3 = 'SELECT * FROM transport t LEFT JOIN transport_appointment_loaded_freight t2 ON t.`ID`=t2.`TRANSPORT_ID` LEFT JOIN tollgate t3 ON t.`EXIT_TOLLGATE_ID`=t3.`ID` LEFT JOIN tollgate t4 ON t.`ENTRANCE_TOLLGATE_ID`=t4.`ID` LEFT JOIN district d ON LEFT(t3.`DISTRICT_ID`,2)=d.`ID` LEFT JOIN district d2 ON LEFT(t4.`DISTRICT_ID`,2)=d2.`ID` WHERE t.`CHECK_RECEIVE_TIME`>"2020-05-06 00:00:00" AND t.`CHECK_RECEIVE_TIME`<"2021-05-10 00:00:00" AND t2.`TYPE_ID`="20101" AND d.`NAME`!=d2.`NAME`;'
    # 获取到从2020年5月6日到2021年5月10日所有苹果的输入到陕西或输出到陕西各县的运输数据
    sql4 = 'SELECT * FROM transport t LEFT JOIN transport_appointment_loaded_freight t2 ON t.`ID`=t2.`TRANSPORT_ID` LEFT JOIN tollgate t3 ON t.`EXIT_TOLLGATE_ID`=t3.`ID` LEFT JOIN tollgate t4 ON t.`ENTRANCE_TOLLGATE_ID`=t4.`ID` LEFT JOIN district d ON t3.`DISTRICT_ID`=d.`ID` LEFT JOIN district d2 ON t4.`DISTRICT_ID`=d2.`ID` WHERE t.`CHECK_RECEIVE_TIME`>"2020-05-06 00:00:00" AND t.`CHECK_RECEIVE_TIME`<"2021-05-10 00:00:00" AND t2.`TYPE_ID`="20101" AND d.`NAME`!=d2.`NAME` AND LEFT(t.`ENTRANCE_TOLLGATE_ID`,2)="61" AND LEFT(t.`EXIT_TOLLGATE_ID`,2)="61";'
    # 获取到从2020年5月6日到2021年5月10日所有苹果的运输数据（未除掉同省内运输的数据）
    sql5 = 'SELECT * FROM transport t LEFT JOIN transport_appointment_loaded_freight t2 ON t.`ID`=t2.`TRANSPORT_ID` LEFT JOIN tollgate t3 ON t.`EXIT_TOLLGATE_ID`=t3.`ID` LEFT JOIN tollgate t4 ON t.`ENTRANCE_TOLLGATE_ID`=t4.`ID` LEFT JOIN district d ON LEFT(t3.`DISTRICT_ID`,2)=d.`ID` LEFT JOIN district d2 ON LEFT(t4.`DISTRICT_ID`,2)=d2.`ID` WHERE t.`CHECK_RECEIVE_TIME`>"2020-05-06 00:00:00" AND t.`CHECK_RECEIVE_TIME`<"2021-05-10 00:00:00" AND t2.`TYPE_ID`="20101" AND t.`ENTRANCE_TOLLGATE_ID`!=t.`EXIT_TOLLGATE_ID`;'
    # 获取到从2020年5月6日到2021年5月10日所有苹果的运输数据（未除掉同省内运输的数据）
    sql6 = "SELECT * FROM `transport` t1 LEFT JOIN `transport_check_loaded_freight` t2 ON t1.`ID`=t2.`TRANSPORT_ID` WHERE t2.`TYPE_ID`='20101' AND t1.`CHECK_TIME`>='2020-05-07 00:00:00' AND t1.`CHECK_TIME`<='2020-05-08 23:59:59' AND (t1.`ENTRANCE_TOLLGATE_ID` LIKE '61%' OR t1.`EXIT_TOLLGATE_ID` LIKE '61%');"
    # 获取到从2020年5月6日到2021年5月6日全国所有苹果的运输数据（产地细化到县的）
    sql7 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS` FROM `transport` t1,`transport_check_loaded_freight` t2 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t2.`TYPE_ID`='20101' AND t1.`CHECK_TIME`>='2021-02-06 00:00:00' AND t1.`CHECK_TIME`<='2021-05-06 00:00:00';"

    # 过去全国各收费站数据
    sql8 = "SELECT `ID`,`NAME`,`DISTRICT_ID` FROM `tollgate`"

    # 获取各省市表数据
    sql9 = "SELECT `ID`,`NAME` FROM `district`"

    sql10 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t3.`NAME`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS` FROM `transport` t1,`transport_check_loaded_freight` t2,`tollgate` t3 WHERE t1.`ENTRANCE_TOLLGATE_ID`=t3.`ID` AND t1.`ID`=t2.`TRANSPORT_ID` AND t2.`TYPE_ID`='20101' AND (t3.`CODE`='G0091210010080' OR t3.`CODE`='G0001210040050');"

    # 获取全国全水果数据
    sql11 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t2.`TYPE_ID`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS` FROM `transport` t1,`transport_check_loaded_freight` t2 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t1.`CHECK_TIME`>='2020-05-06 00:00:00' AND t1.`CHECK_TIME`<='2020-06-06 00:00:00';"

    # 获取陕西绿通数据
    sql13 = "SELECT t1.`ID`,t1.`VEHICLE_CLASS`,t1.`INSPECTOR`,t1.`COME_FROM`,t1.`TOLLGATE_ID`,t1.`AXIS`,t1.`CRATE_TYPE`,t1.`VEHICLE_TYPE`,t1.`PLATE_NUMBER`,t1.`TOLLGATE_NAME`,t1.`CHEAT_TYPE`,t1.`CHEAT_DESC`,t1.`CHECK_RESULT`,t1.`PASS_TIME`,t1.`MEDIA_TYPE`,t1.`TRANSACTION_TIME`,t1.`PAY_FEE`,t1.`GROUP_ID`,t1.`TON`,t1.`SUBJECTIVE_CHEAT`,t1.`REVIEWER_ID`,t1.`UPDATE_BY`,t1.`IN_USE`,t1.`STATUS`,t1.`OPERATE_SECOND`,t1.`TAKE_PHOTO_SECOND`,t1.`PASS_ID`,t1.`IS_EMERGENCY`,t2.`TYPE_ID` FROM `record_waybill` t1,`loaded_freight` t2 WHERE t1.`ID`=t2.`WAYBILL_ID` AND t1.`PASS_TIME`>='2015-09-05 00:00:00' AND t1.`PASS_TIME`<='2016-09-05 00:00:00';"

    sql14 = "SELECT t1.*,t2.`TYPE_ID` FROM `record_waybill` t1,`loaded_freight` t2 WHERE t1.`ID`=t2.`WAYBILL_ID` AND t1.`PASS_TIME`>='2018-08-05 00:00:00' AND t1.`PASS_TIME`<='2018-12-05 00:00:00';"

    # 获取所有水果种类数据
    sql12 = "SELECT t.ID,t.NAME FROM `freight_type` AS t WHERE t.IN_USE='1'"

    # 获取照片数据表
    sql15 = "SELECT * FROM `pass_evidence` AS p WHERE p.`TIME`>='2016-05-05 00:00:00' AND p.`TIME`<='2016-06-05 00:00:00';"

    # 获取陕西绿通的全品种数据-freight_type
    sql17 = "SELECT ID,NAME,DEL_FLAG FROM `freight_type`;"

    # 获取照片数据表
    sql18 = "SELECT t1.`ID`,t1.`PLATE_NUMBER`,t1.`PASS_TIME`,t1.`TOLLS` FROM `record_waybill` t1 WHERE t1.`PASS_TIME`>='2020-06-05 00:00:00' AND t1.`PASS_TIME`<='2021-06-05 00:00:00';"

    # 获取陕西绿通车的箱型数据
    sql19 = "SELECT t1.`PLATE_NUMBER`,t1.`PASS_TIME`,t1.`CRATE_TYPE`,t1.`IN_USE` FROM `record_waybill` t1 WHERE t1.`PASS_TIME`>='2020-06-05 00:00:00' AND t1.`PASS_TIME`<='2021-06-05 00:00:00';"

    # 获取门架车流量数据
    sql20 = "SELECT * FROM auditdata.grantrysectdata20210528 as g WHERE g.traderesult=0 and g.roadid='G3001' limit 1"

    # 获取全国从2021年5月到9月的数据
    sql21 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS`,t2.`TYPE_ID` FROM `transport` t1,`transport_check_loaded_freight` t2 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t1.`CHECK_TIME`>='2021-08-06 00:00:00' AND t1.`CHECK_TIME`<='2021-09-06 00:00:00';"

    # 获取全国从2021年9月到11月的数据
    sql22 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS`,t2.`TYPE_ID` FROM `transport` t1,`transport_check_loaded_freight` t2 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t1.`CHECK_TIME`>'2021-12-09 00:00:00' AND t1.`CHECK_TIME`<='2021-12-26 00:00:00';"

    sql_22 = "select * from tollgrantry"
    # 获取MySQL数据库数据
    # 获取pub_organ下所有收费站的数据
    db = OperationMysql('60.205.149.33', 18607, 'cx', 'Lvtongcx2019', 'freeway_appointment')
    db.get_data_in_for('20221201', '20230101', 'E:/蓝德智慧交通/LvTong/数据集/06.全绿通数据/绿通原始数据')
    # 获取全部的单次画像和长期画像
    # db = OperationMysql('192.168.0.182', )
    # sql16 = "SELECT * FROM `pub_organ` WHERE TYPE_ID='tollgate'"
    # data = db.search_one(sql21)
    # data = pd.DataFrame(data)
    # data.to_csv('../../数据集/06.全绿通数据/原始数据/全水果数据11-11.csv')
    # sql8 = "SELECT t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_RECEIVE_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t3.`NAME`,d1.`NAME`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS` FROM `transport` t1,`transport_check_loaded_freight` t2,`tollgate` t3,`district` d1 WHERE t1.`ID`=t2.`TRANSPORT_ID` AND t2.`TYPE_ID`='20101' AND t1.`ENTRANCE_TOLLGATE_ID`=t3.`ID` AND t3.`DISTRICT_ID`=d1.`ID` AND t1.`CHECK_TIME`>='2020-05-06 00:00:00' AND t1.`CHECK_TIME`<='2021-05-10 00:00:00';"
    # data = db.search_one(sql8)
    # data = pd.DataFrame(data)
    # data.to_csv('全国数据2.csv')t1.`APPOINTMENT_USER_ID`,t1.`VEHICLE_ID`,t1.`CHECK_TIME`,t1.`VEHICLE_TYPE`,t1.`CRATE_TYPE`,t1.`WEIGHT`,t1.`ENTRANCE_TOLLGATE_ID`,t1.`EXIT_TOLLGATE_ID`,t3.`NAME`,t1.`INSPECTOR_ID`,t1.`REVIEWER_ID`,t1.`PASS_ID`,t1.`CHECK_RESULT`,t1.`MEDIA_TYPE`,t1.`REASON`,t1.`IS_VALID`,t1.`STATUS`

    # 进行数据的数据库写入
    # db.write_one('middle_data', ["PASSID","enVehiclePlate","enIdenVehiclePlate","enGantryID","entryStationID","entryTime","enMediaType","enVehicleType","enVehicleClass","entryWeight","enOBUVehiclePlate","enCPUVehiclePlate","exVehiclePlate","exIdenVehiclePlate","exGantryID","exitStationID","exitTime","exMediaType","exVehicleType","exVehicleClass","exOBUVehiclePlate","exCPUVehiclePlate","entryTimeFromExit","payFee","vehiclePlate","gantryPath","intervalPath","gantryNum","gantryTimePath","gantryTypePath","gantryTotalFee","gantryTotalLength","firstGantryTime","endGantryTime","processVehicleType","ifTypeChange","processVehicleClass","ifClassChange","enTotalVehiclePlate","exTotalVehiclePlate"],
    #              [["1.050113091703E+34","YZ13800","","G030N61001002510010","G030N610010010","2021/11/1 15:30:11","1","1","8","","YZ13800_3","YZ13800_0","YZ13800","","G003061002000410040","G0030610020030","2021/11/1 15:50:15","1","1","8","YZ13800_3","YZ13800_0","2021/11/1 15:30:11","0","YZ13800","G030N61001002510|G003061002000410","G030N61001002510|G003061002000410","2","2021-11-01 15:30:11|2021-11-01 15:50:15","1|1","0","2200","2021/11/1 15:30:11","2021/11/1 15:50:15","","","","","YZ13800_3","YZ13800_3"]],
    #              ['varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'datetime', 'float', 'float', 'float', 'float', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'varchar', 'datetime', 'float', 'float', 'float', 'varchar', 'varchar', 'datetime', 'float', 'varchar', 'varchar', 'varchar', 'int', 'varchar', 'varchar', 'float', 'int', 'datetime', 'datetime', 'float', 'float', 'float', 'float', 'varchar', 'varchar'])
    # db.write_one('middle_data',
    #              ["PASSID","enVehiclePlate"],
    #              [["1.050113091702E+34","YZ13800"]], )

    # 获取Oracle数据库数据
    sql_0 = "select t1.*,t2.name from dbbase2020.grantrysectdata20200614 t1 left join dbbase2020.tollgrantry t2 on " \
            "t1.gantryid =t2.id where passid='020000510101650026592120200614111231' and traderesult = 0"
    sql_1 = "select * from changda.tollinterval"
    sql_2 = "select * from dbbase2020.grantrysectdata20200614 WHERE hourbatchno>'2020061422'"
    sql_3 = "select ENID, EXID, VTYPE, FEE, M, SGROUP from CHANGDA.SHORTESTPATHFD"
    sql_4 = "select g.BIZID, g.GANTRYID,g.BACKUPSIGN, g.ENTRYSTATION,g.EXITSTATION,g.POINTTYPE,g.GANTRYORDERNUM,g.GANTRYTYPE,g.TRANSTIME,g.FEE,g.MEDIATYPE,g.VEHICLEPLATE,g.IDENTIFYVEHICLEID,g.VEHICLETYPE,g.VEHICLECLASS,g.ENTOLLSTATIONNAME,g.ENTIME,g.PASSID,g.LASTGANTRYTIME,g.OBUVEHICLEPLATE, g.OBUVEHICLETYPE, g.ENWEIGHT, g.CPUVEHICLEPLATE, g.CPUVEHICLETYPE, g.TRADERESULT, g.FEEVEHICLETYPE, g.VALIDFLAG, g.isFixData, t.STATIONFX from dbbase2020.grantrysectdata20200614 g left join changda.tollgrantry t on g.GANTRYID=t.ID WHERE hourbatchno>='2020061412' and hourbatchno<='2020062012' and g.TRADERESULT=0"
    # 牌识数据获取
    sql_5 = "select g.GANTRYID,g.POINTTYPE, g.PICTIME,g.VEHICLEPLATE,g.VEHICLEMODEL,g.VEHICLECOLOR,g.VALIDFLAG,g.TRANSFERMARK from dbbase2020.identifysectdata20200614 g WHERE g.HOURBATCHNO>'2020061400' and HOURBATCHNO<'2020061412'"
    sql_6 = "select g.exitstation,g.extimestr, g.exvehicleid, g.exidentifyvehicleid, g.passid,g.mediatype,g.validflag,g.transpaytype,g.feemileage,g.exitfeetype from dbbase2020.laneexit20200614 g WHERE g.hourbatchno>'2020061422'"
    sql_7 = "select g.entrystation,g.entimestr, g.passid, g.mediatype, g.envehicleid,g.enidentifyvehicleid from dbbase2020.laneentry20200614 g WHERE g.hourbatchno>'2020061422'"
    sql_8 = "select g.GANTRYID,g.POINTTYPE, g.PICTIME,g.VEHICLEPLATE,g.VEHICLEMODEL,g.VEHICLECOLOR,g.VALIDFLAG,g.TRANSFERMARK from dbbase2020.identifysectdata20200617 g"
    sql_9 = "select g.GANTRYID,g.POINTTYPE, g.PICTIME,g.VEHICLEPLATE,g.VEHICLEMODEL,g.VEHICLECOLOR,g.VALIDFLAG,g.TRANSFERMARK from dbbase2020.identifysectdata20200618 g"
    sql_10 = "select g.GANTRYID,g.POINTTYPE, g.PICTIME,g.VEHICLEPLATE,g.VEHICLEMODEL,g.VALIDFLAG,g.DRIVEDIR from dbbase2020.identifysectdata20200614 g"
    sql_11 = "select g.GANTRYID,g.POINTTYPE, g.PICTIME,g.VEHICLEPLATE,g.VEHICLEMODEL,g.VEHICLECOLOR,g.VALIDFLAG,g.TRANSFERMARK from dbbase2020.identifysectdata20200620 g"
    sql_12 = "SELECT laneid, extimestr, passid, exvehicleid, fee, pathmileage FROM dbbase2020.LANEEXIT20200620 where mod(recordtype ,2) = 1 and paytype<>110 and paykind<> 110 and dealstatus <>256"

    # db = OperationOracle()
    # data1 = db.search_one(sql_10)
    # data1 = pd.DataFrame(data1)
    # data1.to_csv('../../数据集/09.陕西全路网数据/data_iden_2020061400_24.csv')
    # db = OperationOracle()
    # data2 = db.search_one(sql_10)
    # data2 = pd.DataFrame(data2)
    # data2.to_csv('../../数据集/09.陕西全路网数据/data_iden_2020061400_24.csv')
    # db = OperationOracle()
    # data3 = db.search_one(sql_11)
    # data3 = pd.DataFrame(data3)
    # data3.to_csv('../../数据集/09.陕西全路网数据/data_iden_2020062000_24.csv')
