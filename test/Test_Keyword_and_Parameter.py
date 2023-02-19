"""
    -*- coding: UTF-8 -*-
    Copyright (c) 2023, Sun HaoYu. All rights reserved.
    function: 进行关键参数库的测试
"""

import pytest
import key_management.Keyword_and_Parameter as kp

'''
    创建时间：2023/2/16
    内容：进行基础函数库的测试
'''


# 测试Parameter_Management类
# Multi Options
# TODO
def test_Parameter_Management_initial():
    model = kp.run_function_with_input(type, '123456')
    assert model


# sad Path
# TODO
# def test_run_function_with_input_sum():
#     model = bf.run_function_with_input(sum, '123456')
#     assert not model


# TODO
# def test_run_function_with_input_None():
#     model = bf.run_function_with_input(None, '123456')
#     assert not model

# Default Value

# 测试charge_inputs_relation_with_sign函数
# Multi Options
# TODO input_value1:‘11111’ input_value2:‘11111’ compare_sign:‘=’
def test_charge_inputs_relation_with_sign_input_same_data():
    model = bf.charge_inputs_relation_with_sign('11111', '11111', '=')
    assert model


def test_charge_inputs_relation_with_sign_input_not_same_data():
    model = bf.charge_inputs_relation_with_sign('11111', '', '=')
    assert not model


# TODO input_value1:30 input_value2:10 compare_sign:‘>’
def test_charge_inputs_relation_with_sign_input_bigger_data():
    model = bf.charge_inputs_relation_with_sign(30, 10, '>')
    assert model


# TODO input_value1:30 input_value2:[10, 30] compare_sign:‘in’
def test_charge_inputs_relation_with_sign_in():
    model = bf.charge_inputs_relation_with_sign(30, [10, 30], 'in')
    assert model


def test_charge_inputs_relation_with_sign_not_in():
    model = bf.charge_inputs_relation_with_sign(30, [10, 20], 'in')
    assert not model

# sad Path
# TODO compare_sign:'=' and input_value1 or input_value2:None

# TODO fun:None and value:'123456


# 测试create_random_string函数
# TODO Multi input
def test_create_random_string():
    result = bf.create_random_string(3, 'num')
    assert len(result)


# 测试get_point_distance_target_point函数
# TODO Multi input
def test_get_point_distance_target_point():
    result = bf.get_point_distance_target_point([[0, 0], [1, 1], [1, 2], [2, 2]], 3, 2)
    assert result


# 测试compute_rate_of_two_parameters函数
# TODO Multi input
def test_compute_rate_of_two_parameters():
    result = bf.compute_rate_of_two_parameters(1, 10)
    assert result


# 测试compute_release_of_tools函数
# TODO Multi input, health enough
def test_compute_release_of_enough_health():
    result = bf.compute_release_of_tools([10, 12], 20, 1)
    assert result


# TODO Multi input, health enough
def test_compute_release_of_low_health():
    result = bf.compute_release_of_tools([10, 12], 10, 1)
    assert result


# 测试get_number_from_string函数
# TODO Single input string have number
def test_get_number_from_string():
    result = bf.get_number_from_string('K3.495')
    assert result


# TODO Single input string all number
def test_get_number_from_string_all_num():
    result = bf.get_number_from_string('3.495')
    assert result


# 测试compute_road_distance_of_points函数
# TODO Multi input with two string have number
def test_compute_road_distance_of_points():
    result = bf.compute_road_distance_of_points('K3.495', 'K4.99')
    assert result


if __name__ == '__main__':
    test_compute_road_distance_of_points()