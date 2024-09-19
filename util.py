#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/8/18 20:54
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : util.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version  : V1.4 - ZL.Z：2022/5/19
# 		      删除vup_duration_from_vuvInfo与vup_duration_from_vuvTextGrid的p_dur_thr参数，
# 		      并修改方法为duration_from_vuvInfo与duration_from_vuvTextGrid，仅返回有声段和无声段（包含了轻音）
#             V1.3 - ZL.Z：2022/3/30
# 		      添加音频分割audio_split、合并audio_join、汉字数字转换成阿拉伯数字cn2an方法
#             V1.2 - ZL.Z：2022/3/17
# 		      添加audio_word_align方法
#             V1.1 - ZL.Z：2022/3/2
# 		      添加delete_punctuation方法
#             V1.0 - ZL.Z：2021/8/18
# 		      First version.
# @License  : None
# @Brief    : 通用方法集合

import os
import csv
import math
import shutil
import parselmouth
from parselmouth.praat import call

PUNCTUATIONS = set(u''' :!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')  # 标点符号合集（含空格）


def delete_punctuation(text_list):
    """
    将文本或列表中的标点符号删除
    :param text_list: 文本或列表
    :return: 删除后的结果
    """
    if type(text_list) is list:
        filter_punt = lambda l: list(filter(lambda x: x not in PUNCTUATIONS, l))  # 对list
    else:
        filter_punt = lambda s: ''.join(filter(lambda x: x not in PUNCTUATIONS, s))  # 对str/unicode
    return filter_punt(text_list)


def write_csv(data, filename):
    """写入csv文件"""
    if not filename.endswith('.csv'):  # 补上后缀名
        filename += '.csv'
    # 中文需要设置成utf-8格式,为了防止excel打开中文乱码，使用utf-8-sig编码，会在文本前插入签名\ufeff，
    # 表示BOM(Byte Order Mark)，是一个不显示的标识字段，表示字节流高位在前还是低位在前
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:  # 以追加模式、无新行、utf-8编码打开文件
        f_csv = csv.writer(f)  # 先写入表头
        # for item in data:
        #     f_csv.writerow(item)  # 一行一行写入
        f_csv.writerows(data)  # 写入多行


def read_csv(filename):
    """读取csv文件"通过csv.reader()来打开csv文件，返回的是一个列表格式的迭代器，
    可以通过next()方法获取其中的元素，也可以使用for循环依次取出所有元素。"""
    if not filename.endswith(".csv"):  # 补上后缀名
        filename += ".csv"
    data = []  # 所读到的文件数据
    with open(filename, "r", encoding="utf-8-sig") as f:  # 以读模式、utf-8编码打开文件
        f_csv = csv.reader(f)  # f_csv对象，是一个列表的格式
        for row in f_csv:
            data.append(row)
    return data


def duration_from_vuvInfo(vuv_info: str):
    """
    从praat的包含‘U’/'V'的TextGrid对象中List得到的Info文本信息分割语音：有声段voice segments、无声段unvoice segments
    这里的有声端可作为浊音段，无声段可以作为停顿段(静音段pause segments)，但这里的无声段也包含了轻音
    :param vuv_info: 包含‘U’/'V'字段的TextGrid对象中List得到的Info文本信息
    :return: segments_voice, segments_unvoice
            float, list(n_segments, 2),对应每一段的起始和结束时间，单位为s
    """
    segments_voice, segments_unvoice = [], []
    for text_line in vuv_info.strip('\n').split('\n'):
        text_line = text_line.split('\t')
        if 'text' not in text_line:
            tmin = float(text_line[0])
            text = text_line[1]
            tmax = float(text_line[2])
            if text == 'V':  # voice段
                segments_voice.append([tmin, tmax])
            elif text == 'U':  # unvoice段,其中包含轻音和静音，要想分割，需进一步分离
                segments_unvoice.append([tmin, tmax])
    return segments_voice, segments_unvoice


def duration_from_vuvTextGrid(vuv_file):
    """
    从praat得到的vuv.TextGrid文件中分割语音：有声段voice segments、无声段unvoice segments
    这里的有声端可作为浊音段，无声段可以作为停顿段(静音段pause segments)，但这里的无声段也包含了轻音
    :param vuv_file: 从praat得到的vuv.TextGrid文件
    :return: segments_voice, segments_unvoice
            float, list(n_segments, 2),对应每一段的起始和结束时间，单位为s
    """
    with open(vuv_file) as f:
        segments_voice, segments_unvoice = [], []
        data_list = f.readlines()
        for data_index in range(len(data_list)):
            data_list[data_index] = data_list[data_index].strip()  # 去掉换行符
            if data_list[data_index] == '"V"':  # voice段
                # 标识字符前两行为起始的duration
                segments_voice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
            elif data_list[data_index] == '"U"':  # unvoice段,其中包含轻音和静音，要想分割，需进一步分离
                segments_unvoice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
        return segments_voice, segments_unvoice


def audio_join(input_dir, output_dir, joint_silence_len=1.5, samp_freq=16000):
    """
    将声音文件合并
    :param input_dir: 输入包含待合并的音频文件夹
    :param output_dir: 音频输出文件夹
    :param joint_silence_len: 合并音频间隔，默认1.5秒
    :param samp_freq: 音频采样率，这里合并音频需要统一采样率，默认16kHz
    :return: None
    """
    assert len(os.listdir(input_dir)), f'输入文件夹{input_dir}中不存在音频文件'
    if len(os.listdir(input_dir)) == 1:
        shutil.copy(os.path.join(input_dir, os.listdir(input_dir)[0]), output_dir)
    aud_sil = call('Create Sound from formula', 'sil', 1, 0.0, joint_silence_len, samp_freq, '0')
    auds = []
    for _i in os.listdir(input_dir):
        i_aud = os.path.join(input_dir, _i)
        auds.append(parselmouth.Sound(i_aud))
        auds.append(aud_sil)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 先删除输出文件夹
    os.mkdir(output_dir)  # 重新创建
    sound = parselmouth.Sound.concatenate(auds)
    save_name = os.path.basename(os.listdir(input_dir)[0]).split('-')[0] + '.wav'
    sound.save(os.path.join(output_dir, save_name), parselmouth.SoundFileFormat.WAV)


# 系数
CN_NUM = {
    u'〇': 0,
    u'一': 1,
    u'二': 2,
    u'三': 3,
    u'四': 4,
    u'五': 5,
    u'六': 6,
    u'七': 7,
    u'八': 8,
    u'九': 9,

    u'零': 0,
    u'壹': 1,
    u'贰': 2,
    u'叁': 3,
    u'肆': 4,
    u'伍': 5,
    u'陆': 6,
    u'柒': 7,
    u'捌': 8,
    u'玖': 9,

    u'貮': 2,
    u'两': 2,
    u'俩': 2,
    u'倆': 2,
    u'营': 0,
    u'其': 7,
    u'西': 7,
    u'气': 7,
    u'吧': 8,
    u'就': 9,
}
# 基数
CN_UNIT = {
    u'十': 10,
    u'拾': 10,
    u'是': 10,
    u'实': 10,
    u'时': 10,
    u'百': 100,
    u'佰': 100,
    # u'千': 1000,
    # u'仟': 1000,
    # u'万': 10000,
    # u'萬': 10000,
    # u'亿': 100000000,
    # u'億': 100000000,
    # u'兆': 1000000000000,
}


def cn2an(chinese_number):
    """
    汉字数字转换成阿拉伯数字
    :param chinese_number: 汉字数字
    :return: 阿拉伯数字，字符串型
    """
    # 根据系数、基数map将汉字转对应阿拉伯数字
    tmp = []
    for d in chinese_number[::-1]:  # 遍历每个chinese_number中元素的倒序
        if CN_NUM.__contains__(d):
            tmp.append(CN_NUM[d])  # 系数对应阿拉伯数字
        elif CN_UNIT.__contains__(d):
            tmp.append(CN_UNIT[d])  # 基数对应阿拉伯数字
        elif d.isnumeric():  # 若元素中含有阿拉伯数字字符
            try:
                tmp.append(int(d))  # 原样返回阿拉伯数字
            except ValueError:
                pass
    if not tmp:  # 若元素全为非数字字符，此时tmp列表为空退出
        return -1
    # 系数直接加入tmp2，基数相邻相乘或者加入tmp2并在前面补1（解决“十一”， “十”这种基数前没系数的情况）
    tmp_len = len(tmp)
    tmp2 = []
    for i in range(0, tmp_len):
        if tmp[i] > 9:  # 系数大于9
            if i == tmp_len - 1 or tmp[i + 1] > tmp[i]:  # “十一”， “十”这种基数前没系数的情况
                tmp2.append(tmp[i])
                tmp2.append(1)
            elif tmp[i + 1] > 9:  # 两个基数情况直接相邻相乘
                tmp[i + 1] *= tmp[i]
            else:  # 只有一个基数情况直接加入tmp2
                tmp2.append(tmp[i])
        else:  # 系数小于10，直接加入tmp2
            tmp2.append(tmp[i])
    # 系数直接加入seq，基数根据其大小用-1占住位置让下一个系数在正确位上
    seq = []
    curW = 0
    for t in tmp2:
        if t > 9:
            w = math.log10(t)  # 最终数值长度-1
            while curW < w:
                curW += 1
                seq.append(-1)
        else:
            curW += 1
            seq.append(t)
    # 对于个位是非0的前方是-1的数，要尽可能提到高位
    if seq[0] > 0 and len(seq) > 1 and seq[1] == -1:
        seqLen, p = len(seq), 1
        while p < seqLen and seq[p] == -1:
            p += 1
        # 交换
        seq[p - 1] = seq[0]
        seq[0] = 0
    # seq拼接，-1转为0，其余的保持原数值
    return "".join([str(n if n >= 0 else 0) for n in seq[::-1]])


