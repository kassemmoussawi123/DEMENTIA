# -*- coding: utf-8 -*-
# Copyright (c) 2024. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2023/11/10 09:47
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : dataset.py
# @Software : Python 3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.1.0 - ZL.Z：2024/9/10
# @License  : None
# @Brief    : 数据准备、特征提取

from config import *
from util import *
import regex as re
import os
import glob
import pandas as pd
from wordcloud import WordCloud
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
import librosa
from pathos.pools import ProcessPool as Pool
import parselmouth
from parselmouth.praat import call
from speechpy.processing import cmvn
from transformers import TFBertModel, BertTokenizer, TFRobertaModel, RobertaTokenizer, \
    TFDistilBertModel, DistilBertTokenizer, TFAlbertModel, AlbertTokenizer
from typing import Union, List, Tuple
import hanlp
import nltk
from transformers import logging
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
import ptitprince as pt
from statannotations.Annotator import Annotator
from collections import OrderedDict

logging.set_verbosity_error()
nltk.data.path.append(NLTK_DATA_PATH)
# HanLP的Native API输入单位为句子，需使用多语种分句模型或基于规则的分句函数先行分句
# devices=-1仅用CPU，以避免多线程调用GPU报错（RuntimeError: Cannot re-initialize CUDA in forked subprocess.
# To use CUDA with multiprocessing, you must use the 'spawn' start method）
DEVICE = -1  # 若使用全部GPU则DEVICE = None，此时不能并行提取该任务特征；设置devices=-1仅用CPU，则可以并行（这里python原生Hanlp无法并行，程序会卡住不动）
HanLP = hanlp.pipeline().append(hanlp.utils.rules.split_sentence, output_key='sentences')\
    .append(hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE,
                       devices=DEVICE), output_key='xlm')


def extract_data_from_cha(input_f_cha: Union[str, List[str]], remove_marker: bool = False) -> pd.DataFrame:
    """
    从.cha文件中提取文本数据
    :param input_f_cha: .cha文件路径或路径列表
    :param remove_marker: 是否删除cha文件中的标记，如停顿、重复、语气等CLAN标记
    :return: pd.DataFrame(par_data_list):speech/clean_speech/clean_par_speech/joined_all_speech/joined_par_speech/
            per_sent_times/total_time/time_before_par_speech/time_between_sents
    """
    if type(input_f_cha) is str:
        input_f_cha = [input_f_cha]
    par_data_list = []
    for cha_file in input_f_cha:
        par = {'id': os.path.basename(cha_file)[:-4]}
        f = iter(open(cha_file, encoding='UTF-8'))
        speech = []
        try:  # 获取PAR和INV文本
            curr_speech = ''
            while True:
                line = next(f)
                if line.startswith('@ID'):
                    participant = [i.strip() for i in line.split('|')]
                    if participant[2] == 'PAR':
                        par['sex'] = ['f', 'm'].index(participant[4][0])
                        par['age'] = int(participant[3].replace(';', ''))
                        par['label'] = 0 if participant[5] == 'Control' else 1 if participant[5] != '_X_' else np.NAN
                        par['mmse'] = np.NAN if (len(participant[8]) == 0 or participant[8] == '_Y_') else float(participant[8])
                        par['set'] = 'train' if participant[5] != '_X_' else 'test'
                if line.startswith('*PAR:') or line.startswith('*INV'):
                    curr_speech = line
                elif len(curr_speech) != 0 and not (line.startswith('%') or line.startswith('*')):
                    curr_speech += line
                elif len(curr_speech) > 0:
                    speech.append(curr_speech)
                    curr_speech = ''
        except StopIteration:
            pass
        clean_par_speech = []
        clean_all_speech = []
        par_speech_time_segments = []
        all_speech_time_segments = []
        is_par = False
        for _s in speech:
            def _parse_time(_s):  # 查找时间点
                return [*map(int, re.search('\x15(\d*_\d*)\x15', _s).groups()[0].split('_'))]

            def _clean(_s):  # 删除无关信息
                _s = re.sub('\x15\d*_\d*\x15', '', _s)  # remove time block
                _s = re.sub('\[.*\]', '', _s)  # remove other speech artifacts [.*]
                _s = _s.strip()
                # remove tab, new lines, inferred speech??, ampersand, &, etc
                _s = re.sub('\n\t', ' ', _s)
                _s = re.sub('\t|\n', '', _s)
                if remove_marker:  # remove inferred speech??, ampersand, &, etc marker
                    _s = re.sub('<|>|\[/\]|\[//\]|\[///\]|&=clears throat|=sings|=laughs|=clears:throat|=sighs|=hums|'
                                '=chuckles|=grunt|=finger:tap|=claps|=snif|=coughs|=tapping|\(\.\)|\(\.\.\)|'
                                '\(\.\.\.\)|/|xxx|\+|\(|\)|&', '', _s)
                return _s

            if _s.startswith('*PAR:'):
                is_par = True
            elif _s.startswith('*INV:'):
                is_par = False
                _s = re.sub('\*INV:\t', '', _s)  # remove prefix
            if is_par:
                _s = re.sub('\*PAR:\t', '', _s)  # remove prefix
                par_speech_time_segments.append(_parse_time(_s))
                clean_par_speech.append(_clean(_s))
            all_speech_time_segments.append(_parse_time(_s))
            clean_all_speech.append(_clean(_s))

        par['speech'] = speech  # 原始转录文本
        par['clean_speech'] = clean_all_speech  # 删除无关信息的转录文本，包括PAR和INV，列表，每句话为一个元素
        par['clean_par_speech'] = clean_par_speech  # 删除无关信息的转录文本，仅包括PAR，列表，每句话为一个元素
        par['joined_all_speech'] = ' '.join(clean_all_speech)  # 整合删除无关信息的转录文本，包括PAR和INV
        par['joined_par_speech'] = ' '.join(clean_par_speech)  # 整合删除无关信息的转录文本，仅包括PAR

        # sentence times
        par['per_sent_times'] = [par_speech_time_segments[i][1] - par_speech_time_segments[i][0] for i in
                                 range(len(par_speech_time_segments))]  # 被试PAR每句话的时长
        par['total_time'] = par_speech_time_segments[-1][1] - par_speech_time_segments[0][0]  # PAR全部时间总和
        par['time_before_par_speech'] = par_speech_time_segments[0][0]  # PAR开始叙述的时间点
        par['time_between_sents'] = [
            0 if i == 0 else max(0, par_speech_time_segments[i][0] - par_speech_time_segments[i - 1][1])
            for i in range(len(par_speech_time_segments))]  # PAR每句话的间隔时间，第一句话为距INV结束的时间间隔
        par_data_list.append(par)
    return pd.DataFrame(par_data_list)


class HandcraftedFeatures:
    """获取自发言语任务的手工特征"""
    def __init__(self, input_f_audio: Union[str, parselmouth.Sound], input_f_trans: str, f0min: int = 75,
                 f0max: int = 600, sil_thr: float = -25.0, min_sil: float = 0.1, min_snd: float = 0.1):
        """
        初始化
        :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans: 输入文本转录文件，cha类似的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        :param sil_thr: 相对于音频最大强度的最大静音强度值(dB)。如imax是最大强度，则最大静音强度计算为sil_db=imax-|sil_thr|
                        强度小于sil_db的间隔被认为是静音段。sil_thr越大，则语音段越可能被识别为静音段，这里默认为-25dB
        :param min_sil: 被认为是静音段的最小持续时间(s)。
                        默认0.1s，即该值越大，则语音段越可能被识别为有声段（若想把爆破音归为有声段，则将该值设置较大）
        :param min_snd: 被认为是有声段的最小持续时间(s)，即不被视为静音段的最小持续时间。
                        默认0.1s，低于该值被认为是静音段，即该值越大，则语音段越可能被识别为静音段
        """
        self.f_audio = input_f_audio
        self.f_trans = input_f_trans
        self.f0min = f0min
        self.f0max = f0max
        self.sound = parselmouth.Sound(self.f_audio)
        self.total_duration = self.sound.get_total_duration()
        self.text_grid_vuv = call(self.sound, "To TextGrid (silences)", 100, 0.0, sil_thr, min_sil, min_snd, 'U', 'V')
        self.vuv_info = call(self.text_grid_vuv, "List", False, 10, False, False)
        self.par_data_all = extract_data_from_cha(self.f_trans, remove_marker=False)
        par_data = extract_data_from_cha(self.f_trans, remove_marker=True)
        self.text = par_data.loc[0, 'joined_par_speech']
        self.text_seg_list = nltk.word_tokenize(self.text)  # 分词结果列表
        self.text_seg_list_no_punct = delete_punctuation(self.text_seg_list)  # 删除标点符号后的分词结果
        self.text_posseg_list = nltk.pos_tag(self.text_seg_list, tagset='universal')  # 分词结果列表，含词性(包含所有词性)
        self.sent_num = len(par_data.loc[0, 'clean_par_speech'])
        self.doc = HanLP(self.text)['xlm']  # 基于多任务学习模型的全部结果，Document类型，该结果用于获取除了语篇的其他特征

    def func_phrase_rate(self, tree_tag: str):
        """
        计算特定类型短语比例：特定类型短语数量/句子数量，其中特定类型短语设定为至少包含一个特定类型及其附属词的特定类型短语，
        且为最大长度，即不包含特定类型短语中的特定类型短语
        :param tree_tag: 基于Penn Treebank的短语类型标签，参见https://hanlp.hankcs.com/docs/annotations/constituency/ptb.html
        :return: npr，特定类型短语率
        """
        con = self.doc['con']  # 该语篇包含多条句子,为list类型，其中元素为phrasetree.tree.Tree短语结构树类型
        np_l = []
        for child in con:  # 对于每一个子树
            last_str = ''
            for subtree in child.subtrees(lambda t: t.label() == tree_tag):  # 查找标签为tree_tag的特定类型短语
                # 至少包含一个特定类型及其附属词的特定类型短语，且不包含特定类型短语中的特定类型短语
                if len(subtree.leaves()) > 1 and ''.join(subtree.leaves()) not in last_str:
                    np_l.append(subtree.leaves())
                last_str = ''.join(subtree.leaves())
        return len(np_l) / self.sentence_num()

    def func_calc_yngve_score(self, tree, parent):
        """
        递归计算句法复杂度指标：Yngve评分
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild Cognitive
        Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        https://github.com/meyersbs/SPLAT/blob/fd211e49582c64617d509db5746b99075a25ad9b/splat/complexity/__init__.py#L279
        https://github.com/neubig/util-scripts/blob/96c91e43b650136bb88bbb087edb1d31b65d389f/syntactic-complexity.py
        :param tree: phrasetree.tree.Tree类型短语结构树
        :param parent: 父节点评分，初始调用为0
        :return: Yngve评分
        """
        if type(tree) == str:
            return parent
        else:
            count = 0
            for i, child in enumerate(reversed(tree)):
                count += self.func_calc_yngve_score(child, parent + i)
            return count

    def func_get_yngve_list(self):
        """
        获取整个文本所有子树的yngve评分列表，其中每个子树的yngve=全部叶子的yngve和/叶子数，即在叶子尺度上的平均yngve得分
        :return: 整个文本所有子树的yngve评分列表
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        yngve_l = []
        for child in con:  # 对于每一个子树
            if child.label() != 'PU':  # 排除仅包含一个标点符号的子树
                yngve_l.append(self.func_calc_yngve_score(child, 0) / len(child.leaves()))  # 计算该结构树的yngve得分
        return yngve_l

    def f0_std(self):
        """
        计算基频的标准偏差
        :return: f0_sd, float, semitones
        """
        pitch_obj = call(self.sound, "To Pitch", 0.0, self.f0min, self.f0max)
        f0_sd = call(pitch_obj, "Get standard deviation", 0.0, 0.0, "semitones")
        return f0_sd

    def duration_pause_intervals(self):
        """
        计算停顿间隔时间的中位值
        :return: dpi, float, ms
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        duration_p = np.array(segments_u)[:, 1] - np.array(segments_u)[:, 0]
        dpi = float(1000 * np.median(duration_p))
        return dpi

    def voiced_rate(self):
        """
        计算语音速率：每秒出现的浊音段数量
        :return: rate, float, 1/s
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        rate = len(segments_v) / self.total_duration
        return rate

    def hesitation_ratio(self):
        """
        计算犹豫率：犹豫的总持续时间除以总演讲时间，其中犹豫被定义为持续超过30毫秒的没有说话
        ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5337522/
        :return: hesi_ratio, float
        """
        segments_v, segments_u = duration_from_vuvInfo(self.vuv_info)
        try:
            duration_p = np.array(segments_u)[:, 1] - np.array(segments_u)[:, 0]
            hesi_ratio = np.sum(duration_p[duration_p > 0.03]) / self.total_duration
            return hesi_ratio
        except IndexError:
            if not len(segments_u):
                return 0.0
            elif not len(segments_v):
                return 1.0

    def empty_word_frequency(self):
        """
        计算“空动词”词频：即oh/=laughs/down/well/some/what/fall，以及xxx（表示胡言乱语或杂乱语）词语的总数
        ref: Yuan J, Bian Y, Cai X, et al. Disfluencies and fine-tuning pre-trained language models for
             detection of alzheimer’s disease [C]. Interspeech 2020. 2020: 2162-2166.
        :return: empty_word_freq, int
        """
        word_l = self.par_data_all.loc[0, 'joined_par_speech'].split(' ')
        empty_word = ['oh', 'uh', '&uh', '=laughs', '&=laughs', 'down', 'well', 'some', 'what', 'fall', 'xxx',
                      'she', 'he', 'him', 'hm', 'it']
        empty_word_freq = 0
        freq_dist = nltk.FreqDist(word_l)
        for wd, freq in freq_dist.items():
            if wd in empty_word:
                empty_word_freq += freq
        return empty_word_freq

    def word_rate(self):
        """
        计算该任务被试话语的速率：每秒词语数（不包括标点）
        :return: 每秒词语数，单位词/s
        """
        return len(self.text_seg_list_no_punct) / self.total_duration

    def function_word_ratio(self):
        """
        计算该任务被试话语中虚词(包括副词/介词/连词/代词/限定词/基数，不包括非语素词和标点)与所有词（不包括标点）的比值
        :return: 虚词与所有词（不包括标点）的比值fw_ratio, float
        """
        # NLTK universal的虚词标注
        func_w_l = ['ADV', 'ADP', 'CONJ', 'PRON', 'DET', 'NUM']
        func_w = [(x[0], x[-1]) for x in self.text_posseg_list if x[-1] in func_w_l]
        fw_ratio = len(func_w) / len(self.text_seg_list_no_punct)
        return fw_ratio

    def lexical_density(self):
        """
        计算该任务被试话语的词汇密度：词汇词(即实义词，包括实义动词/名词/形容词)与总词汇数（不包括标点）比率
        :return: 词汇密度ld, float
        """
        lw_l = ['VERB', 'NOUN', 'ADJ']  # NLTK universal的动词/名词/形容词标注
        # 排除一些常见的无实义动词/名词
        ex_verb = ['is', 'am', 'are', 'was', 'were', "'s", "'m", "'re", 'can', 'cannot', 'could', 'couldn', "'t",
                   "'d", 'uh', 'um', 'mhm', 'oh']
        lw = [(x[0], x[-1]) for x in self.text_posseg_list if (x[-1] in lw_l) and (x[0].lower() not in ex_verb)]
        ld = len(lw) / len(self.text_seg_list_no_punct)
        return ld

    def mean_len_utter(self):
        """
        计算平均话语长度（ Mean Length of Utterance，MLU）：每句话中词语的数量，即词数（不包括标点）与句子数之比
        :return: 词数与句子数之比，float
        """
        return len(self.text_seg_list_no_punct) / self.sentence_num()

    def noun_phrase_rate(self):
        """
        计算名词短语比例：名词短语数量/句子数量，其中名词短语设定为至少包含一个名词及其附属词的名词短语，且为最大长度，即不包含名词短语中的名词短语
        :return: npr，名词短语率
        """
        return self.func_phrase_rate('NP')

    def verb_phrase_rate(self):
        """
        计算动词短语比例：动词短语数量/句子数量，其中动词短语设定为至少包含一个动词及其附属词的动词短语，且为最大长度，即不包含动词短语中的动词短语
        :return: npr，动词短语率
        """
        return self.func_phrase_rate('VP')

    def parse_tree_height(self):
        """
        计算结构树的平均高度：在全部句子中，结构子树高度的平均值（不包括标点符号）
        :return: 结构树的平均高度
        """
        con = self.doc['con']  # 该语篇包含多条句子，con为phrasetree.tree.Tree短语结构树类型
        pth_l = []
        for child in con:  # 对于每一个子树
            if child.label() != 'PU':  # 排除标点符号
                pth_l.append(child.height())
        return np.mean(pth_l)

    def total_yngve_depth(self):
        """
        计算所有子树的Yngve深度总和：在全部句子中，结构子树Yngve深度的总和
        :return: 所有子树的Yngve深度总和
        """
        return sum(self.func_get_yngve_list())

    def total_dependency_distance(self):
        """
        根据依存句法结果，计算在全部句子上平均总的依存距离：每个句子的总依存距离之和/句子数，
        其中依存距离定义为每条依存连接的支配词（中心词）与从属词（修饰词）之间的距离
        ref: B. Roark, M. Mitchell, and K. Hollingshead, "Syntactic complexity measures for detecting Mild
        Cognitive Impairment," presented at the Proceedings of the Workshop on BioNLP 2007, 2007.
        :return: 全部句子上总依存距离的平均值
        """
        dep = self.doc['dep']  # 所有句子的依存句法树列表，第i个二元组表示第i个单词的[中心词的下标, 与中心词的依存关系]
        sen_dep = []
        for i_dep in dep:
            i_dep_total = 0
            for j_dep in i_dep:
                if j_dep[0] != 0:
                    i_dep_total += abs(i_dep.index(j_dep) + 1 - j_dep[0])
            sen_dep.append(i_dep_total)
        return np.mean(sen_dep)

    def get_all_feat(self, prefix=''):
        """
        获取当前所有特征
        :param prefix: pd.DataFrame类型特征列名的前缀
        :return: 该类的全部特征, pd.DataFrame类型
        """
        f0_sd = self.f0_std()
        dpi = self.duration_pause_intervals()
        voiced_rate = self.voiced_rate()
        hesi_r = self.hesitation_ratio()
        em_freq = self.empty_word_frequency()
        wr = self.word_rate()
        fwr = self.function_word_ratio()
        ld = self.lexical_density()
        mlu = self.mean_len_utter()
        n_pr = self.noun_phrase_rate()
        v_pr = self.verb_phrase_rate()
        tree_h = self.parse_tree_height()
        yngve_total = self.total_yngve_depth()
        dep_dist_total = self.total_dependency_distance()
        feat = {prefix+"F0 SD(st)": [f0_sd], prefix+"DPI(ms)": [dpi],
                prefix+"Voiced Rate(1/s)": [voiced_rate], prefix+"Hesitation Ratio": [hesi_r],
                prefix+"Empty Word Freq": [em_freq], prefix+"Word Rate(-/s)": [wr], prefix+"Function Word Ratio": [fwr],
                prefix+"Lexical Density": [ld],
                prefix+"MLU": [mlu], prefix + "Noun Phrase Rate": [n_pr], prefix + "Verb Phrase Rate": [v_pr],
                prefix + "Parse Tree Height": [tree_h], prefix + "Yngve Depth Total": [yngve_total],
                prefix + "Dependency Distance Total": [dep_dist_total], }
        return pd.DataFrame(feat)


def feat_mfcc(input_f_audio: Union[str, parselmouth.Sound]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算39维MFCC系数：13个MFCC特征（第一个系数为能量c0）及其对应的一阶和二阶差分
    :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
    :return: 13*3维MFCC特征及其倒谱均值方差归一化值，每一列为一个MFCC特征向量 np.ndarray[shape=(n_frames, 39), dtype=float32]
    """
    sound = parselmouth.Sound(input_f_audio)
    mfcc_obj = sound.to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=0.01,
                             firstFilterFreqency=100.0, distance_between_filters=100.0)  # 默认额外包含c0
    mfcc_f = mfcc_obj.to_array().T
    mfcc_delta1 = librosa.feature.delta(mfcc_f)  # 一阶差分
    mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)  # 二阶差分
    mfcc = np.hstack((mfcc_f, mfcc_delta1, mfcc_delta2))  # 整合成39维MFCC特征
    # 倒谱均值方差归一化: Cepstral Mean and Variance Normalization,CMVN
    mfcc_f_cmvn = cmvn(mfcc_f, variance_normalization=True)
    mfcc_delta1_cmvn = librosa.feature.delta(mfcc_f_cmvn)  # 一阶差分
    mfcc_delta2_cmvn = librosa.feature.delta(mfcc_f_cmvn, order=2)  # 二阶差分
    mfcc_cmvn = np.hstack((mfcc_f_cmvn, mfcc_delta1_cmvn, mfcc_delta2_cmvn))  # 整合成39维倒谱均值方差归一化MFCC特征
    return mfcc.astype(np.float32), mfcc_cmvn.astype(np.float32)


def feat_bert(input_text: Union[str, List[str]], pretrained_model: str = 'bert-base-uncased') -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    获取基于预训练模型BERT及其变体的文本嵌入
    :param input_text: 输入的文本
    :param pretrained_model: 预训练BERT模型名，包括bert-base-uncased/roberta-base/distilbert-base-uncased/albert-base-v2
    :return: 文本经过预训练模型的token输出np.ndarray[shape=(样本数,序列长度(词数,排除首尾标记,510),最后隐藏层尺寸(句嵌入数768))，float32];
             attention_mask: np.ndarray([样本数,最大token单词数(排除首尾标记,510)], int32)
    """
    if type(input_text) is str:
        input_text = [input_text]
    if pretrained_model == 'bert-base-uncased':
        model_class, tokenizer_class = TFBertModel, BertTokenizer
    elif pretrained_model == 'roberta-base':
        model_class, tokenizer_class = TFRobertaModel, RobertaTokenizer
    elif pretrained_model == 'distilbert-base-uncased':
        model_class, tokenizer_class = TFDistilBertModel, DistilBertTokenizer
    elif pretrained_model == 'albert-base-v2':
        model_class, tokenizer_class = TFAlbertModel, AlbertTokenizer
    else:
        raise ValueError("模型输入错误，请从bert-base-uncased/roberta-base/distilbert-base-uncased/albert-base-v2中选择")
    # 加载预训练BERT模型/tokenizer
    tokenizer = tokenizer_class.from_pretrained(os.path.join(BERT_MODEL_PATH, pretrained_model))
    model = model_class.from_pretrained(os.path.join(BERT_MODEL_PATH, pretrained_model))
    # 对文本进行分词并转换成id序列，token切分，设置将句子转化成对应模型的输入形式，最大读取句子长度为512个词
    # 为批处理，加快速度，将所有列表填充为相同的大小，即最后补零填充至最大长度，此时truncation应设为True，确保长度一致
    # 字典类型，包括input_ids/token_type_ids/attention_mask np.ndarray([样本数,最大token单词数], int32)
    encoded_input = tokenizer(input_text, max_length=512, padding='max_length', truncation=True, return_tensors='tf')
    # 对于句子分类问题，常简单地用第二维中的首列元素，即[CLS]这一token，此时features为二维numpy数组(样本数,最后隐藏层尺寸)，
    # 包含了数据集中所有句子的句嵌入。但这里采用包含时间序列的词级别的token作为下游任务的输入，此时为(样本数,序列长度,最后隐藏层尺寸)(排除首尾标记)
    last_hidden_states = model(encoded_input)[0]
    features = last_hidden_states[:, 1:-1, :].numpy()
    attention_mask = encoded_input['attention_mask'][:, 1:-1].numpy()
    return features, attention_mask


def feat_handcrafted(input_f_audio: Union[str, parselmouth.Sound], input_f_trans: str) -> np.ndarray:
    """
    获取自发言语任务的手工特征
    :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
    :param input_f_trans: 输入文本转录文件，cha类似的文件格式
    :return: 14维手工声学/语言学特征 np.ndarray[shape=(14, ), dtype=float32]
    """
    hf = HandcraftedFeatures(input_f_audio, input_f_trans).get_all_feat()
    np_hf = hf.to_numpy()[0].astype(np.float32)
    return np_hf


class GetFeatures:
    """计算基于自发言语任务的各类特征"""
    def __init__(self, datasets_dir: Union[str, os.PathLike], save_dir: Union[str, os.PathLike],
                 get_text: bool = True, test_info_file: str = ''):
        """
        初始化
        :param datasets_dir: 输入包含的数据集文件的路径
        :param save_dir: 数据保存路径
        :param get_text: 是否获取转录文本
        :param test_info_file: 测试集信息文件
        """
        self.text_f_list = glob.glob(os.path.join(datasets_dir, r'*/transcription/**/*.cha'), recursive=True)
        if get_text:
            par_data = extract_data_from_cha(self.text_f_list, remove_marker=False)
            text_all = pd.read_csv(test_info_file).merge(par_data, how='outer', on='id').reset_index(drop=True)
            # 将相同列的值进行合并，值为 NaN 的将被非 NaN 的值覆盖
            for col in ['sex', 'age', 'label', 'mmse', 'set']:
                text_all[col] = text_all[f'{col}_x'].combine_first(text_all[f'{col}_y'])
                text_all.drop(columns=[f'{col}_x', f'{col}_y'], inplace=True)
            text_all.iloc[:, -5:-2] = text_all.iloc[:, -5:-2].astype(int)
            text_all = text_all[text_all.columns.tolist()[:1] + text_all.columns.tolist()[-5:] +
                                text_all.columns.tolist()[1:-6]]
            text_all.to_csv(os.path.join(save_dir, 'trans.csv'), encoding="utf-8-sig", index=False)
        else:
            text_all = pd.read_csv(os.path.join(save_dir, 'trans.csv'))
        self.text_subinfo = text_all[['id', 'sex', 'age', 'label', 'mmse', 'set',
                                      'clean_par_speech', 'joined_par_speech']]
        self.save_dir = save_dir

    def get_features_noembedding(self, text_file: Union[str, os.PathLike]) -> pd.DataFrame:
        """
        获取对应音频/文本的除文本嵌入的全部特征
        :param text_file: 文本文件
        :return: pd.DataFrame，特征及其对应标签等信息
        """
        print("---------- Processing %d / %d: %s ----------" %
              (self.text_f_list.index(text_file) + 1, len(self.text_f_list), text_file))
        audio_file = text_file.replace('transcription', 'Full_wave_enhanced_audio').replace('.cha', '.wav')
        subid = os.path.basename(text_file)[:-4]
        label = self.text_subinfo[self.text_subinfo['id'] == subid]['label'].values[0]
        sex = self.text_subinfo[self.text_subinfo['id'] == subid]['sex'].values[0]
        age = self.text_subinfo[self.text_subinfo['id'] == subid]['age'].values[0]
        mmse = self.text_subinfo[self.text_subinfo['id'] == subid]['mmse'].values[0]
        setf = self.text_subinfo[self.text_subinfo['id'] == subid]['set'].values[0]
        feats = pd.DataFrame({'id': [subid], 'sex': [sex], 'age': [age], 'label': [label],
                              'mmse': [mmse], 'set': [setf]})
        mfcc_raw, mfcc_cmvn = feat_mfcc(audio_file)
        ft_mfcc = pd.DataFrame({'mfcc_raw': [mfcc_raw], 'mfcc_cmvn': [mfcc_cmvn]})
        ft_hc = pd.DataFrame({'handcrafted': [feat_handcrafted(audio_file, text_file)]})
        feats = pd.concat([feats, ft_mfcc, ft_hc], axis=1)
        return feats

    def get_features(self, n_jobs=None) -> pd.DataFrame:
        """
        并行处理，保存所有特征至本地文件
        :param n_jobs: 并行运行CPU核数，默认为None;若为1非并行，若为-1或None,取os.cpu_count()全部核数,-1/正整数/None类型
        :return: pd.DataFrame，数据的全部特征及其对应标签等信息
        """
        feats_all, feats_emd = pd.DataFrame(), self.text_subinfo[['id']]
        for md in ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "albert-base-v2"]:
            emb_feat, attention_mask = feat_bert(self.text_subinfo['joined_par_speech'].tolist(), md)
            feats_emd = pd.concat([feats_emd, pd.DataFrame({md: emb_feat.tolist(),
                                                            f'mask_{md}': attention_mask.tolist()})], axis=1)
        if n_jobs == -1:
            n_jobs = None
        if n_jobs == 1:
            res = []
            for i_subj in self.text_f_list:
                res.append(self.get_features_noembedding(i_subj))
        else:
            with Pool(n_jobs) as pool:
                res = pool.map(self.get_features_noembedding, self.text_f_list)
        frame_len = []
        for _res in res:
            frame_len.append(_res['mfcc_raw'].values[0].shape[0])
            feats_all = pd.concat([feats_all, _res], ignore_index=True)
        frame_len_max = int(np.ceil(np.mean(frame_len)))  # 7526

        def adjust_len(arr):
            if arr.shape[0] < frame_len_max:
                return np.vstack([arr, np.zeros((frame_len_max - arr.shape[0], arr.shape[1]))])  # 不足部分补零
            else:
                return arr[:frame_len_max, :]  # 超过部分截断

        feats_all['mfcc_raw'] = feats_all['mfcc_raw'].apply(adjust_len)
        feats_all['mfcc_cmvn'] = feats_all['mfcc_cmvn'].apply(adjust_len)
        # print(np.array(feats_all['mfcc_raw'].tolist()).reshape((-1, frame_len_max, 39)).shape)
        feats_all = feats_all.merge(feats_emd, how='outer', on='id')
        feats_all.sort_values(by=['set', 'id'], inplace=True, ignore_index=True)
        feats_all.to_pickle(os.path.join(self.save_dir, 'feats.pkl'))
        print(feats_all)
        return feats_all


def extract_data_from_cha_pitt(input_f_cha: Union[str, List[str]], remove_marker: bool = False) -> pd.DataFrame:
    """
    从.cha文件中提取文本数据
    :param input_f_cha: .cha文件路径或路径列表
    :param remove_marker: 是否删除cha文件中的标记，如停顿、重复、语气等CLAN标记
    :return: pd.DataFrame(par_data_list):speech/clean_speech/clean_par_speech/joined_all_speech/joined_par_speech/
            per_sent_times/total_time/time_before_par_speech/time_between_sents
    """
    if type(input_f_cha) is str:
        input_f_cha = [input_f_cha]
    par_data_list = []
    for cha_file in input_f_cha:
        par = {'id': os.path.basename(cha_file)[:-4]}
        f = iter(open(cha_file, encoding='UTF-8'))
        speech = []
        try:  # 获取PAR和INV文本
            curr_speech = ''
            while True:
                line = next(f)
                if line.startswith('@ID'):
                    participant = [i.strip() for i in line.split('|')]
                    if participant[2] == 'PAR':
                        par['sex'] = np.NAN if (len(participant[4]) == 0) else ['f', 'm'].index(participant[4][0])
                        par['age'] = np.NAN if (len(participant[3]) == 0) else int(participant[3].replace(';', ''))
                        par['label'] = 0 if participant[5] == 'Control' else 1
                        par['mmse'] = np.NAN if (len(participant[8]) == 0) else float(participant[8])
                if line.startswith('*PAR:') or line.startswith('*INV'):
                    curr_speech = line
                elif len(curr_speech) != 0 and not (line.startswith('%') or line.startswith('*')):
                    curr_speech += line
                elif len(curr_speech) > 0:
                    speech.append(curr_speech)
                    curr_speech = ''
        except StopIteration:
            pass
        clean_par_speech = []
        clean_all_speech = []
        par_speech_time_segments = []
        all_speech_time_segments = []
        is_par = False
        for _s in speech:
            def _parse_time(_s):  # 查找时间点
                return [*map(int, re.search('\x15(\d*_\d*)\x15', _s).groups()[0].split('_'))]

            def _clean(_s):  # 删除无关信息
                _s = re.sub('\x15\d*_\d*\x15', '', _s)  # remove time block
                _s = re.sub('\[.*\]', '', _s)  # remove other speech artifacts [.*]
                _s = _s.strip()
                # remove tab, new lines, inferred speech??, ampersand, &, etc
                _s = re.sub('\n\t', ' ', _s)
                _s = re.sub('\t|\n', '', _s)
                if remove_marker:  # remove inferred speech??, ampersand, &, etc marker
                    _s = re.sub('<|>|\[/\]|\[//\]|\[///\]|&=clears throat|=sings|=laughs|=clears:throat|=sighs|=hums|'
                                '=chuckles|=grunt|=finger:tap|=claps|=snif|=coughs|=tapping|\(\.\)|\(\.\.\)|'
                                '\(\.\.\.\)|/|xxx|\+|\(|\)|&', '', _s)
                return _s

            if _s.startswith('*PAR:'):
                is_par = True
            elif _s.startswith('*INV:'):
                is_par = False
                _s = re.sub('\*INV:\t', '', _s)  # remove prefix
            if is_par:
                _s = re.sub('\*PAR:\t', '', _s)  # remove prefix
                par_speech_time_segments.append(_parse_time(_s))
                clean_par_speech.append(_clean(_s))
            all_speech_time_segments.append(_parse_time(_s))
            clean_all_speech.append(_clean(_s))

        par['speech'] = speech  # 原始转录文本
        par['clean_speech'] = clean_all_speech  # 删除无关信息的转录文本，包括PAR和INV，列表，每句话为一个元素
        par['clean_par_speech'] = clean_par_speech  # 删除无关信息的转录文本，仅包括PAR，列表，每句话为一个元素
        par['joined_all_speech'] = ' '.join(clean_all_speech)  # 整合删除无关信息的转录文本，包括PAR和INV
        par['joined_par_speech'] = ' '.join(clean_par_speech)  # 整合删除无关信息的转录文本，仅包括PAR

        # sentence times
        par['per_sent_times'] = [par_speech_time_segments[i][1] - par_speech_time_segments[i][0] for i in
                                 range(len(par_speech_time_segments))]  # 被试PAR每句话的时长
        par['total_time'] = par_speech_time_segments[-1][1] - par_speech_time_segments[0][0]  # PAR全部时间总和
        par['time_before_par_speech'] = par_speech_time_segments[0][0]  # PAR开始叙述的时间点
        par['time_between_sents'] = [
            0 if i == 0 else max(0, par_speech_time_segments[i][0] - par_speech_time_segments[i - 1][1])
            for i in range(len(par_speech_time_segments))]  # PAR每句话的间隔时间，第一句话为距INV结束的时间间隔
        par_data_list.append(par)
    return pd.DataFrame(par_data_list)


class HandcraftedFeaturesPitt(HandcraftedFeatures):
    """获取自发言语任务的手工特征"""
    def __init__(self, input_f_audio: Union[str, parselmouth.Sound], input_f_trans: str, f0min: int = 75,
                 f0max: int = 600, sil_thr: float = -25.0, min_sil: float = 0.1, min_snd: float = 0.1):
        """
        初始化
        :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
        :param input_f_trans: 输入文本转录文件，cha类似的文件格式
        :param f0min: 最小追踪pitch,默认75Hz
        :param f0max: 最大追踪pitch,默认600Hz
        :param sil_thr: 相对于音频最大强度的最大静音强度值(dB)。如imax是最大强度，则最大静音强度计算为sil_db=imax-|sil_thr|
                        强度小于sil_db的间隔被认为是静音段。sil_thr越大，则语音段越可能被识别为静音段，这里默认为-25dB
        :param min_sil: 被认为是静音段的最小持续时间(s)。
                        默认0.1s，即该值越大，则语音段越可能被识别为有声段（若想把爆破音归为有声段，则将该值设置较大）
        :param min_snd: 被认为是有声段的最小持续时间(s)，即不被视为静音段的最小持续时间。
                        默认0.1s，低于该值被认为是静音段，即该值越大，则语音段越可能被识别为静音段
        """
        self.f_audio = input_f_audio
        self.f_trans = input_f_trans
        self.f0min = f0min
        self.f0max = f0max
        self.sound = parselmouth.Sound(self.f_audio)
        self.total_duration = self.sound.get_total_duration()
        self.text_grid_vuv = call(self.sound, "To TextGrid (silences)", 100, 0.0, sil_thr, min_sil, min_snd, 'U', 'V')
        self.vuv_info = call(self.text_grid_vuv, "List", False, 10, False, False)
        self.par_data_all = extract_data_from_cha_pitt(self.f_trans, remove_marker=False)
        par_data = extract_data_from_cha_pitt(self.f_trans, remove_marker=True)
        self.text = par_data.loc[0, 'joined_par_speech']
        self.text_seg_list = nltk.word_tokenize(self.text)  # 分词结果列表
        self.text_seg_list_no_punct = delete_punctuation(self.text_seg_list)  # 删除标点符号后的分词结果
        self.text_posseg_list = nltk.pos_tag(self.text_seg_list, tagset='universal')  # 分词结果列表，含词性(包含所有词性)
        self.sent_num = len(par_data.loc[0, 'clean_par_speech'])
        self.doc = HanLP(self.text)['xlm']  # 基于多任务学习模型的全部结果，Document类型，该结果用于获取除了语篇的其他特征


def feat_handcrafted_pitt(input_f_audio: Union[str, parselmouth.Sound], input_f_trans: str) -> np.ndarray:
    """
    获取自发言语任务的手工特征
    :param input_f_audio: 输入.wav音频文件，或是praat所支持的文件格式
    :param input_f_trans: 输入文本转录文件，cha类似的文件格式
    :return: 14维手工声学/语言学特征 np.ndarray[shape=(14, ), dtype=float32]
    """
    hf = HandcraftedFeaturesPitt(input_f_audio, input_f_trans).get_all_feat()
    np_hf = hf.to_numpy()[0].astype(np.float32)
    return np_hf


class GetFeaturesPitt:
    """计算基于自发言语任务的各类特征"""
    def __init__(self, datasets_dir: Union[str, os.PathLike], save_dir: Union[str, os.PathLike],
                 get_text: bool = True):
        """
        初始化
        :param datasets_dir: 输入包含的数据集文件的路径
        :param save_dir: 数据保存路径
        :param get_text: 是否获取转录文本
        """
        self.text_f_list = glob.glob(os.path.join(datasets_dir, r'Transcription/**/Pitt/**/cookie/*.cha'),
                                     recursive=True)[32:]
        if get_text:
            par_data = extract_data_from_cha_pitt(self.text_f_list, remove_marker=False)
            par_data.to_csv(os.path.join(save_dir, 'trans_pitt.csv'), encoding="utf-8-sig", index=False)
        else:
            par_data = pd.read_csv(os.path.join(save_dir, 'trans_pitt.csv'))
        self.text_subinfo = par_data[['id', 'sex', 'age', 'label', 'mmse',
                                      'clean_par_speech', 'joined_par_speech']]
        self.save_dir = save_dir

    def get_features_noembedding(self, text_file: Union[str, os.PathLike]) -> pd.DataFrame:
        """
        获取对应音频/文本的除文本嵌入的全部特征
        :param text_file: 文本文件
        :return: pd.DataFrame，特征及其对应标签等信息
        """
        print("---------- Processing %d / %d: %s ----------" %
              (self.text_f_list.index(text_file) + 1, len(self.text_f_list), text_file))
        audio_file = text_file.replace('Transcription', 'Media').replace('.cha', '.mp3')
        subid = os.path.basename(text_file)[:-4]
        label = self.text_subinfo[self.text_subinfo['id'] == subid]['label'].values[0]
        sex = self.text_subinfo[self.text_subinfo['id'] == subid]['sex'].values[0]
        age = self.text_subinfo[self.text_subinfo['id'] == subid]['age'].values[0]
        mmse = self.text_subinfo[self.text_subinfo['id'] == subid]['mmse'].values[0]
        feats = pd.DataFrame({'id': [subid], 'sex': [sex], 'age': [age], 'label': [label],
                              'mmse': [mmse]})
        mfcc_raw, mfcc_cmvn = feat_mfcc(audio_file)
        ft_mfcc = pd.DataFrame({'mfcc_cmvn': [mfcc_cmvn]})
        try:
            hd = feat_handcrafted_pitt(audio_file, text_file)
        except:
            hd = np.NAN
        ft_hc = pd.DataFrame({'handcrafted': [hd]})
        feats = pd.concat([feats, ft_mfcc, ft_hc], axis=1)
        return feats

    def get_features(self, n_jobs=None) -> pd.DataFrame:
        """
        并行处理，保存所有特征至本地文件
        :param n_jobs: 并行运行CPU核数，默认为None;若为1非并行，若为-1或None,取os.cpu_count()全部核数,-1/正整数/None类型
        :return: pd.DataFrame，数据的全部特征及其对应标签等信息
        """
        feats_all, feats_emd = pd.DataFrame(), self.text_subinfo[['id']]
        n_samp = len(self.text_subinfo['joined_par_speech'].tolist())
        emb_feat = []
        for i_bs in range(5):
            if i_bs < 4:
                _emb_feat, _ = feat_bert(self.text_subinfo['joined_par_speech'].tolist()[i_bs*n_samp//5:(i_bs+1)*n_samp//5],
                                        "distilbert-base-uncased")
            else:
                _emb_feat, _ = feat_bert(self.text_subinfo['joined_par_speech'].tolist()[i_bs*n_samp//5:],
                                         "distilbert-base-uncased")
            emb_feat += _emb_feat.tolist()
        feats_emd = pd.concat([feats_emd, pd.DataFrame({"distilbert-base-uncased": emb_feat})], axis=1)
        if n_jobs == -1:
            n_jobs = None
        if n_jobs == 1:
            res = []
            for i_subj in self.text_f_list:
                res.append(self.get_features_noembedding(i_subj))
        else:
            with Pool(n_jobs) as pool:
                res = pool.map(self.get_features_noembedding, self.text_f_list)
        for _res in res:
            feats_all = pd.concat([feats_all, _res], ignore_index=True)
        frame_len_max = 7526

        def adjust_len(arr):
            if arr.shape[0] < frame_len_max:
                return np.vstack([arr, np.zeros((frame_len_max - arr.shape[0], arr.shape[1]))])  # 不足部分补零
            else:
                return arr[:frame_len_max, :]  # 超过部分截断

        feats_all['mfcc_cmvn'] = feats_all['mfcc_cmvn'].apply(adjust_len)
        feats_all = feats_all.merge(feats_emd, how='outer', on='id')
        feats_all.dropna(inplace=True)
        feats_all.sort_values(by=['id'], inplace=True, ignore_index=True)
        feats_all.to_pickle(os.path.join(self.save_dir, 'feats_pitt.pkl'))
        print(feats_all)
        return feats_all


def word_cloud_show(trans_csv_f: Union[str, os.PathLike], mask_f: Union[str, os.PathLike],
                    fig_save_dir: Union[str, os.PathLike]):
    """
    显示并保存词云图
    :param trans_csv_f: 转录文本csv文件
    :param mask_f: 词云图底图遮罩
    :param fig_save_dir: 结果图片保存路径
    :return: None
    """
    text_all = pd.read_csv(trans_csv_f)
    text_ad = ' '.join(text_all[text_all['label'] == 1]['joined_par_speech'].tolist())
    text_ad = re.sub('<|>|\[/\]|\[//\]|\[///\]|\(\.\)|\(\.\.\)|\(\.\.\.\)|/|xxx|\+|\(|\)|&', '', text_ad)
    text_hc = ' '.join(text_all[text_all['label'] == 0]['joined_par_speech'].tolist())
    text_hc = re.sub('<|>|\[/\]|\[//\]|\[///\]|\(\.\)|\(\.\.\)|\(\.\.\.\)|/|xxx|\+|\(|\)|&', '', text_hc)
    fd_ad = nltk.FreqDist(text_ad.split(' '))
    fd_hc = nltk.FreqDist(text_hc.split(' '))
    text_ad_hc, text_hc_ad = [], []
    for wd_ad, freq_ad in fd_ad.items():
        if (wd_ad not in fd_hc.keys()) and (freq_ad > 10):
            text_ad_hc.append(wd_ad)
        elif (wd_ad in fd_hc.keys()) and (freq_ad > 10) and (freq_ad > fd_hc[wd_ad]):
            text_ad_hc.append(wd_ad)
    for wd_hc, freq_hc in fd_hc.items():
        if wd_hc not in fd_ad.keys() and (freq_hc > 10):
            text_hc_ad.append(wd_hc)
        elif (wd_hc in fd_ad.keys()) and (freq_hc > 10) and (freq_hc > fd_ad[wd_hc]):
            text_hc_ad.append(wd_hc)
    text_ad_hc, text_hc_ad = ' '.join(text_ad_hc), ' '.join(text_hc_ad)
    stopwords = {'+/.', '+...'}
    wc = WordCloud(max_words=100, collocations=False, background_color="white", min_word_length=2,
                   mask=np.array(Image.open(mask_f)), stopwords=stopwords, max_font_size=160,
                   contour_width=1.5, contour_color='pink', random_state=rs)
    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    fig.add_subplot(1, 2, 1)
    wc.generate(text_ad_hc)
    wc.recolor(random_state=rs)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    fig.add_subplot(1, 2, 2)
    wc.generate(text_hc_ad)
    wc.recolor(random_state=rs)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    fig.tight_layout(pad=0.01, w_pad=0.01)
    fig_file = os.path.join(fig_save_dir, f'wordCloud/wordCloud.png')
    if not os.path.exists(os.path.dirname(fig_file)):
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close('all')


def database_dur(datasets_dir: Union[str, os.PathLike], fig: bool = False) -> pd.DataFrame:
    """
    计算数据集中各个音频时长的统计量
    :param datasets_dir: 输入包含的数据集文件的路径
    :param fig: 是否绘制数据时长频率直方图
    :return: 数据集统计描述
    """
    subj_l, dur_l = [], []
    for wav_f in glob.iglob(os.path.join(datasets_dir, r'*/Full_wave_enhanced_audio/**/*.wav'), recursive=True):
        subj_l.append(wav_f.split(os.sep)[-3])
        dur_l.append(librosa.get_duration(filename=wav_f))
    data_info = pd.DataFrame({"subj": subj_l, "dur": dur_l})
    dat_info = data_info.describe()
    thr_high = dat_info.loc['75%', 'dur'] + 1.5 * (dat_info.loc['75%', 'dur'] - dat_info.loc['25%', 'dur'])
    print(dat_info, thr_high)
    # count=156, dur_mean=75.301155s, dur_std=38.383694s, dur_min=26.064263, dur_max=268.477188,
    # dur_25%=51.919717, dur_50%=70.053005, dur_75%=85.025901, thr_high=134.68517857142854
    if fig:
        plt.figure(figsize=(8, 6), tight_layout=True)
        plt.xlabel('Duration (s)', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Count', fontdict={'family': font_family, 'size': 18})
        ax = sns.histplot(data_info['dur'].to_numpy(), kde=True, color='steelblue',
                          line_kws={"lw": 3}, binwidth=10)
        ax.lines[0].set_color('red')
        plt.xticks(fontsize=16, fontproperties=font_family)
        plt.yticks(fontsize=16, fontproperties=font_family)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = f'results/datasetDur/ADReSS_histplot.png'
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    transparent=True)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.show()
        plt.close('all')
    return dat_info


def correlation(data: pd.DataFrame, save_dir: Union[str, os.PathLike], method='spearman',
                padjust='none') -> pd.DataFrame:
    """
    相关性矩阵
    :param data: 数据，包含预测值，即y
    :param save_dir: 保存结果路径
    :param method: 计算相关性的方法
    :param padjust: p值校正方法
    :return: 相关性矩阵df_r_p_value，下三角为r值，上三角为p值（若校正则为校正后的）
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df_r_p_value = data.rcorr(method=method, upper='pval', decimals=6, padjust=padjust, stars=False)
    df_r_p_value.to_csv(os.path.join(save_dir, "r_p_value.csv"))
    res_sign_corr = data.rcorr(method=method, upper='pval', decimals=6, padjust=padjust, stars=True)
    res_sign_corr.to_csv(os.path.join(save_dir, "sign_corr.csv"))
    print(df_r_p_value)
    r_matrix = np.tril(df_r_p_value.to_numpy(), k=-1).astype(np.float32) + \
               np.tril(df_r_p_value.to_numpy(), k=-1).T.astype(np.float32) + np.eye(df_r_p_value.shape[0])
    p_matrix = np.triu(df_r_p_value.to_numpy(), k=1).astype(np.float32) + \
               np.triu(df_r_p_value.to_numpy(), k=1).T.astype(np.float32) + np.eye(df_r_p_value.shape[0])
    pval_stars = {0.001: '***', 0.01: '**', 0.05: '*'}

    def replace_pval(x):
        for key, value in pval_stars.items():
            if x < key:
                return value
        return ''
    p_matrix_star = pd.DataFrame(p_matrix).applymap(replace_pval).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7), tight_layout=True)
    mask = np.zeros_like(r_matrix, dtype=bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型,此时全为False
    mask[np.triu_indices_from(mask, k=1)] = True  # 返回除对角线的矩阵上三角，并将其设置为true，作为热力图掩码进行屏蔽
    ax_up = sns.heatmap(r_matrix, mask=mask, annot=True, ax=ax, cbar=False,
                        annot_kws={'size': 8, 'color': 'k'}, fmt='.2f', square=True,
                        cmap="coolwarm", xticklabels=False, yticklabels=False)  # 下三角r值
    mask = np.zeros_like(r_matrix, dtype=bool)
    mask[np.tril_indices_from(mask)] = True  # 返回包含对角线的矩阵下三角，并将其设置为true，作为热力图掩码进行屏蔽'shrink': 1.0,
    cbar_kws = {'aspect': 30, "format": "%.2f", "pad": 0.01}
    ax_lo = sns.heatmap(r_matrix, mask=mask, annot=p_matrix_star, ax=ax_up,
                        annot_kws={'size': 14, 'weight': 'bold', 'color': 'k'}, fmt='', square=True, vmin=-1, vmax=1,
                        cmap="coolwarm", xticklabels=df_r_p_value.columns, yticklabels=df_r_p_value.index,
                        cbar_kws=cbar_kws)  # 上三角p值
    cbar = ax_lo.collections[-1].colorbar
    # cbar.ax.set_ylabel('相关性系数', fontdict={'family': font_family, 'size': 16})
    cbar.ax.tick_params(labelsize=12, length=3)
    cbar.outline.set_visible(False)
    plt.tick_params(bottom=False, left=False)
    ax_lo.set_xticklabels(ax_lo.get_xticklabels(), fontsize=14, fontproperties=font_family, rotation=45,
                          ha="right", rotation_mode="anchor")
    ax_lo.set_yticklabels(ax_lo.get_yticklabels(), fontsize=14, fontproperties=font_family, rotation=0)
    fig_file = os.path.join(save_dir, f'corr_heatmap.png')
    if not os.path.exists(os.path.dirname(fig_file)):
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
    plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                transparent=True)
    plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    plt.close()
    return df_r_p_value


def ttest(data_x, data_y, parametric=True, alternative='two-sided', paired=False) -> pd.DataFrame:
    """
    T检验和对应的非参数检验：单样本（data_y为一单值时）、独立样本（当数据不符合方差齐性假设时，这里自动使用Welch-t检验校正）、配对样本
    :param data_x: 输入数据，array_like
    :param data_y: 输入数据，array_like or float，当为单值时，使用单样本T检验
    :param parametric: 是否使用非参数检验, False时使用非参数检验
    :param alternative: 备择假设：双尾/单尾（'two-sided'/'one-sided'/'greater'/'less'）
    :param paired: 是否为配对样本, True时使用配对样本检验
    :return: pandas.DataFrame，对应的检验统计结果
    """
    if parametric:  # 数据符合正态分布，参数检验：T检验
        # T检验：单样本（data_y为一单值时）、独立样本（当数据不符合方差齐性假设时，这里自动使用Welch-t检验校正）、配对样本
        res = pg.ttest(data_x, data_y, paired, alternative)
    else:  # 数据不符合正态分布，非参数检验
        if paired:  # 配对样本，使用Wilcoxon signed-rank test Wilcoxon符号秩检验
            res = pg.wilcoxon(data_x, data_y, alternative)
        else:  # 独立样本，使用Wilcoxon rank-sum test Wilcoxon秩和检验，即Mann-Whitney U Test
            res = pg.mwu(data_x, data_y, alternative)
    return res


def adjust_box_widths(ax, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """
    for c in ax.get_children():
        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)
            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])
        if isinstance(c, Line2D):
            if c.get_label() == 'cap':
                xmin = float(c.get_xdata()[0])
                xmax = float(c.get_xdata()[-1])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                c.set_xdata([xmin_new, xmax_new])


def handcrafted_ttest(data: pd.DataFrame, save_dir: Union[str, os.PathLike], between=None, parametric=None,
                      outliers_detect: bool = True, plot_kind: str = 'raincloud', grp_name=None) -> pd.DataFrame:
    """
    手工特征进行T检验并绘图
    :param data: 数据，包含标签，即label
    :param save_dir: 结果保存路径
    :param between: 包含组别的列的名称，即标签名，string类型
    :param parametric: T检验/对应的非参数检验，True/False/None，None为根据每次检验的数据的正态性进行自动判断
    :param outliers_detect: 是否进行异常值检测（对每列特征的异常值赋值为nan）
    :param plot_kind: 绘图类型，可选violin/raincloud
    :param grp_name: list或OrderedDict类型，元素为string类型,可视化中，纵坐标，即组的名称映射，顺序为从上到下的显示顺序
                list: ['0', '1']，按照所列的顺序从上到下依次显示（元素必须存在于data[between]列中）
                OrderedDict: {'0':'first', '1':'second'}，
                将键替换为值，并按照所列的顺序从上到下依次显示（键必须存在于data[between]列中），替换后的值同时在save_dir文件中对应更改
    :return: t检验结果
    """
    data[between] = data[between].astype(str)
    if isinstance(grp_name, list):
        order = grp_name
    elif isinstance(grp_name, OrderedDict):
        data[between] = data[between].map(grp_name)
        order = list(grp_name.values())
    elif grp_name is None:
        order = data[between].unique().tolist()
    else:
        raise TypeError(f'{grp_name} 错误格式，应为list或OrderedDict格式')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    res = []
    for i_feat in data.drop(columns=[between]).columns:
        data_x = data[data[between] == order[0]].reset_index(drop=True)[i_feat]
        data_y = data[data[between] == order[-1]].reset_index(drop=True)[i_feat]
        # 利用箱线图法，即根据四分位数进行异常值检测
        if outliers_detect:
            x_thr_high = data_x.quantile(0.75) + 1.5 * (data_x.quantile(0.75) - data_x.quantile(0.25))
            x_thr_low = data_x.quantile(0.25) - 1.5 * (data_x.quantile(0.75) - data_x.quantile(0.25))
            y_thr_high = data_y.quantile(0.75) + 1.5 * (data_y.quantile(0.75) - data_y.quantile(0.25))
            y_thr_low = data_y.quantile(0.25) - 1.5 * (data_y.quantile(0.75) - data_y.quantile(0.25))
            # 异常值则判为无效数据，此时清空，即赋值为nan
            data_x_filter = data_x[(data_x >= x_thr_low) & (data_x <= x_thr_high)]
            data_y_filter = data_y[(data_y >= y_thr_low) & (data_y <= y_thr_high)]
        else:
            data_x_filter = data_x.copy()
            data_y_filter = data_y.copy()
        x_data = data_x_filter
        y_data = data_y_filter
        if parametric is None:  # 根据正态性自动判断
            stat, p = stats.shapiro(x_data - y_data)  # 正态性检验：Shapiro-Wilk Test
            if p < 0.05:  # 非正态分布，使用非参数检验
                parametric = False
            else:
                parametric = True
        t_res = ttest(x_data, y_data, parametric)
        par_dict = {}
        for k_par in t_res.columns:
            par_dict[k_par] = t_res[k_par][0]
        res.append(par_dict)
        pval = t_res['p-val'][0]
        plt.figure(figsize=(6, 6), clear=True, tight_layout=True)
        plot_args = {'data': data, 'x': between, 'y': i_feat, 'order': order, 'orient': 'v'}
        if plot_kind == 'violin':
            ax = sns.violinplot(width=.6, cut=2, palette="pastel", scale="area", inner=None, **plot_args)
            ax = sns.boxplot(width=.6, color="gray", zorder=10, boxprops={'facecolor': 'none', "zorder": 20},
                             showfliers=True, ax=ax, capprops={'label': 'cap'}, **plot_args)
            # ax = sns.swarmplot(palette="colorblind", size=5, ax=ax, **plot_args)
            ax = pt.stripplot(width=.6, size=5, dodge=True, ax=ax, zorder=5, **plot_args)
            adjust_box_widths(ax, .2)
            ax.set_ylim(ax.get_ylim()[0], 1.25 * ax.get_ylim()[-1])
        elif plot_kind == 'raincloud':
            plot_args['orient'] = 'h'
            ax = pt.RainCloud(cut=2, width_viol=1., pointplot=True, point_size=5,
                              alpha=.65, dodge=True, move=.2, **plot_args)
        else:
            raise ValueError(f'无效的绘图类型：{plot_kind}，请从violin/raincloud中选择!')
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        plt.xticks(fontproperties=font_family)
        plt.yticks(fontproperties=font_family)
        plt.gca().tick_params(labelsize=18, direction='in', color='k', length=6, width=1.5)
        pairs = [(order[0], order[-1])]
        if plot_args['orient'] == 'h':
            plot_args['x'], plot_args['y'] = i_feat, between
            annotator = Annotator(ax, pairs, **plot_args)
            plt.ylabel('')
            ax.xaxis.label.set_size(24)
            plt.tick_params('y', labelsize=24)
        else:
            annotator = Annotator(ax, pairs, plot='violinplot', **plot_args)
            plt.xlabel('')
            ax.yaxis.label.set_size(24)
            plt.tick_params('x', labelsize=24)
        # annotator.configure(text_format='full', loc='inside', fontsize=16, show_test_name='', text_offset=5)
        annotator.configure(text_format='star', loc='inside', fontsize=24)
        annotator.set_pvalues([pval]).annotate()
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.grid(False)
        f_name = re.sub(r' \([\S\s]*\)$', '', i_feat)
        fig_file = os.path.join(save_dir, f'{f_name}.png')
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    transparent=True)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.show()
        plt.close()

    p_value_de = pd.DataFrame(res, index=data.drop(columns=[between]).columns)
    print(p_value_de)
    p_value_de.to_csv(os.path.join(save_dir, 'ttest_detail.csv'), encoding="utf-8-sig")
    return p_value_de


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(current_path, r"data")
    res_path = os.path.join(current_path, r"results")
    test_info_f = os.path.join(data_path, r"testSetInfo.csv")

    # count=156, dur_mean=75.301155s, dur_std=38.383694s, dur_min=26.064263, dur_max=268.477188,
    # dur_25%=51.919717, dur_50%=70.053005, dur_75%=85.025901, thr_high=134.68517857142854
    database_dur(DATA_PATH, fig=True)
    # GetFeatures(DATA_PATH, save_dir=data_path, get_text=False, test_info_file=test_info_f).get_features(1)
    word_cloud_show(os.path.join(data_path, 'trans.csv'), os.path.join(data_path, 'brain.jpg'), fig_save_dir=res_path)
    feat_data = pd.read_pickle(os.path.join(current_path, r'data/feats.pkl'))
    # 缺失值分组均值填充
    feat_data['mmse'].fillna(feat_data[(feat_data['set'] == 'train') & (feat_data['label'] == 0)]['mmse'].mean(),
                             inplace=True)
    feat_data.rename(columns={'mmse': 'MMSE', 'Empty Word Freq': 'EWF'}, inplace=True)
    feat_data_hd = pd.concat([feat_data[['label', 'MMSE']],
                              pd.DataFrame(feat_data['handcrafted'].tolist(),
                                           columns=["F0 SD", "DPI", "Voiced Rate", "Hesitation Ratio",
                                                    "EWF", "Word Rate", "Function Word Ratio",
                                                    "Lexical Density", "MLU", "Noun Phrase Rate", "Verb Phrase Rate",
                                                    "Parse Tree Height", "Yngve Depth Total",
                                                    "Dependency Distance Total"])], axis=1)
    # 手工特征组间差异
    handcrafted_ttest(feat_data_hd.drop(columns=['MMSE']), save_dir=os.path.join(res_path, 'ttest'), between='label',
                      outliers_detect=True, parametric=True, plot_kind='violin',
                      grp_name=OrderedDict({'0': 'HC', '1': 'AD'}))
    # 手工特征相关性
    correlation(feat_data_hd.drop(columns=['label']), save_dir=os.path.join(res_path, 'corr'),
                method='spearman', padjust='none')

    # 获取Pitt数据集对应特征：排除无法提取手工特征的样本（有严重噪音等问题的音频），共443个样本(272个痴呆，171个对照)
    GetFeaturesPitt(DATA_PATH_PITT, save_dir=data_path, get_text=False).get_features(1)
    pitt_info = pd.read_pickle(os.path.join(data_path, r'feats_pitt.pkl'))[['id', 'sex', 'age', 'mmse', 'label']]
    pitt_info.to_csv(os.path.join(data_path, 'pittInfo.csv'), encoding="utf-8-sig", index=False)
    print(pitt_info)
    mean_sd = lambda x: f"{x.mean():.2f} ({x.std(ddof=0):.2f})" if not x.empty else "N/A"
    n_ad_m = pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 1)].shape[0]
    n_ad_f = pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 0)].shape[0]
    n_hc_m = pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 1)].shape[0]
    n_hc_f = pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 0)].shape[0]
    age_ad_m = mean_sd(pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 1)]['age'])
    age_ad_f = mean_sd(pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 0)]['age'])
    age_hc_m = mean_sd(pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 1)]['age'])
    age_hc_f = mean_sd(pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 0)]['age'])
    mmse_ad_m = mean_sd(pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 1)]['mmse'])
    mmse_ad_f = mean_sd(pitt_info[(pitt_info['label'] == 1) & (pitt_info['sex'] == 0)]['mmse'])
    mmse_hc_m = mean_sd(pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 1)]['mmse'])
    mmse_hc_f = mean_sd(pitt_info[(pitt_info['label'] == 0) & (pitt_info['sex'] == 0)]['mmse'])
    result_table = pd.DataFrame({f'AD (n = {n_ad_m+n_ad_f})': [f"M (n = {n_ad_m})", age_ad_m, mmse_ad_m,
                                                               f"F (n = {n_ad_f})", age_ad_f, mmse_ad_f],
                                 f'Controls (n = {n_hc_m+n_hc_f})': [f"M (n = {n_hc_m})", age_hc_m, mmse_hc_m,
                                                                     f"F (n = {n_hc_f})", age_hc_f, mmse_hc_f]},
                                index=["", "Age", "MMSE", "", "Age", "MMSE"])
    print(result_table)

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(os.path.join(current_path, r"results/finished.txt"), "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")

