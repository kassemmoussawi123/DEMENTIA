# -*- coding: utf-8 -*-
# Copyright (c) 2024. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2023/11/14 20:14
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : models.py
# @Software : Python 3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.1.0 - ZL.Z：2024/9/10
# @License  : None
# @Brief    : 多模态AD检测模型

from config import *
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, LayerNormalization, BatchNormalization, \
    Activation, MultiHeadAttention, Conv1D, Concatenate, Add, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras import losses, metrics, callbacks, optimizers
from attention import Attention
from typing import Union, Tuple, Any
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import seaborn as sns
from cycler import cycler
from concretedropout.tensorflow import ConcreteDenseDropout, get_weight_regularizer, get_dropout_regularizer
from transformers import logging
from adjustText import adjust_text

logging.set_verbosity_error()


def scaled_sigmoid(x: tf.Tensor) -> tf.Tensor:
    return tf.multiply(tf.sigmoid(x), 30)


class DementiaDetectionModel:
    """
    联合混合注意力与多模态表征的多任务学习AD检测模型（DEMENTIA）
    joint hybriD attEntion and Multimodal rEpreseNtation with mulTI-task leArning for AD detection model
    """

    def __init__(self, data_file: Union[str, os.PathLike], params_config: dict[str, Any], 
                 model_save_dir: Union[str, os.PathLike], model_name: str = "DEMENTIA"):
        """
        初始化
        :param data_file: 数据集文件
        :param params_config: 模型参数配置字典
        :param model_save_dir: 模型保存路径
        :param model_name: 模型名称
        """
        self.config = params_config
        if {'sig_reg', 'sig_cls'}.issubset(self.config.keys()):
            assert not (self.config['sig_reg'] and self.config['sig_cls']), "参数sig_reg和sig_cls不能同时为True"
        if {'audio', 'text', 'handcraft'}.issubset(self.config.keys()):
            assert self.config['audio'] or self.config['text'] or self.config['handcraft'], \
                "参数audio/text/handcraft不能同时为False"
        data_file = os.path.normpath(data_file)
        if data_file.endswith('pkl'):
            feat_data = pd.read_pickle(data_file)  # type: pd.DataFrame
        else:
            raise ValueError('无效数据，仅接受.pkl数据集文件')
        feat_data = feat_data.sample(frac=1, random_state=rs).reset_index(drop=True)  # 打乱样本
        feat_data['mmse'].fillna(feat_data[(feat_data['set'] == 'train') & (feat_data['label'] == 0)]['mmse'].mean(),
                                 inplace=True)
        bert = self.config['bert']
        # shape=[样本数，音频序列长度，特征维数]=[108,7526,39]
        self.train_data_audio = np.array(feat_data[feat_data['set'] == 'train']['mfcc_cmvn'].tolist(), dtype=np.float32)
        # shape=[108,7526]
        train_audio_mask = np.where(np.ma.masked_equal(self.train_data_audio, 0).mask, 0, 1)[:, :, 0]
        self.train_audio_mask = train_audio_mask[:, tf.newaxis]  # shape=(B, T, S)=(108, 1, 7526)，后面T和H会自动广播
        self.train_data_text = np.array(feat_data[feat_data['set'] == 'train'][bert].tolist(),
                                        dtype=np.float32)  # shape=[样本数，文本序列长度，特征维数]=[108,510,768]
        train_text_mask = np.array(feat_data[feat_data['set'] == 'train'][f'mask_{bert}'].tolist(),
                                   dtype=np.float32)  # shape=[样本数，文本序列长度]=[108,510]
        self.train_text_mask = train_text_mask[:, tf.newaxis]  # shape=(B, T, S)=(108, 1, 510)，后面T和H会自动广播
        train_data_hand = np.array(feat_data[feat_data['set'] == 'train']['handcrafted'].tolist(),
                                   dtype=np.float32)  # shape=[样本数，特征维数]=[108,14]
        ss_hand = StandardScaler()
        self.train_data_hand = ss_hand.fit_transform(train_data_hand)
        self.train_label = np.array(feat_data[feat_data['set'] == 'train']['label'].tolist(), dtype=int)
        self.train_mmse = np.array(feat_data[feat_data['set'] == 'train']['mmse'].tolist(), dtype=np.float16)
        self.train_age = np.array(feat_data[feat_data['set'] == 'train']['age'].tolist(), dtype=np.float16)
        self.train_sex = np.array(feat_data[feat_data['set'] == 'train']['sex'].tolist(), dtype=int)

        self.test_data_audio = np.array(feat_data[feat_data['set'] == 'test']['mfcc_cmvn'].tolist(), dtype=np.float32)
        test_audio_mask = np.where(np.ma.masked_equal(self.test_data_audio, 0).mask, 0, 1)[:, :, 0]
        self.test_audio_mask = test_audio_mask[:, tf.newaxis]
        self.test_data_text = np.array(feat_data[feat_data['set'] == 'test'][bert].tolist(),
                                       dtype=np.float32)
        test_text_mask = np.array(feat_data[feat_data['set'] == 'test'][f'mask_{bert}'].tolist(), dtype=np.float32)
        self.test_text_mask = test_text_mask[:, tf.newaxis]
        test_data_hand = np.array(feat_data[feat_data['set'] == 'test']['handcrafted'].tolist(), dtype=np.float32)
        self.test_data_hand = ss_hand.transform(test_data_hand)
        self.test_label = np.array(feat_data[feat_data['set'] == 'test']['label'].tolist(), dtype=int)
        self.test_mmse = np.array(feat_data[feat_data['set'] == 'test']['mmse'].tolist(), dtype=np.float16)
        self.test_age = np.array(feat_data[feat_data['set'] == 'test']['age'].tolist(), dtype=np.float16)
        self.test_sex = np.array(feat_data[feat_data['set'] == 'test']['sex'].tolist(), dtype=int)
        # print(self.train_data_audio.shape, self.test_audio_mask.shape, self.train_data_text.shape,
        #       self.train_text_mask.shape, self.train_data_hand.shape)
        self.model_save_dir = os.path.join(model_save_dir, model_name)  # 保存模型路径
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.num_heads = 4
        self.model_name = model_name
        self.wr = get_weight_regularizer(self.train_label.shape[0], l=1e-2, tau=1.0)
        self.dr = get_dropout_regularizer(self.train_label.shape[0], tau=1.0, cross_entropy_loss=True)

    def model_create_modal(self):
        """
        构建模型：输入模态选择
        :return: 返回模型
        """
        if self.config['handcraft']:
            input_hand = Input(shape=(self.train_data_hand.shape[-1]), name="in_h")  # (, 14)
        if self.config['audio']:
            input_audio = Input(shape=(self.train_data_audio.shape[1], self.train_data_audio.shape[-1]),
                                name="in_a")  # (, 7526, 39)
            att_mask_a = Input(shape=(self.train_audio_mask.shape[1], self.train_audio_mask.shape[-1]),
                               name="mask_a")  # (, 1, 7526)
            # 音频编码器
            bilstm = Bidirectional(LSTM(128, return_sequences=True))(input_audio)  # shape=(None, 7526, 128*2)
            bilstm_ap = AveragePooling1D(pool_size=7)(bilstm)  # (, 1075, 128*2)
            att_mask_a_mp = MaxPooling1D(pool_size=7, data_format='channels_first')(att_mask_a)  # (, 1, 1075)
            mha_a = MultiHeadAttention(num_heads=self.num_heads, key_dim=bilstm_ap.shape[-1], name="mha_a")
            att_a = mha_a(query=bilstm_ap, value=bilstm_ap, attention_mask=att_mask_a_mp)  # shape=(None, 1075, 128*2)
        if self.config['text']:
            # 文本编码器
            input_text = Input(shape=(self.train_data_text.shape[1], self.train_data_text.shape[-1]),
                               name="in_t")  # (, 510, 768)
            att_t = input_text  # (, 510, 768)
        if self.config['audio'] and self.config['text']:
            # 跨模态注意力
            cmha_in_a = Conv1D(filters=128, kernel_size=att_a.shape[1] - att_t.shape[1] + 1, strides=1,
                               padding='valid', use_bias=False)(att_a)  # (,510,128)
            cmha_in_t = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(att_t)  # (,510,128)
            cmha_a = MultiHeadAttention(num_heads=self.num_heads, key_dim=cmha_in_t.shape[-1], name="cmha_a")
            cmha_out_a = cmha_a(query=cmha_in_a, key=cmha_in_t, value=cmha_in_t)  # (, 510, 128)
            ln_a = LayerNormalization()(cmha_out_a)  # (, 510, 128)
            out_a = Add()([cmha_in_a, ln_a])  # (, 510, 128)
            cmha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=cmha_in_a.shape[-1], name="cmha_t")
            cmha_out_t = cmha_t(query=cmha_in_t, key=cmha_in_a, value=cmha_in_a)  # (, 510, 128)
            ln_t = LayerNormalization()(cmha_out_t)  # (, 510, 128)
            out_t = Add()([cmha_in_t, ln_t])  # (, 510, 128)
            cmha_out = Concatenate()([out_a, out_t])  # (, 510, 128*2)
            global_att_luong = Attention(128, name="attention")(cmha_out)  # (, 128)
            input_at = ConcreteDenseDropout(Dense(64), weight_regularizer=self.wr,
                                            dropout_regularizer=self.dr)(global_att_luong)  # (, 64)
            if self.config['handcraft']:
                fusion = Concatenate()([input_at, input_hand])  # (None, 64+14)
                inputs = [input_audio, att_mask_a, input_text, input_hand]
            else:
                fusion = input_at
                inputs = [input_audio, att_mask_a, input_text]
        elif self.config['audio'] and self.config['handcraft']:
            gap = GlobalAveragePooling1D()(att_a)  # shape=(, 128*2)
            fusion = Concatenate()([gap, input_hand])  # (, 128*2+14)
            inputs = [input_audio, att_mask_a, input_hand]
        elif self.config['text'] and self.config['handcraft']:
            gap = GlobalAveragePooling1D()(att_t)  # shape=(, 768)
            fusion = Concatenate()([gap, input_hand])  # (, 768+14)
            inputs = [input_text, input_hand]
        elif self.config['audio']:
            fusion = GlobalAveragePooling1D()(att_a)
            inputs = [input_audio, att_mask_a]
        elif self.config['text']:
            fusion = GlobalAveragePooling1D()(att_t)
            inputs = input_text
        elif self.config['handcraft']:
            fusion = input_hand
            inputs = input_hand
        # 多任务学习
        layer_dp = ConcreteDenseDropout(Dense(32), weight_regularizer=self.wr, dropout_regularizer=self.dr)(fusion)
        layer_bn = BatchNormalization()(layer_dp)
        layer_ac = Activation("relu")(layer_bn)
        output_reg_mmse = Dense(1, activation=scaled_sigmoid, name="reg_mmse")(layer_ac)
        output_cls_ad = Dense(1, activation="sigmoid", name="cls_ad")(layer_ac)
        if self.config['sig_reg']:
            model = Model(inputs=inputs, outputs=output_reg_mmse, name=self.model_name)
            loss = {"reg_mmse": losses.MeanSquaredError()}
            loss_weights = None
            mts = {"reg_mmse": metrics.RootMeanSquaredError()}
        elif self.config['sig_cls']:
            model = Model(inputs=inputs, outputs=output_cls_ad, name=self.model_name)
            loss = {"cls_ad": losses.BinaryCrossentropy()}
            loss_weights = None
            mts = {"cls_ad": "acc"}
        else:
            model = Model(inputs=inputs, outputs=[output_reg_mmse, output_cls_ad], name=self.model_name)
            loss = {"reg_mmse": losses.MeanSquaredError(), "cls_ad": losses.BinaryCrossentropy()}
            # loss_weights = {"reg_mmse": 0.05, "cls_ad": 8.}
            loss_weights = {"reg_mmse": self.config['lw_cls'] / 100, "cls_ad": self.config['lw_cls']}
            mts = {"reg_mmse": metrics.RootMeanSquaredError(), "cls_ad": "acc"}
        opt = optimizers.SGD(learning_rate=self.config['lr'], momentum=0.9, nesterov=True)
        model.compile(loss=loss, optimizer=opt, metrics=mts, loss_weights=loss_weights)
        # model.summary()
        plot_model(model, to_file=f'{self.model_save_dir}/model_summary.png', show_shapes=True,
                   dpi=600, expand_nested=False)
        return model

    def model_create(self):
        """
        构建模型
        :return: 返回模型
        """
        if {'audio', 'text', 'handcraft'}.issubset(self.config.keys()):
            if not (self.config['audio'] and self.config['text'] and self.config['handcraft']):
                return self.model_create_modal()
        # 多模态输入
        input_audio = Input(shape=(self.train_data_audio.shape[1], self.train_data_audio.shape[-1]), name="in_a")  # (, 7526, 39)
        att_mask_a = Input(shape=(self.train_audio_mask.shape[1], self.train_audio_mask.shape[-1]), name="mask_a")  # (, 1, 7526)
        input_text = Input(shape=(self.train_data_text.shape[1], self.train_data_text.shape[-1]), name="in_t")  # (, 510, 768)
        input_hand = Input(shape=(self.train_data_hand.shape[-1]), name="in_h")  # (, 14)
        # 音频编码器
        bilstm = Bidirectional(LSTM(128, return_sequences=True))(input_audio)  # shape=(None, 7526, 128*2)
        bilstm_ap = AveragePooling1D(pool_size=7)(bilstm)  # (, 1075, 128*2)
        if self.config['mha_audio']:
            att_mask_a_mp = MaxPooling1D(pool_size=7, data_format='channels_first')(att_mask_a)  # (, 1, 1075)
            mha_a = MultiHeadAttention(num_heads=self.num_heads, key_dim=bilstm_ap.shape[-1], name="mha_a")
            att_a = mha_a(query=bilstm_ap, value=bilstm_ap, attention_mask=att_mask_a_mp)  # shape=(None, 1075, 128*2)
        else:
            att_a = bilstm_ap  # (, 1075, 128*2)
        # 文本编码器
        att_t = input_text  # (, 510, 768)
        # 跨模态注意力
        # 为保证音频和文本处在同一特征尺度空间，通过卷积核大小对音频进行减少尺寸：o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # 这里o=510,i=26843,p=0,d=1,s=1,即输出=(输入-kernel size)/strides+1
        # 公式ref：https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        cmha_in_a = Conv1D(filters=128, kernel_size=att_a.shape[1] - att_t.shape[1] + 1, strides=1,
                           padding='valid', use_bias=False)(att_a)  # (,510,128)
        cmha_in_t = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)(att_t)  # (,510,128)
        if self.config['mha_cross']:
            cmha_a = MultiHeadAttention(num_heads=self.num_heads, key_dim=cmha_in_t.shape[-1], name="cmha_a")
            cmha_out_a = cmha_a(query=cmha_in_a, key=cmha_in_t, value=cmha_in_t)  # (, 510, 128)
            ln_a = LayerNormalization()(cmha_out_a)  # (, 510, 128)
            out_a = Add()([cmha_in_a, ln_a])  # (, 510, 128)
            cmha_t = MultiHeadAttention(num_heads=self.num_heads, key_dim=cmha_in_a.shape[-1], name="cmha_t")
            cmha_out_t = cmha_t(query=cmha_in_t, key=cmha_in_a, value=cmha_in_a)  # (, 510, 128)
            ln_t = LayerNormalization()(cmha_out_t)  # (, 510, 128)
            out_t = Add()([cmha_in_t, ln_t])  # (, 510, 128)
            cmha_out = Concatenate()([out_a, out_t])  # (, 510, 128*2)
        else:
            cmha_out = Concatenate()([cmha_in_a, cmha_in_t])  # (, 510, 128*2)
        if self.config['att_global']:
            global_att_luong = Attention(128, name="attention")(cmha_out)  # (, 128)
        else:
            global_att_luong = GlobalAveragePooling1D()(cmha_out)  # shape=(, 128*2)
        input_at = ConcreteDenseDropout(Dense(64), weight_regularizer=self.wr,
                                        dropout_regularizer=self.dr)(global_att_luong)  # (, 64)
        fusion = Concatenate()([input_at, input_hand])  # (None, 64+14)
        # 多任务学习
        layer_dp = ConcreteDenseDropout(Dense(32), weight_regularizer=self.wr, dropout_regularizer=self.dr)(fusion)  # (, 32)
        layer_bn = BatchNormalization()(layer_dp)
        layer_ac = Activation("relu")(layer_bn)
        output_reg_mmse = Dense(1, activation=scaled_sigmoid, name="reg_mmse")(layer_ac)
        output_cls_ad = Dense(1, activation="sigmoid", name="cls_ad")(layer_ac)
        inputs = [input_audio, att_mask_a, input_text, input_hand]
        if self.config['sig_reg']:
            model = Model(inputs=inputs, outputs=output_reg_mmse, name=self.model_name)
            loss = {"reg_mmse": losses.MeanSquaredError()}
            loss_weights = None
            mts = {"reg_mmse": metrics.RootMeanSquaredError()}
        elif self.config['sig_cls']:
            model = Model(inputs=inputs, outputs=output_cls_ad, name=self.model_name)
            loss = {"cls_ad": losses.BinaryCrossentropy()}
            loss_weights = None
            mts = {"cls_ad": "acc"}
        else:
            model = Model(inputs=inputs, outputs=[output_reg_mmse, output_cls_ad], name=self.model_name)
            loss = {"reg_mmse": losses.MeanSquaredError(), "cls_ad": losses.BinaryCrossentropy()}
            # loss_weights = {"reg_mmse": 0.05, "cls_ad": 8.}
            loss_weights = {"reg_mmse": self.config['lw_cls'] / 100, "cls_ad": self.config['lw_cls']}
            mts = {"reg_mmse": metrics.RootMeanSquaredError(), "cls_ad": "acc"}
        opt = optimizers.SGD(learning_rate=self.config['lr'], momentum=0.9, nesterov=True)
        model.compile(loss=loss, optimizer=opt, metrics=mts, loss_weights=loss_weights)
        # model.summary()
        plot_model(model, to_file=f'{self.model_save_dir}/model_summary.png', show_shapes=True,
                   dpi=600, expand_nested=False)
        return model

    def model_train_evaluate(self, fit: bool = True):
        """
        模型训练与评估
        :param fit: 是否进行训练
        :return: 模型结果
        """
        save_model = os.path.join(self.model_save_dir, f"{self.model_name}.h5")
        callback_es = callbacks.EarlyStopping(monitor='loss', patience=10)
        callback_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5)
        callback_cp = callbacks.ModelCheckpoint(filepath=save_model, save_best_only=True, monitor="val_loss")
        cbks = [callback_es, callback_lr, callback_cp]
        batch_size, epochs = self.config['batch_size'], self.config['epochs']
        inputs_train, inputs_test = {}, {}
        for i_modal, i_in in {'audio': [{"in_a": self.train_data_audio, "mask_a": self.train_audio_mask},
                                        {"in_a": self.test_data_audio, "mask_a": self.test_audio_mask}],
                              'text': [{"in_t": self.train_data_text}, {"in_t": self.test_data_text}],
                              'handcraft': [{"in_h": self.train_data_hand}, {"in_h": self.test_data_hand}]}.items():
            if self.config[i_modal]:
                inputs_train.update(i_in[0])
                inputs_test.update(i_in[-1])
        outputs = {"reg_mmse": self.train_mmse} if self.config['sig_reg'] else {"cls_ad": self.train_label} if \
            self.config['sig_cls'] else {"reg_mmse": self.train_mmse, "cls_ad": self.train_label}
        with distribute_strategy.scope():
            if fit:
                model = self.model_create()
                model.fit(inputs_train, outputs, validation_split=0.3,
                          batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True, callbacks=cbks)
            else:
                if os.path.exists(save_model):  # 存在已训练模型且设置加载，
                    print("----------加载模型：{}----------".format(save_model))
                else:
                    raise FileNotFoundError("模型不存在，无法评估，请先训练")
            custom_objects = {'Attention': Attention, 'scaled_sigmoid': scaled_sigmoid,
                              'ConcreteDenseDropout': ConcreteDenseDropout}
            model_best = load_model(save_model, custom_objects=custom_objects)
            model_out_tr = model_best(inputs_train)
            model_out = model_best(inputs_test)
        mts_res = {}
        for lab, yt_mmse, yt_label, yout in [('mts_train', self.train_mmse, self.train_label, model_out_tr),
                                             ('mts_test', self.test_mmse, self.test_label, model_out)]:
            acc, precision, recall, f1_score, rmse = np.nan, np.nan, np.nan, np.nan, np.nan
            if self.config['sig_reg'] or not self.config['sig_cls']:
                y_pred_mmse = yout.numpy().flatten() if self.config['sig_reg'] else yout[0].numpy().flatten()
                rmse = mean_squared_error(yt_mmse, y_pred_mmse, squared=False)
            if self.config['sig_cls'] or not self.config['sig_reg']:
                y_pred_proba_ad = yout.numpy().flatten() if self.config['sig_cls'] else yout[-1].numpy().flatten()
                y_pred_label_ad = (y_pred_proba_ad > 0.5).astype("int32")
                acc = accuracy_score(yt_label, y_pred_label_ad)
                precision, recall, f1_score, support = \
                    precision_recall_fscore_support(yt_label, y_pred_label_ad,
                                                    average='binary', zero_division=1)  # 测试集各项指标
            mts_res[lab] = dict(acc=acc, precision=precision, recall=recall, f1=f1_score, rmse=rmse)
        setup_seed(rs)
        # run.finish()
        print(mts_res)
        return mts_res


def model_ablation(data_file: Union[str, os.PathLike], model_dir: Union[str, os.PathLike],
                   params_optimal: dict[str, Any], params_ablation: list[Tuple[str, dict[str, bool]]],
                   perf_comp_f: Union[str, os.PathLike], fit: bool = True, load_data: bool = True) -> pd.DataFrame:
    """
    模型消融比较
    :param data_file: 数据集文件
    :param model_dir: 模型路径
    :param params_optimal: 最优的模型参数配置，格式为{参数名: 参数值}
    :param params_ablation: 模型消融配置，格式为[(配置名, {参数名: 参数值})]
    :param perf_comp_f: 模型性能比较的csv文件
    :param fit: 是否进行训练
    :param load_data: 是否加载之前模型已评估好的结果直接获取指标数据
    :return: 各模型配置的结果
    """
    if load_data:
        df_res = pd.read_csv(perf_comp_f)
    else:
        hp_all = {'DEMENTIA': params_optimal}
        for i_pa in params_ablation:
            hp_ab = hp_optimal.copy()
            hp_ab.update(i_pa[-1])
            hp_all[i_pa[0]] = hp_ab
        conf_l, acc_tr_l, pre_tr_l, rec_tr_l, f1_tr_l, rmse_tr_l = [], [], [], [], [], []
        acc_l, pre_l, rec_l, f1_l, rmse_l = [], [], [], [], []
        for mn, hp in hp_all.items():
            print(f"------- Running {mn} model: {hp} -------\n")
            model = DementiaDetectionModel(data_file, params_config=hp, model_save_dir=model_dir, model_name=mn)
            res = model.model_train_evaluate(fit=fit)
            conf_l.append(mn)
            acc_tr_l.append(f"{res['mts_train']['acc'] * 100:.2f}")
            pre_tr_l.append(f"{res['mts_train']['precision'] * 100:.2f}")
            rec_tr_l.append(f"{res['mts_train']['recall'] * 100:.2f}")
            f1_tr_l.append(f"{res['mts_train']['f1'] * 100:.2f}")
            rmse_tr_l.append(f"{res['mts_train']['rmse']:.2f}")
            acc_l.append(f"{res['mts_test']['acc'] * 100:.2f}")
            pre_l.append(f"{res['mts_test']['precision'] * 100:.2f}")
            rec_l.append(f"{res['mts_test']['recall'] * 100:.2f}")
            f1_l.append(f"{res['mts_test']['f1'] * 100:.2f}")
            rmse_l.append(f"{res['mts_test']['rmse']:.2f}")
        df_res = pd.DataFrame({"Model Config": conf_l, "tr-Acc/%": acc_tr_l, "tr-Pre/%": pre_tr_l, "tr-Rec/%": rec_tr_l,
                               "tr-F1/%": f1_tr_l, "tr-RMSE": rmse_tr_l, "Acc/%": acc_l, "Pre/%": pre_l,
                               "Rec/%": rec_l, "F1/%": f1_l, "RMSE": rmse_l})
        df_res.to_csv(perf_comp_f, encoding="utf-8-sig", index=False)
    print(df_res)
    return df_res


def compare_with_sota(sota_comp: dict[str, list[dict[str, float]]], sota_comp_dir: Union[str, os.PathLike]):
    """
    与SOTA结果相比
    SOTA来源参考：Shi M, Cheung G, Shahamiri S R. Speech and language processing with deep learning for dementia
                diagnosis: A systematic review. Psychiatry Research, 2023, 329: 115538.
    :param sota_comp: SOTA结果（格式为{'Acc': [{'Wang et al., 2022': [93.75]},{'Yuan et al., 2020': [89.58]},
                    {'Liu et al., 2022': [87.50]}], ..., 'RMSE': [{'Koo et al., 2020': 3.74},{'Searle et al., 2020': 4.32},
                    {'Farzana & Parde, 2020': 4.34}]）
    :param sota_comp_dir: 比较结果绘图保存路径
    :return: None
    """
    clss = {k: v for k, v in sota_comp.items() if k != 'RMSE'}
    rmses = {k: v for d in sota_comp['RMSE'] for k, v in d.items()}
    fig, ax_cls = plt.subplots(constrained_layout=True, figsize=(9, 9))
    plt.xlim(0, len(list(clss.values())[0]) + len(rmses.keys()) + 2)
    ax_cls.set_ylabel('Acc/F1/Rec/Pre (%)', fontdict={'family': font_family, 'size': 18})
    mk = ['s', '^', 'X', '*', 'D'] * len(clss.keys())
    texts = []
    for i_index, (cls_k, cls_v) in enumerate(clss.items()):
        cls_d = {k: v for d in cls_v for k, v in d.items()}
        ax_cls.scatter(range(1, len(cls_d.values()) + 1), cls_d.values(), marker=mk[i_index],
                       c=sns.color_palette("Set2")[:len(cls_d.values())], s=160, label=cls_k)
        for ii in range(len(cls_d.values())):
            text = ax_cls.text(range(1, len(cls_d.values()) + 1)[ii]+.15,
                               list(cls_d.values())[ii]-np.random.uniform(0.09, 0.11),
                               f'{list(cls_d.values())[ii]}'.rstrip("0").rstrip("."), ha='left', va='center', c='k',
                               fontdict={'family': font_family, 'size': 18})
            texts.append(text)
    adjust_text(texts, only_move={'text': 'y'})
    ax_cls.text(2.5 / ax_cls.get_xlim()[-1], -0.2, 'Classification Task', ha='center', va='top', c='k',
                fontdict={'family': font_family, 'size': 20}, transform=ax_cls.transAxes)
    ax_cls.legend(loc="lower left", prop={'family': font_family, 'size': 16}, labelspacing=.5, handletextpad=.1)
    ax_cls.set_ylim(80, 98)
    ax_cls.tick_params(direction='in', color='k', length=5, width=1)
    ax_cls.set_xticklabels(ax_cls.get_xticklabels(), fontdict={'family': font_family, 'size': 18},
                           rotation=25, ha="right", rotation_mode="anchor")
    plt.yticks(fontsize=14, fontproperties=font_family)
    ax_cls.get_xticklines()[8].set_visible(False)

    ax_rmse = ax_cls.twinx()
    ax_rmse.set_ylabel('RMSE', fontdict={'family': font_family, 'size': 18})
    ax_rmse.set_prop_cycle(cycler(color=sns.color_palette("Set2")))
    bars_rmse = []
    for i_index, (i_name, i_rmse) in enumerate(rmses.items(), start=len(clss) + 2):
        bar = ax_rmse.bar(i_index, i_rmse, width=1, label=i_name)
        bars_rmse.append(bar)
        ax_rmse.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), f'{i_rmse:.2f}',
                     ha='center', va='bottom', c='k', fontdict={'family': font_family, 'size': 18})
    ax_rmse.text(bars_rmse[2][0].get_x() / ax_rmse.get_xlim()[-1], -0.2, 'Regression Task', ha='center', va='top', c='k',
                 fontdict={'family': font_family, 'size': 20}, transform=ax_rmse.transAxes)
    ax_rmse.set_ylim(3, 4.5)
    ax_rmse.tick_params(direction='in', color='k', length=5, width=1)
    ax_rmse.tick_params(axis='x', length=0)
    plt.yticks(fontsize=14, fontproperties=font_family)
    plt.xticks(range(1, len(list(clss.values())[0]) + len(rmses.keys()) + 2),
               [list(i.keys())[0] for i in list(clss.values())[0]] + [''] + list(rmses.keys()))
    for sp in plt.gca().spines:
        plt.gca().spines[sp].set_color('k')
        plt.gca().spines[sp].set_linewidth(1)
    plt.grid(False)
    fig_file = os.path.join(sota_comp_dir, f'sotaComp/sotaComp.png')
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


class Visualization:
    """
    可视化模型内部状态
    """

    def __init__(self, data_file: Union[str, os.PathLike], fig_save_dir: Union[str, os.PathLike]):
        """
        初始化
        :param data_file: 数据集文件
        :param fig_save_dir: 图片保存路径
        """
        data_file = os.path.normpath(data_file)
        if data_file.endswith('pkl'):
            feat_data = pd.read_pickle(data_file)  # type: pd.DataFrame
        else:
            raise ValueError('无效数据，仅接受.pkl数据集文件')
        feat_data = feat_data.sample(frac=1, random_state=rs).reset_index(drop=True)  # 打乱样本
        feat_data['mmse'].fillna(feat_data[(feat_data['set'] == 'train') & (feat_data['label'] == 0)]['mmse'].mean(),
                                 inplace=True)
        self.feat_data = feat_data
        bert = 'distilbert-base-uncased'
        self.data_audio = np.array(feat_data['mfcc_cmvn'].tolist(), dtype=np.float32)
        audio_mask = np.where(np.ma.masked_equal(self.data_audio, 0).mask, 0, 1)[:, :, 0]
        self.audio_mask = audio_mask[:, tf.newaxis]
        self.data_text = np.array(feat_data[bert].tolist(), dtype=np.float32)
        data_hand = np.array(feat_data['handcrafted'].tolist(), dtype=np.float32)
        self.hand_feat_name = ["F0 SD", "DPI", "Voiced Rate", "Hesitation Ratio", "EWF", "Word Rate",
                               "Function Word Ratio", "Lexical Density", "MLU", "Noun Phrase Rate",
                               "Verb Phrase Rate", "Parse Tree Height", "Yngve Depth Total", "Dependency Distance Total"]
        ss_hand = StandardScaler()
        self.data_hand = ss_hand.fit_transform(data_hand)
        self.label = np.array(feat_data['label'].tolist(), dtype=int)
        self.mmse = np.array(feat_data['mmse'].tolist(), dtype=np.float16)
        self.id = np.array(feat_data['id'].tolist(), dtype=str)
        self.custom_objects = {'Attention': Attention, 'scaled_sigmoid': scaled_sigmoid,
                               'ConcreteDenseDropout': ConcreteDenseDropout}
        self.fig_save_dir = fig_save_dir
        self.data_id = None

    def viz_tsne(self, model_file: Union[str, os.PathLike]):
        """
        将融合模型的最后全连接层之前的ReLU激活层输出作为特征，进行t-SNE降维并可视化
        :param model_file: 模型文件
        :return: None
        """
        from sklearn.manifold import TSNE
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型不存在，无法评估，请先训练")
        inputs = {"in_a": self.data_audio, "mask_a": self.audio_mask,
                  "in_t": self.data_text, "in_h": self.data_hand}
        model = load_model(model_file, custom_objects=self.custom_objects)
        model_tsne = Model(inputs=model.input, outputs=model.layers[-3].output)  # 最后全连接层之前的ReLU激活层
        intermed = model_tsne.predict(inputs)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate='auto', init='pca',
                    random_state=rs, n_jobs=-1)
        intermed_tsne = tsne.fit_transform(intermed)
        tx = intermed_tsne[:, 0]
        ty = intermed_tsne[:, 1]
        plt.figure(figsize=(6, 6), tight_layout=True)
        plt.xlabel('Dimension 1', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Dimension 2', fontdict={'family': font_family, 'size': 18})
        colors = sns.color_palette("Set2")[:len(np.unique(self.label))]
        for idx, c in enumerate(colors):
            indices = [i for i, l in enumerate(self.label) if idx == l]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            plt.scatter(current_tx, current_ty, label={0: 'HC', 1: 'AD'}[idx])
        plt.legend(loc="upper left", prop={'family': font_family, 'size': 18}, handlelength=1.0, handletextpad=0.3)
        plt.xticks(fontsize=12, fontproperties=font_family)
        plt.yticks(fontsize=12, fontproperties=font_family)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = os.path.join(self.fig_save_dir, f'tsne.png')
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

    def viz_audio(self, model_file: Union[str, os.PathLike], datasets_dir: Union[str, os.PathLike]):
        """
        可视化音频模态
        :param model_file: 模型文件
        :param datasets_dir: 包含的数据集文件的路径
        :return: None
        """
        import librosa.display
        from keract import get_activations
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型不存在，无法评估，请先训练")
        inputs = {"in_a": self.data_audio, "mask_a": self.audio_mask}
        model = load_model(model_file, custom_objects=self.custom_objects)
        model_out = model.predict(inputs)
        y_pred_proba_ad = model_out[-1].flatten()
        y_pred_label_ad = (y_pred_proba_ad > 0.5).astype("int32")
        id_cor = self.id[y_pred_label_ad == self.label]
        label_cor = self.label[y_pred_label_ad == self.label]
        audio = [glob.glob(os.path.join(datasets_dir, f'*/Full_wave_enhanced_audio/**/{i}.wav'), recursive=True)[0]
                 for i in id_cor]
        data_id = pd.DataFrame({'id': id_cor, 'label': label_cor, 'audio': audio})
        id_used = ['S056', 'S191']
        for sub in data_id.index:
            id = data_id.loc[sub, 'id']
            lab = data_id.loc[sub, 'label']
            if id not in id_used:
                continue
            wav_f = data_id.loc[sub, 'audio']
            print(f"Subject ID: {id}; True and Predicted Label: {lab}")
            wav_data, sr = librosa.load(wav_f, sr=None)
            audio_time = int(len(wav_data) / sr)
            inps = {"in_a": self.data_audio[np.argwhere(self.id == id).flatten()[0], ...][np.newaxis, :], 
                    "mask_a": self.audio_mask[np.argwhere(self.id == id).flatten()[0], ...][np.newaxis, :]}
            att_wg = np.mean(get_activations(model, inps, layer_names='mha_a')['mha_a'], axis=-1)
            mel_spec = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_fft=512, hop_length=341,
                                                      window="hamming", n_mels=26, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec)
            fig = plt.figure(figsize=(9, 3))
            ax_mel = fig.add_subplot(111, label="mel")
            ax_att = fig.add_subplot(111, label="att", frame_on=False)
            img = librosa.display.specshow(log_mel_spec, sr=sr, hop_length=341, x_axis="s", y_axis="mel",
                                           fmax=8000, ax=ax_mel)
            ax_mel.set_xlabel('Time (s)', fontdict={'family': font_family, 'size': 18})
            ax_mel.set_ylabel('Frequency (Hz)', fontdict={'family': font_family, 'size': 18})
            ax_mel.set_xlim(0, np.ceil(audio_time))
            ax_mel.set_ylim(0, ax_mel.get_ylim()[-1])
            ax_mel.tick_params(direction='in', color='k', length=3, width=1, labelsize=12)
            cax_mel = fig.add_axes([ax_mel.get_position().x1 + 0.01, ax_mel.get_position().y0, 0.015,
                                    ax_mel.get_position().height])
            cbar_mel = fig.colorbar(img, cax=cax_mel, format="%+02.0f")
            cbar_mel.ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # 最多显示5个刻度值
            cbar_mel.outline.set_visible(False)
            cbar_mel.ax.set_ylabel('Mel Spectrum (dB)', fontdict={'family': font_family, 'size': 16})
            cbar_mel.ax.tick_params(labelsize=10, length=3)
            for sp in ax_mel.spines:
                ax_mel.spines[sp].set_visible(True)
                ax_mel.spines[sp].set_color('k')
                ax_mel.spines[sp].set_linewidth(1)
            _ax_att = sns.heatmap(att_wg, cmap='GnBu', xticklabels=False, yticklabels=False, ax=ax_att, cbar=False, alpha=1)
            ax_att.xaxis.tick_top()
            ax_att.yaxis.tick_right()
            ax_att.xaxis.set_label_position('top')
            ax_att.yaxis.set_label_position('right')
            ax_att.set_xlim(0, att_wg.shape[-1])
            ax_att.set_ylim(-0.2, 8)
            cax_att = fig.add_axes([ax_mel.get_position().x0, ax_mel.get_position().y1 + 0.01, 
                                    ax_mel.get_position().width, 0.05])
            cbar_att = fig.colorbar(_ax_att.collections[0], cax=cax_att, format="%.6f", 
                                    orientation='horizontal', ticklocation='top')
            cbar_att.ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            cbar_att.outline.set_visible(False)
            cbar_att.ax.set_xlabel('Attention Score', fontdict={'family': font_family, 'size': 16})
            cbar_att.ax.tick_params(labelsize=10, length=3)
            fig_file = os.path.join(self.fig_save_dir, f'audio/{id}_{lab}.png')
            if not os.path.exists(os.path.dirname(fig_file)):
                os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
            plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
            plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02)
            plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.show()
            plt.close('all')
            
    def viz_text(self, model_file: Union[str, os.PathLike], trans_file_dir: Union[str, os.PathLike]):
        """
        可视化文本模态
        :param model_file: 模型文件
        :param trans_file_dir: 转录文本csv文件路径
        :return: None
        """
        from transformers import TFDistilBertModel, DistilBertTokenizer
        from lime.lime_text import LimeTextExplainer
        import re
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型不存在，无法评估，请先训练")
        inputs = {"in_t": self.data_text}
        model = load_model(model_file, custom_objects=self.custom_objects)
        model_out = model.predict(inputs)
        y_pred_proba_ad = model_out[-1].flatten()
        y_pred_label_ad = (y_pred_proba_ad > 0.5).astype("int32")
        id_cor = self.id[y_pred_label_ad == self.label]
        label_cor = self.label[y_pred_label_ad == self.label]
        text_all = pd.read_csv(trans_file_dir)
        _text = text_all[text_all['id'].isin(id_cor)][['id', 'joined_par_speech']]
        _text['id'] = pd.Categorical(_text['id'], categories=id_cor, ordered=True)
        text = _text.sort_values(by='id')['joined_par_speech'].tolist()
        data_id = pd.DataFrame({'id': id_cor, 'label': label_cor, 'text': text})
        
        def predictor(tt: str):
            tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(BERT_MODEL_PATH, 'distilbert-base-uncased'))
            bert_model = TFDistilBertModel.from_pretrained(os.path.join(BERT_MODEL_PATH, 'distilbert-base-uncased'))
            encoded_input = tokenizer(tt, max_length=512, padding='max_length', truncation=True, return_tensors='tf')
            last_hidden_states = bert_model(encoded_input)[0]
            features = last_hidden_states[:, 1:-1, :].numpy()
            _model_out = model.predict(features)
            probas = _model_out[-1]
            probas = np.hstack([1 - probas, probas])
            return probas
        
        def colorize(texts, tw):
            words = re.split(r'([ &\'@]+)', texts)
            wg_factor = 0.9 / np.abs(tw[0][-1])
            str_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        margin: 0;
                        padding: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        background-color: #ffffff;
                    }
                    .content {
                        background-color: #ffffff;
                        padding: 50px;
                    }
                </style>
            </head>
            <body>
                <div class="content">
                    <p style="text-align: justify;">
                        <span style="font-family: 'times new roman', times, serif; font-size: 18pt;">
            """
            for wd in words:
                fill, alpha = False, '#ffffff'
                for tt, wg in tw:
                    if wd == tt:
                        fill, alpha = True, wg
                if fill:
                    if alpha >= 0:
                        cmap = matplotlib.colormaps['Oranges']
                    else:
                        cmap = matplotlib.colormaps['Blues']
                    color = rgb2hex(cmap(abs(alpha) * wg_factor)[:3])
                    str_html += f"""<span class="barcode"; style="color: black; background-color: {color}">{wd}</span>\n"""
                else:
                    if wd != ' ':
                        wd = wd.replace('<', '&lt;').replace('>', '&gt;')
                        str_html += f"""<span class="barcode"; style="color: black; background-color: {alpha}">{wd}</span>\n"""
            str_html += "\n</span></p></div></body></html>"
            return str_html
        
        id_used = ['S056', 'S191']
        for sub in data_id.index:
            id = data_id.loc[sub, 'id']
            lab = data_id.loc[sub, 'label']
            if id not in id_used:
                continue
            txt_exp = data_id.loc[sub, 'text']
            print(f"Subject ID: {id}; True and Predicted Label: {lab}")
            explainer = LimeTextExplainer(class_names=['健康对照', 'AD'], random_state=rs)
            exp = explainer.explain_instance(txt_exp, predictor, num_features=1000, num_samples=200)
            tw_paired = exp.as_list()
            pw_paired = exp.local_exp[exp.available_labels()[0]]
            # 文本背景颜色为岭回归系数，即权重向量，越冷色调代表该单词对健康对照越重要，越暖色调代表该单词对AD越重要
            html_str = colorize(txt_exp, tw_paired)
            max_wg = max(tw_paired, key=lambda x: x[1])[1]
            min_wg = min(tw_paired, key=lambda x: x[1])[1]
            cmap_max = matplotlib.colormaps['Oranges' if max_wg >= 0 else 'Blues']
            color_max = rgb2hex(cmap_max(abs(max_wg) * 0.9 / np.abs(tw_paired[0][-1]))[:3])
            cmap_min = matplotlib.colormaps['Oranges' if min_wg >= 0 else 'Blues']
            color_min = rgb2hex(cmap_min(abs(min_wg) * 0.9 / np.abs(tw_paired[0][-1]))[:3])
            fig, ax = plt.subplots(constrained_layout=True, figsize=(11, 1))
            sm = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('custom_cmap', [color_min, color_max]),
                                       norm=plt.Normalize(vmin=min_wg, vmax=max_wg))
            cbar = plt.colorbar(sm, cax=ax, orientation='horizontal', ticklocation='top', format="%.6f")
            cbar.set_ticks([min_wg, 0, max_wg] if min_wg < 0 < max_wg else [min_wg, max_wg])
            cbar.ax.set_xticklabels([f'{min_wg:.6f}', '0', f'{max_wg:.6f}'] if min_wg < 0 < max_wg else
                                    [f'{min_wg:.6f}', f'{max_wg:.6f}'])
            cbar.outline.set_visible(False)
            cbar.set_label('Importance Weight', fontdict={'family': font_family, 'size': 24})
            cbar.ax.tick_params(labelsize=16, length=6)
            fig_file = os.path.join(self.fig_save_dir, f'text/{id}_{lab}-colorbar.png')
            if not os.path.exists(os.path.dirname(fig_file)):
                os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02, transparent=True)
            # plt.show()
            plt.close()
            fig_file = os.path.join(self.fig_save_dir, f'text/{id}_{lab}.html')
            with open(fig_file, 'w') as f:
                f.write(html_str)
            exp.save_to_file(fig_file.replace('.html', '_lime.html'))

    def viz_handcraft(self, model_file: Union[str, os.PathLike]):
        """
        利用SHAP对手工特征模型进行解释分析
        :param model_file: 模型文件
        :return: None
        """
        import shap
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型不存在，无法评估，请先训练")
        model = load_model(model_file, custom_objects=self.custom_objects)
        model_reg = Model(inputs=model.input, outputs=model.layers[-2].output)
        explainer = shap.DeepExplainer(model_reg, self.data_hand)
        shap_values = explainer.shap_values(self.data_hand)[:, :, 0]
        plt.figure(figsize=(8, 6), tight_layout=True)
        shap.summary_plot(shap_values, self.data_hand, max_display=20, feature_names=self.hand_feat_name,
                          axis_color='black', show=False, color_bar=False)
        plt.xlabel('SHAP value (impact on model output)', fontdict={'family': font_family, 'size': 18})
        plt.xticks(fontsize=12, fontproperties=font_family)
        plt.yticks(fontsize=18, fontproperties=font_family)
        plt.axvline(x=0, color='black', lw=1.5)
        m = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue)
        m.set_array(np.array([0, 1]))
        cb = plt.colorbar(m, ticks=[0, 1])
        cb.ax.tick_params(labelsize=14, length=0)
        cb.set_ticklabels(['Low', 'High'])
        cb.set_label('Normalized Feature Value', labelpad=0, fontdict={'family': font_family, 'size': 16})
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = os.path.join(self.fig_save_dir, f'handcraft/handcraft.png')
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


def eval_pitt(data_file: Union[str, os.PathLike], model_file: Union[str, os.PathLike]):
    """
    评估Pitt数据集
    DEMENTIA模型：{'acc': 0.8081264108352144, 'precision': 0.8978723404255319, 
                  'recall': 0.7757352941176471, 'f1': 0.8323471400394478, 'rmse': 4.377868}
    :param data_file: 数据集文件
    :param model_file: 模型文件
    """
    data_file = os.path.normpath(data_file)
    if data_file.endswith('pkl'):
        feat_data = pd.read_pickle(data_file)  # type: pd.DataFrame
    else:
        raise ValueError('无效数据，仅接受.pkl数据集文件')
    feat_data = feat_data.sample(frac=1, random_state=rs).reset_index(drop=True)  # 打乱样本
    bert = 'distilbert-base-uncased'
    data_audio = np.array(feat_data['mfcc_cmvn'].tolist(), dtype=np.float32)
    audio_mask = np.where(np.ma.masked_equal(data_audio, 0).mask, 0, 1)[:, :, 0]
    audio_mask = audio_mask[:, tf.newaxis]
    data_text = np.array(feat_data[bert].tolist(), dtype=np.float32)
    _data_hand = np.array(feat_data['handcrafted'].tolist(), dtype=np.float32)
    ss_hand = StandardScaler()
    data_hand = ss_hand.fit_transform(_data_hand)
    label = np.array(feat_data['label'].tolist(), dtype=int)
    mmse = np.array(feat_data['mmse'].tolist(), dtype=np.float16)
    custom_objects = {'Attention': Attention, 'scaled_sigmoid': scaled_sigmoid,
                      'ConcreteDenseDropout': ConcreteDenseDropout}
    if not os.path.exists(model_file):
        raise FileNotFoundError("模型不存在，无法评估，请先训练")
    inputs = {"in_a": data_audio, "mask_a": audio_mask, "in_t": data_text, "in_h": data_hand}
    model = load_model(model_file, custom_objects=custom_objects)
    model_out = model.predict(inputs)
    y_pred_mmse = model_out[0].flatten()
    y_pred_proba_ad = model_out[-1].flatten()
    y_pred_label_ad = (y_pred_proba_ad > 0.5).astype("int32")
    acc = accuracy_score(label, y_pred_label_ad)
    precision, recall, f1_score, support = precision_recall_fscore_support(label, y_pred_label_ad,
                                                                           average='binary', zero_division=1)
    rmse = mean_squared_error(mmse, y_pred_mmse, squared=False)
    mts_res = dict(acc=acc, precision=precision, recall=recall, f1=f1_score, rmse=rmse)
    print(mts_res)
    return mts_res
        

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    feat_all_f = os.path.join(current_path, r'data/feats.pkl')
    feat_all_f_pitt = os.path.join(current_path, r'data/feats_pitt.pkl')
    model_path = os.path.join(current_path, r'models')
    res_path = os.path.join(current_path, r'results')
    data_path = os.path.join(current_path, r"data")

    hp_optimal = {'bert': 'distilbert-base-uncased', 'mha_audio': True, 'mha_cross': True, 'att_global': True,
                  'sig_reg': False, 'sig_cls': False, 'audio': True, 'text': True, 'handcraft': True,
                  'lw_cls': 2, 'lr': 0.003, 'batch_size': 16, 'epochs': 100}
    # DEMENTIA模型训练与评估
    dem_model = DementiaDetectionModel(feat_all_f, params_config=hp_optimal,
                                       model_save_dir=model_path, model_name='DEMENTIA')
    dem_model.model_train_evaluate(fit=False)

    # 模型消融实验
    hp_ablation = [('NO-mha_audio', {'mha_audio': False}), ('NO-mha_cross', {'mha_cross': False}),
                   ('NO-att_global', {'att_global': False}),
                   ('NO-attention', {'mha_audio': False, 'mha_cross': False, 'att_global': False}),
                   ('ONLY-mha_audio', {'mha_cross': False, 'att_global': False}),
                   ('ONLY-mha_cross', {'mha_audio': False, 'att_global': False}),
                   ('ONLY-att_global', {'mha_audio': False, 'mha_cross': False}),
                   ('ONLY-reg', {'sig_reg': True}), ('ONLY-cls', {'sig_cls': True}),
                   ('NO-audio', {'audio': False}), ('NO-text', {'text': False}), ('NO-handcraft', {'handcraft': False}),
                   ('ONLY-audio', {'text': False, 'handcraft': False}),
                   ('ONLY-text', {'audio': False, 'handcraft': False}),
                   ('ONLY-handcraft', {'audio': False, 'text': False})]
    res_ab = model_ablation(feat_all_f, model_path, hp_optimal, hp_ablation, fit=False, load_data=True,
                            perf_comp_f=os.path.join(current_path, r'results/model_ablation.csv'))

    # 模型与当前最先进结果比较
    dem_res = res_ab.loc[res_ab['Model Config'] == 'DEMENTIA',
                         ['Acc/%', 'F1/%', 'Rec/%', 'Pre/%', 'RMSE']].to_dict(orient='records')[0]
    comp_res = {'Acc': [{'Ours': dem_res['Acc/%']}, {'Wang et al., 2022': 93.75}, {'Yuan et al., 2020': 89.58},
                        {'Liu et al., 2022': 87.50}],
                'F1': [{'Ours': dem_res['F1/%']}, {'Wang et al., 2022': 93.9}, {'Yuan et al., 2020': 88.9},
                       {'Liu et al., 2022': 87}],
                'Rec': [{'Ours': dem_res['Rec/%']}, {'Wang et al., 2022': 95.8}, {'Yuan et al., 2020': 83.3},
                        {'Liu et al., 2022': 88}],
                'Pre': [{'Ours': dem_res['Pre/%']}, {'Wang et al., 2022': 92}, {'Yuan et al., 2020': 95.2},
                        {'Liu et al., 2022': 88}],
                'RMSE': [{'Ours': dem_res['RMSE']}, {'Koo et al., 2020': 3.74}, {'Searle et al., 2020': 4.32},
                         {'Farzana et al., 2020': 4.34}]}
    compare_with_sota(sota_comp=comp_res, sota_comp_dir=res_path)

    # 模型内部可视化：可解释性
    for mn in ['DEMENTIA', 'ONLY-audio', 'ONLY-text', 'ONLY-handcraft']:
        viz = Visualization(feat_all_f, os.path.join(res_path, f'viz/afterFusion/tsne_{mn}'))
        viz.viz_tsne(os.path.join(model_path, f'{mn}/{mn}.h5'))
    viz = Visualization(feat_all_f, os.path.join(res_path, f'viz/multimodality'))
    viz.viz_text(os.path.join(model_path, 'ONLY-text/ONLY-text.h5'), os.path.join(data_path, 'trans.csv'))
    viz.viz_audio(os.path.join(model_path, 'ONLY-audio/ONLY-audio.h5'), DATA_PATH)
    viz.viz_handcraft(os.path.join(model_path, 'ONLY-handcraft/ONLY-handcraft.h5'))

    # 评估Pitt数据集
    eval_pitt(feat_all_f_pitt, os.path.join(model_path, 'DEMENTIA/DEMENTIA.h5'))

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
