from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, AveragePooling2D, BatchNormalization, TimeDistributed, LSTM, Conv1D, Bidirectional, ELU, concatenate

from constants import EMOTIONS, GENDERS


def get_model_1(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(c1)
    mp2 = MaxPooling2D(strides=3)(c2)

    c3 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp2)

    c2 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(c3)
    ap1 = AveragePooling2D(pool_size=2)(c2)

    f1 = Flatten()(ap1)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_1')

    return model


def get_model_2(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(bn1)
    mp2 = MaxPooling2D(strides=3)(c2)

    c3 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp2)

    c2 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(c3)
    ap1 = AveragePooling2D(pool_size=2)(c2)

    f1 = Flatten()(ap1)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_2')

    return model


def get_model_3(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(bn1)
    mp2 = MaxPooling2D(strides=3)(c2)

    c3 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp2)

    c2 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(c3)
    ap1 = AveragePooling2D(pool_size=2)(c2)

    f1 = TimeDistributed(Flatten())(ap1)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_3')

    return model


def get_model_4(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=3)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_4')

    return model


def get_model_5(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=3)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm = LSTM(16, return_sequences=True)(f1)

    c2 = Conv1D(3, kernel_size=4, padding='same', activation='relu')(lstm)

    f1 = Flatten()(c2)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_5')

    return model


def get_model_6(input_layer, model_name_prefix=''):
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=3)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    bi_lstm = Bidirectional(LSTM(16, return_sequences=True))(f1)

    c2 = Conv1D(3, kernel_size=4, padding='same', activation='relu')(bi_lstm)

    f1 = Flatten()(c2)

    # emotion part
    emo_d1 = Dense(512, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_6')

    return model


def get_model_7(input_layer, model_name_prefix=''):
    bn1 = BatchNormalization()(input_layer)
    c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(bn1)
    mp1 = MaxPooling2D(strides=3)(c1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_7')

    return model


def get_model_8(input_layer, model_name_prefix=''):
    c1 = Conv2D(32, kernel_size=8, padding='same', activation='relu')(input_layer)
    mp1 = MaxPooling2D(strides=7)(c1)

    c2 = Conv2D(16, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=5)(c2)

    c3 = Conv2D(4, kernel_size=2, padding='same', activation='relu')(mp2)
    mp3 = MaxPooling2D(strides=3)(c3)

    f1 = TimeDistributed(Flatten())(mp3)

    f2 = Flatten()(f1)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f2)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_8')

    return model


def get_model_9(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9')

    return model


def get_model_9_1(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(196, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9_1')

    return model


def get_model_9_2(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=4, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(320, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9_2')

    return model


def get_model_9_3(input_layer, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=4, padding='same', activation='relu')(mp1)
    mp2 = AveragePooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(320, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9_3')

    return model


def get_model_10(input_layer, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(5, kernel_size=4, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    c3 = Conv2D(3, kernel_size=2, padding='same', activation='relu')(mp2)
    mp3 = MaxPooling2D(strides=2)(c3)

    f1 = TimeDistributed(Flatten())(mp3)
    lstm = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(320, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_10')

    return model


def get_model_11(input_layer, model_name_prefix=''):
    c1 = Conv2D(9, kernel_size=16, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=4)(bn1)

    c2 = Conv2D(7, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    c3 = Conv2D(5, kernel_size=4, padding='same', activation='relu')(mp2)
    ap1 = AveragePooling2D(strides=2)(c3)

    c4 = Conv2D(3, kernel_size=2, padding='same', activation='relu')(ap1)
    mp3 = MaxPooling2D(strides=2)(c4)

    f1 = TimeDistributed(Flatten())(mp3)
    lstm1 = LSTM(32, return_sequences=True)(f1)
    lstm2 = LSTM(16, return_sequences=True)(lstm1)

    f1 = Flatten()(lstm2)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_11')

    return model


def get_model_12(input_layer, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=16, padding='same', activation='elu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=4)(bn1)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm1 = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm1)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_12')

    return model


def get_model_12_1(input_layer, model_name_prefix=''):
    c1 = Conv2D(7, kernel_size=16, padding='same')(input_layer)
    bn1 = BatchNormalization()(c1)
    elu = ELU()(bn1)
    mp1 = MaxPooling2D(strides=4)(elu)

    f1 = TimeDistributed(Flatten())(mp1)
    lstm1 = LSTM(32, return_sequences=True)(f1)

    f1 = Flatten()(lstm1)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_12_1')

    return model


def get_model_13(input_layer, model_name_prefix=''):
    def _flb(input, c, c_k, c_s, p_s):
        conv = Conv2D(c, kernel_size=c_k, strides=c_s, padding='same')(input)
        bn = BatchNormalization()(conv)
        elu = ELU()(bn)
        mp = MaxPooling2D(strides=p_s)(elu)
        return mp

    flb1 = _flb(input_layer, 64, 3, 1, 2)

    flb2 = _flb(flb1, 64, 3, 1, 4)

    flb3 = _flb(flb2, 128, 3, 1, 4)

    # flb4 = _flb(flb3, 128, 3, 1, 4)

    f1 = TimeDistributed(Flatten())(flb3)
    lstm1 = LSTM(256, return_sequences=True)(f1)

    f1 = Flatten()(lstm1)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)

    emo_d2 = Dense(128, activation='relu')(emo_d1)

    emo_dr1 = Dropout(0.3)(emo_d2)

    d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_13')

    return model


# ****************************** MULTI OUTPUT MODELS ******************************


def get_model_9_multi(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f2 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(128, activation='relu')(f2)
    emo_dr1 = Dropout(0.3)(emo_d1)
    emo_d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

    # gender part
    gen_d1 = Dense(128, activation='relu')(f2)
    gen_dr1 = Dropout(0.1)(gen_d1)
    gen_d2 = Dense(64, activation='relu')(gen_dr1)
    gen_d_out = Dense(len(GENDERS), activation='softmax', name='gender_output')(gen_d2)

    model = Model(inputs=input_layer, outputs=[emo_d_out, gen_d_out], name=model_name_prefix + '_model_9_multi')

    return model


def get_model_14_multi(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f2 = Flatten()(lstm)

    # gender part
    gen_d1 = Dense(128, activation='relu')(f2)
    gen_dr1 = Dropout(0.1)(gen_d1)
    gen_d2 = Dense(64, activation='relu')(gen_dr1)
    gen_d_out = Dense(len(GENDERS), activation='softmax', name='gender_output')(gen_d2)

    # emotion part
    emo_d1 = Dense(128, activation='relu')(f2)
    emo_dr1 = Dropout(0.3)(emo_d1)

    concat = concatenate([emo_dr1, gen_d_out])

    emo_d2 = Dense(64, activation='relu')(concat)

    emo_d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_d2)

    model = Model(inputs=input_layer, outputs=[emo_d_out, gen_d_out], name=model_name_prefix + '_model_14_multi')

    return model


def get_model_14_1_multi(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)
    mp1 = MaxPooling2D(strides=2)(bn1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
    mp2 = MaxPooling2D(strides=2)(c2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f2 = Flatten()(lstm)

    # gender part
    gen_d1 = Dense(128, activation='relu')(f2)
    gen_dr1 = Dropout(0.1)(gen_d1)
    gen_d_out = Dense(len(GENDERS), activation='softmax', name='gender_output')(gen_dr1)

    # emotion part
    emo_d1 = Dense(128, activation='relu')(f2)
    emo_dr1 = Dropout(0.1)(emo_d1)

    concat = concatenate([emo_dr1, gen_d_out])

    emo_d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(concat)

    model = Model(inputs=input_layer, outputs=[emo_d_out, gen_d_out], name=model_name_prefix + '_model_14_1_multi')

    return model


def get_model_14_2_multi(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same')(input_layer)
    bn1 = BatchNormalization()(c1)
    elu1 = ELU()(bn1)
    mp1 = MaxPooling2D(strides=2)(elu1)

    c2 = Conv2D(3, kernel_size=8, padding='same')(mp1)
    elu2 = ELU()(c2)
    mp2 = MaxPooling2D(strides=2)(elu2)

    f1 = TimeDistributed(Flatten())(mp2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f2 = Flatten()(lstm)

    # gender part
    gen_d1 = Dense(128, activation='relu')(f2)
    gen_dr1 = Dropout(0.1)(gen_d1)
    gen_d_out = Dense(len(GENDERS), activation='softmax', name='gender_output')(gen_dr1)

    # emotion part
    emo_d1 = Dense(128, activation='relu')(f2)
    emo_dr1 = Dropout(0.1)(emo_d1)

    concat = concatenate([emo_dr1, gen_d_out])

    emo_d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(concat)

    model = Model(inputs=input_layer, outputs=[emo_d_out, gen_d_out], name=model_name_prefix + '_model_14_2_multi')

    return model


# ****************************** RL MODELS ******************************


def get_model_9_rl(input_layer, model_name_prefix=''):
    c1 = Conv2D(5, kernel_size=8, padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(c1)

    c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(bn1)

    f1 = TimeDistributed(Flatten())(c2)
    lstm = LSTM(16, return_sequences=True)(f1)

    f1 = Flatten()(lstm)

    # emotion part
    emo_d1 = Dense(256, activation='relu')(f1)
    emo_dr1 = Dropout(0.3)(emo_d1)

    d_out = Dense(len(EMOTIONS), activation='linear', name='emotion_output')(emo_dr1)

    model = Model(inputs=input_layer, outputs=[d_out], name=model_name_prefix + '_model_9')

    return model
