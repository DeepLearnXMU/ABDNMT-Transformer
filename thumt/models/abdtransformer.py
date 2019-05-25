# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

import thumt.interface as interface
import thumt.layers as layers
import thumt.utils.getter as getter


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None, r2l_memory=None, r2l_bias=None,reuse=None):
    with tf.variable_scope(scope, dtype=dtype,
                           values=[inputs, memory, bias, mem_bias, r2l_memory, r2l_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                    if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention",reuse=reuse):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention",reuse=reuse):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                if r2l_memory is not None:
                    with tf.variable_scope("r2l_attention"):
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            r2l_memory,
                            r2l_bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward",reuse=reuse):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params)

    return encoder_output


def r2l_decoder(decoder_input, encoder_output, dec_attn_bias, enc_attn_bias, params):
    scope='r2l'
    if params.share_r2l:
        scope='decoder'
    r2l_state = transformer_decoder(decoder_input, encoder_output, dec_attn_bias, enc_attn_bias, params, scope=scope)
    return r2l_state


def l2r_decoder(r2l_mem, r2l_attn_bias, decoder_input, encoder_output, dec_attn_bias, enc_attn_bias, params, state=None):
    scope='l2r'
    if params.share_r2l:
        scope='decoder'
    outputs = transformer_decoder(decoder_input, encoder_output, dec_attn_bias, enc_attn_bias, params, state,
                                  scope=scope, r2l_memory=r2l_mem, r2l_bias=r2l_attn_bias,reuse=params.share_r2l)
    return outputs


def abd_forward(r2l_input, r2l_bias, l2r_input, l2r_bias, encoder_output, enc_attn_bias,r2l_attn_bias, params, state=None):
    if state is None or 'r2l_memory' not in state:
        r2l_mem = r2l_decoder(r2l_input, encoder_output, r2l_bias, enc_attn_bias, params)
        if state is not None:
            state['r2l_memory'] = r2l_mem
    else:
        r2l_mem = state['r2l_memory']

    l2r_outputs = l2r_decoder(r2l_mem, r2l_attn_bias, l2r_input, encoder_output, l2r_bias, enc_attn_bias, params, state)
    return r2l_mem, l2r_outputs


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    r2l_tgt_seq = features["r2l_target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    r2l_tgt_len = features["r2l_target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)
    r2l_tgt_mask = tf.sequence_mask(r2l_tgt_len,
                                    maxlen=tf.shape(features["r2l_target"])[1],
                                    dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)
    r2l_targets = tf.gather(tgt_embedding, r2l_tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)
        r2l_targets = r2l_targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)
    r2l_targets = r2l_targets * tf.expand_dims(r2l_tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    r2l_bias = layers.attention.attention_bias(tf.shape(r2l_targets)[1],
                                                        "causal", dtype=dtype)
    r2l_attn_bias = layers.attention.attention_bias(r2l_tgt_mask, "masking",
                                                    dtype=dtype)

    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    r2l_decoder_input = tf.pad(r2l_targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)
        r2l_decoder_input = layers.attention.add_timing_signal(r2l_decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)
        r2l_decoder_input = tf.nn.dropout(r2l_decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode == "train":
        r2l_output, l2r_output = abd_forward(r2l_decoder_input, r2l_bias, decoder_input, dec_attn_bias,
                                             encoder_output, enc_attn_bias,r2l_attn_bias, params)

    elif mode == 'infer':
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        r2l_mem, (l2r_output, decoder_state) = abd_forward(r2l_decoder_input, r2l_bias,
                                                     decoder_input, dec_attn_bias,
                                                     encoder_output, enc_attn_bias,
                                                           r2l_attn_bias,
                                                           params, state['decoder'])
        decoder_state['r2l_memory']=r2l_mem
        l2r_output = l2r_output[:, -1, :]
        logits = tf.matmul(l2r_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state}
    else:
        raise NotImplementedError('mode=%s' % mode)

    def loss(decoder_output, labels, mask):
        decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
        logits = tf.matmul(decoder_output, weights, False, True)
        # label smoothing
        ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )
        tgt_mask = tf.cast(mask, ce.dtype)

        ce = tf.reshape(ce, tf.shape(tgt_mask))

        loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
        return loss

    r2l_loss = loss(r2l_output, features['r2l_target'], r2l_tgt_mask)
    l2r_loss = loss(l2r_output, features['target'], tgt_mask)

    return l2r_loss + r2l_loss


def model_graph(features, mode, params):
    encoder_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            if dtype != tf.float32:
                custom_getter = getter.fp32_variable_getter
            else:
                custom_getter = None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
                decoding_graph(features, state, 'infer', params)  # get r2l_memory
                r2l_memory = state['decoder']['r2l_memory']
                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
                state['decoder']['r2l_memory'] = r2l_memory

            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, reuse=True):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=True,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            position_info_type='relative',  # 'absolute' or 'relative'
            max_relative_dis=16,  # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            share_r2l=None,
        )

        return params
