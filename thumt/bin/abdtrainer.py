#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import six

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import tensorflow as tf
import glob
import shutil

tf.gfile.Open = open
tf.gfile.GFile = open
tf.gfile.Exists = os.path.exists
tf.gfile.MakeDirs = os.makedirs
tf.gfile.Glob = glob.glob


def copy(oldpath, newpath, overwrite=False):
    if os.path.exists(newpath) and not overwrite:
        raise RuntimeError('destination exists:%s' % newpath)
    shutil.copy(oldpath, newpath)


tf.gfile.Copy = copy
tf.gfile.Remove = os.remove

import numpy as np
import thumt.data.cache as cache
import thumt.data.dataset as dataset
import thumt.data.record as record
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.distribute as distribute
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimize as optimize
import thumt.utils.parallel as parallel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=3,
                        help="Path of source and target corpus")
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str, nargs=2,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 training")
    parser.add_argument("--distribute", action="store_true",
                        help="Enable distributed training")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", "", ""],
        output="",
        record="",
        model="transformer",
        vocab=["", ""],
        # Default training hyper parameters
        num_threads=6,
        batch_size=4096,
        max_length=256,
        length_multiplier=1,
        mantissa_bits=2,
        warmup_steps=4000,
        train_steps=100000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        update_cycle=1,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        loss_scale=128,
        scale_l1=0.0,
        scale_l2=0.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=1,
        keep_top_checkpoint_max=1,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        validation=["", ""],
        references=[""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.record = args.record or params.record
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocab[0]),
        "target": vocabulary.load_vocabulary(params.vocab[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)

    if distribute.is_distributed_training_mode():
        config.gpu_options.visible_device_list = str(distribute.local_rank())
    elif params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config

def use_all_devices(params):
    from tensorflow.python.client import device_lib
    n_gpus = sum(1 for d in device_lib.list_local_devices() if d.device_type == 'GPU')
    if n_gpus > 1:
        params.device_list = list(range(n_gpus))


def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded


def restore_variables(checkpoint):
    if not checkpoint:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return tf.group(*ops, name="restore_op")


def print_variables():
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)


def main(args):
    if args.distribute:
        distribute.enable_distributed_training()

    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    if not args.distribute or distribute.rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(
            params.output,
            "%s.json" % args.model,
            collect_params(params, model_cls.get_parameters())
        )

    assert 'r2l' in params.input[2]
    # Build Graph
    use_all_devices(params)
    with tf.Graph().as_default():
        if not params.record:
            # Build input queue
            features = dataset.abd_get_training_input(params.input, params)
        else:
            features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )

        update_cycle = params.update_cycle
        features, init_op = cache.cache_features(features, update_cycle)

        # Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        model = model_cls(params)
        # Create global step
        global_step = tf.train.get_or_create_global_step()
        dtype = tf.float16 if args.fp16 else None

        if args.distribute:
            training_func = model.get_training_func(initializer, regularizer,
                                                    dtype)
            loss = training_func(features)
        else:
            # Multi-GPU setting
            sharded_losses = parallel.parallel_model(
                model.get_training_func(initializer, regularizer, dtype),
                features,
                params.device_list
            )
            loss = tf.add_n(sharded_losses) / len(sharded_losses)
            loss = loss + tf.losses.get_regularization_loss()

        # Print parameters
        if not args.distribute or distribute.rank() == 0:
            print_variables()

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        loss, ops = optimize.create_train_op(
            loss, opt, global_step,
            distribute.all_reduce if args.distribute else None, args.fp16,
            params)
        restore_op = restore_variables(args.checkpoint)

        # Validation
        if params.validation and params.references[0]:
            files = params.validation + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.abd_get_evaluation_input
        else:
            eval_input_fn = None

        # Add hooks
        multiplier = tf.convert_to_tensor([update_cycle, 1])

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "source": tf.shape(features["source"]) * multiplier,
                    "target": tf.shape(features["target"]) * multiplier
                },
                every_n_iter=1
            )
        ]

        if args.distribute:
            train_hooks.append(distribute.get_broadcast_hook())

        config = session_config(params)

        if not args.distribute or distribute.rank() == 0:
            # Add hooks
            save_vars = tf.trainable_variables() + [global_step]
            saver = tf.train.Saver(
                var_list=save_vars if params.only_save_trainable else None,
                max_to_keep=params.keep_checkpoint_max,
                sharded=False
            )
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            train_hooks.append(
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=params.output,
                    save_secs=params.save_checkpoint_secs or None,
                    save_steps=params.save_checkpoint_steps or None,
                    saver=saver
                )
            )

        if eval_input_fn is not None:
            if not args.distribute or distribute.rank() == 0:
                train_hooks.append(
                    hooks.EvaluationHook(
                        lambda f: inference.create_inference_graph(
                            [model], f, params
                        ),
                        lambda: eval_input_fn(eval_inputs, params),
                        lambda x: decode_target_ids(x, params),
                        params.output,
                        config,
                        params.keep_top_checkpoint_max,
                        eval_secs=params.eval_secs,
                        eval_steps=params.eval_steps
                    )
                )

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        def step_fn(step_context):
            # Bypass hook calls
            step_context.session.run([init_op, ops["zero_op"]])
            for i in range(update_cycle - 1):
                step_context.session.run(ops["collect_op"])

            return step_context.run_with_hooks(ops["train_op"])

        # Create session, do not use default CheckpointSaverHook
        if not args.distribute or distribute.rank() == 0:
            checkpoint_dir = params.output
        else:
            checkpoint_dir = None

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)

            while not sess.should_stop():
                sess.run_step_fn(step_fn)


if __name__ == "__main__":
    main(parse_args())