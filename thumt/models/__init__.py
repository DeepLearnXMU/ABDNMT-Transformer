# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.rnnsearch_lrp
import thumt.models.transformer
import thumt.models.transformer_lrp
import thumt.models.abdtransformer
import thumt.models.abd2transformer

def get_model(name, lrp=False):
    name = name.lower()

    if name == "rnnsearch":
        if not lrp:
            return thumt.models.rnnsearch.RNNsearch
        else:
            return thumt.models.rnnsearch_lrp.RNNsearch_lrp
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name =='abd':
        return thumt.models.abdtransformer.Transformer
    elif name =='abd2':
        return thumt.models.abd2transformer.Transformer
    elif name == "transformer":
        if not lrp:
            return thumt.models.transformer.Transformer
        else:
            return thumt.models.transformer_lrp.Transformer_lrp
    else:
        raise LookupError("Unknown model %s" % name)
