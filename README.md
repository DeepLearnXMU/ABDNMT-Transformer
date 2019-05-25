#### Introduction

Zhang, Xiangwen, Jinsong Su, Yue Qin, Yang Liu, Rongrong Ji, and Hongji Wang. "Asynchronous bidirectional decoding for neural machine translation." In Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

This repo is based on [THUMT](https://github.com/THUNLP-MT/THUMT).

#### Requirements

Python 2.7 and TensorFlow 1.8.0 is required.

#### Train

```shell
python -u thumt/bin/abdtrainer.py --input /path/to/train.en /path/to/train.de /path/to/train.de.pseudo.r2l --vocabulary /path/to/vocab.en /path/to/vocab.de --model abd --validation /path/to/dev.en /path/to/dev.de.pseudo.r2l --references /path/to/dev.de --parameters=batch_size=8000,train_steps=300000,device_list=[0] --output trained/run
```

#### Translate

```shell
python -u thumt/bin/abdtranslator.py --models abd --input /path/to/test.en /path/to/test.de.pseudo.r2l --output test.de.mt --vocabulary /path/to/vocab.en /path/to/vocab.de --checkpoints trained/run/eval/
```

#### Citation

```
@inproceedings{zhang2018asynchronous,
  title={Asynchronous bidirectional decoding for neural machine translation},
  author={Zhang, Xiangwen and Su, Jinsong and Qin, Yue and Liu, Yang and Ji, Rongrong and Wang, Hongji},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```
