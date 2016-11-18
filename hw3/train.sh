#!/bin/bash
THEANO_FLAGS=device=gpu,floatX=float32 python self_training.py $1 $2
