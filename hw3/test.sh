#!/bin/bash
THEANO_FLAGS=device=gpu,floatX=float32 python self_training_test.py $1 $2 $3
