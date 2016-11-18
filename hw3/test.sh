#!/bin/bash
THEANO_FLAGS=device=gpu,floatX=float32 python autoencoder_testing.py $1 $2 $3
