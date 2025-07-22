#!/bin/bash

DIR=$(realpath $(dirname $0))
cd $DIR

ENV=.environment

source $ENV/bin/activate

mkdir -p output/
python3 file0_main.py
