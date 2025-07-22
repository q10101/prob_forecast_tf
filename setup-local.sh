#!/bin/bash

DIR=$(realpath $(dirname $0))
cd $DIR

ENV=.environment

python3 -m venv --system-site-packages $ENV
source $ENV/bin/activate

pip install matplotlib
pip install scikit-learn
