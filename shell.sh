#!/bin/bash

DIR=$(realpath $(dirname $0))

TARGET=/opt/prob_forecast_tf

sudo docker run -it --rm --runtime=nvidia --gpus all \
    -v $DIR:$TARGET \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/root/.Xauthority \
    -e DISPLAY=$DISPLAY \
    tensorflow/tensorflow:2.13.0-gpu \
    bash
    
