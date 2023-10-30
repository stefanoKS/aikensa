#!/bin/bash

sudo v4l2-ctl -c brightness=0
sudo v4l2-ctl -c contrast=1
sudo v4l2-ctl -c saturation=60
sudo v4l2-ctl -c hue=0
sudo v4l2-ctl -c white_balance_automatic=1
sudo v4l2-ctl -c gamma=100
sudo v4l2-ctl -c gain=0
sudo v4l2-ctl -c power_line_frequency=1
sudo v4l2-ctl -c white_balance_temperature=4600
sudo v4l2-ctl -c sharpness=0
sudo v4l2-ctl -c backlight_compensation=64
sudo v4l2-ctl -c auto_exposure=3
sudo v4l2-ctl -c exposure_time_absolute=156
