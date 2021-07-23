#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:48:49 2021

Bond raffles

@author: sean
"""
import random
from pyexcel_ods import get_data
from itertools import chain

data = get_data('/home/sean/Documents/bingo_participants.ods')
stupid_list = list(data['Sheet1'])
participant_list = list(chain(*stupid_list))
random.shuffle(participant_list)
print(participant_list[0])
