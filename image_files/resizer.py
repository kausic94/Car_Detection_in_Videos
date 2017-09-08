#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:46:59 2017

@author: kausic
"""

import cv2
import numpy as np
import sys

argument=sys.argv[1]
img=cv2.resize(cv2.imread(argument),(300,300))
cv2.imwrite(argument[:-4]+ "_.jpg",img)