# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from scipy import misc


def tpr95(cifar, other):
	# calculate the falsepositive error when tpr is 95%
	# calculate baseline
	T = 1
	start = 0.1
	end = 1.0
	gap = (end - start)/100000
	# f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
	Y1 = other
	X1 = cifar
	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	fprBase = fpr/total
	return fprBase


def auroc(cifar, other):
	# calculate the AUROC
	# calculate baseline
	T = 1
	start = 0.0
	end = 1.0
	gap = (end - start)/100000
	# f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
	Y1 = other
	X1 = cifar
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr

	return aurocBase


def auprIn(cifar, other):
	# calculate the AUPR
	# calculate baseline
	T = 1
	start = 0.0
	end = 1.0
	gap = (end- start)/100000
	precisionVec = []
	recallVec = []
	Y1 = other
	X1 = cifar
	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def auprOut(cifar, other):
	# calculate the AUPR
	# calculate baseline
	T = 1
	start = 0.0
	end = 1.0
	gap = (end- start)/100000
	Y1 = other
	X1 = cifar
	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision
	return auprBase

def detection(cifar, other):
	# calculate the minimum detection error
	# calculate baseline
	T = 1
	start = 0.0
	end = 1
	gap = (end- start)/100000
	Y1 = other
	X1 = cifar
	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

	return errorBase
