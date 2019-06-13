#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'

import tensorflow as tf

class FocalLoss(object):
    def __init__(self,  alpha=0.25, gamma=2):
        """
        :param alpha: A scalar tensor for focal loss alpha hyper-parameter
        :param gamma: A scalar tensor for focal loss gamma hyper-parameter
        """
        self.alpha = alpha
        self.gamma = gamma

    def get_loss(self, logits, labels, weights=None):
        """Compute focal loss for predictions.
                Multi-labels Focal loss formula:
                    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                         ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = labels.
            Args:
             logits: A float tensor of shape [batch_size,num_classes]
             labels: A float tensor of shape [batch_size,num_classes]
             weights: A float tensor of shape [batch_size, num_classes]
            Returns:
                loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(logits)
        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        # For poitive prediction, only need consider front part loss, back part is 0;
        pos_p_sub = tf.where(labels > zeros, labels - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # labels > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(labels > zeros, zeros, sigmoid_p)
        fl_loss = - self.alpha * (pos_p_sub ** self.gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                  - (1 - self.alpha) * (neg_p_sub ** self.gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

        if weights is None:
            loss = tf.reduce_sum(fl_loss*weights)/tf.maximum(tf.reduce_sum(weights), 1e-5)
        else:
            loss = tf.reduce_sum(fl_loss)

        return loss







