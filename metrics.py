'''Loss and metrics for 3D segmentation. '''

import torch
import torch.nn.functional as F


def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = (y_true * y_pred).sum(dim=(0, 1, 2, 3))
    return (2. * intersection) / ((y_true + y_pred).sum(dim=(0, 1, 2, 3)) + epsilon)


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = torch.abs(
        y_true[0, :, :, :, 1] * y_pred[0, :, :, :, 1]).sum()
    return (2. * intersection) / ((y_true[0, :, :, :, 1] ** 2).sum() + (y_pred[0, :, :, :, 1] ** 2).sum() + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = torch.abs(
        y_true[0, :, :, :, 2] * y_pred[0, :, :, :, 2]).sum()
    return (2. * intersection) / ((y_true[0, :, :, :, 2] ** 2).sum() + (y_pred[0, :, :, :, 2] ** 2).sum() + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = torch.abs(
        y_true[0, :, :, :, 3] * y_pred[0, :, :, :, 3]).sum()
    return (2. * intersection) / ((y_true[0, :, :, :, 3] ** 2).sum() + (y_pred[0, :, :, :, 3] ** 2).sum() + epsilon)


def dice_coef_tc(y_true, y_pred, epsilon=1e-6):
    dice_et = dice_coef_enhancing(y_true, y_pred, epsilon)
    dice_necrotic = dice_coef_necrotic(y_true, y_pred, epsilon)
    return (2*dice_et*dice_necrotic)/(dice_necrotic+dice_et+epsilon)


def dice_coef_wt(y_true, y_pred, epsilon=1e-6):
    '''WT represents the union of all three labels'''
    dice_tc = dice_coef_tc(y_true, y_pred, epsilon)
    dice_edema = dice_coef_edema(y_true, y_pred, epsilon)
    return (2*dice_tc*dice_edema)/(dice_tc+dice_edema+epsilon)


def precision(y_true, y_pred):
    true_positives = (y_true * y_pred).sum().float()
    predicted_positives = y_pred.sum().float()
    return true_positives / (predicted_positives + 1e-6)


def sensitivity(y_true, y_pred):
    true_positives = (y_true * y_pred).sum().float()
    possible_positives = y_true.sum().float()
    return true_positives / (possible_positives + 1e-6)


def specificity(y_true, y_pred):
    true_negatives = ((1 - y_true) * (1 - y_pred)).sum().float()
    possible_negatives = (1 - y_true).sum().float()
    return true_negatives / (possible_negatives + 1e-6)
