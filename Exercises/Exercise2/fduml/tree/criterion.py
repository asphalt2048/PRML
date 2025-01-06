"""
criterion
"""

import math


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total_LN = 0
    left_LN = 0
    right_LN = 0
    for t in all_labels:
        total_LN += all_labels[t]
    for t in left_labels:
        left_LN += left_labels[t]
    for t in right_labels:
        right_LN += right_labels[t]
    Entorpy_before = 0
    for t in all_labels:
        Entorpy_before -= all_labels[t]/total_LN * math.log2(all_labels[t]/total_LN)
    Entorpy_after = 0
    t1 = 0
    t2 = 0
    for t in left_labels:
        t1 -= left_labels[t]/left_LN * math.log2(left_labels[t]/left_LN)
    t1 *= left_LN/total_LN
    for t in right_labels:
        t2 -= right_labels[t]/right_LN * math.log2(right_labels[t]/right_LN) 
    t2 *= right_LN/total_LN
    Entorpy_after = t1 + t2
    info_gain = Entorpy_before - Entorpy_after
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    total_LN = y.size
    right_LN = r_y.size
    left_LN = l_y.size
    split_info = 0
    if left_LN == 0:
        split_info = -(right_LN/total_LN * math.log2(right_LN/total_LN))
    elif right_LN == 0:
        split_info = -(left_LN/total_LN * math.log2(left_LN/total_LN))
    else:
        split_info = -(left_LN/total_LN * math.log2(left_LN/total_LN) + right_LN/total_LN * math.log2(right_LN/total_LN))
    info_gain = info_gain/(split_info + 1e-20)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    before = 1
    totol_LN = y.size
    right_LN = r_y.size
    left_LN = l_y.size
    for t in all_labels:
        before -= math.pow(all_labels[t]/totol_LN, 2)
    after = 0
    t1 = 1
    t2 = 1
    for t in left_labels:
        t1 -= math.pow(left_labels[t]/left_LN, 2)
    t1 *= left_LN/totol_LN
    for t in right_labels:
        t2 -= math.pow(right_labels[t]/right_LN, 2)
    t2 *= right_LN/totol_LN
    after = t1 + t2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_max = 0
    l_max = 0
    r_max = 0
    for t in all_labels:
        if all_labels[t] > all_max:
            all_max = all_labels[t]
    for t in left_labels:
        if left_labels[t] > l_max:
            l_max = left_labels[t]
    for t in right_labels:
        if right_labels[t] > r_max:
            r_max = right_labels[t]
    totol_LN = y.size
    right_LN = r_y.size
    left_LN = l_y.size
    before = 1 - all_max/totol_LN
    if left_LN != 0:
        error_l = 1 - l_max/left_LN
    else:
        error_l = 0
    if right_LN != 0:
        error_r = 1 - r_max/right_LN
    else:
        error_r = 0
    after = error_l*(left_LN/totol_LN) + error_r*(right_LN/totol_LN)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
