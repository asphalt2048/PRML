from builtins import range
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        #compute the predicted class of an sample X[i], 
        #output is in the from of [1,C], standing for the probability of X[i] belonging to each of the class in C
        """
        Let's dive into X[i]*W:
        every column in the output(which is actually just one value) is X[i]*W[c],
        where "c" stands for a class, 0 <= c < C,
        it feels like we have a W[][c] for every class c by dividing the category set into "c" and "not c",
        and classificate X[i] in a "winner takes all" fashion: the class that has the biggest value in the 
        output of X[i]*W is the class of X[i]. 
        So it is a one-against-rest multiclass SVM
        """
        scores = X[i].dot(W)
        #real class of X[i] is y[i]
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            #We want the score of "real class" to differ from score in "fake class" at the margin of at least 1
            #note that real class has positive label
            """
            we may not be very familiar with this from of Loss in case of SVM,
            but, the from of loss actually NOT MATTERS.
            Loss = 1/N*( Sum:for all x[i] ( Sum:for every c!=y[i] (max{0, x[i]*W[][c]-x[i]*W[][y[i]]+1}))) + reg*W^2  
            """
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    """
    don't need to vectorize
    """
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #TODO
    dW = np.multiply(dW, 1/num_train)
    dW += np.multiply(W, 2*reg)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #create a all-zero matrix
    Temp1 = np.zeros((X.shape[0], W.shape[1]))
    #computing matrix multiply using @
    S = X@W
    #feature1: high-level indexing A[arg1, arg2], form every rows in arg1, select a element of column given by arg2
    #np.arange(len) creates a array of length "len", and make a[i] = i
    #y is of shape [N,], in numpy, this means it is an array. note the difference between [N,1]
    Temp2 = S[np.arange(S.shape[0]), y]
    #reshape, to-do
    Temp2 = Temp2.reshape(-1, 1)
    #feature2: Broadcasting. to-do
    #This will substitude elements in every row of Temp1 with the value given by the corresponding row of Temp2
    Temp1[:] = Temp2
    S -= Temp1
    #plus 1 to every element in S
    S = S + 1
    #feature3: to-do
    S[S<0] = 0
    S[np.arange(S.shape[0]),y] = 0
    loss += np.sum(S)
    loss = loss/X.shape[0]
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #reuse S, since S now perserve all Xi.c that satisfy: X[i].W[][c] - X[i].W[][yi] + 1 > 0
    #I didn't calculate the derivation directly here, since in svm_loss_native we already konw what to do with dW
    S[S>0] = 1
    #note dW[:, j] += X[i, :] can be rewriten as this
    dW += X.T@S
    #construct a N*C matrix that has value "sum of S[i, :]" on (i, j) if X[i]'s real class is j, and value 0 on others 
    Temp3 = np.zeros((X.shape[0], W.shape[1]))
    Temp3[np.arange(Temp3.shape[0]), y] = np.sum(S, axis=1)
    #note that dW[:, y[i]] -= X[i, :] can be rewriten as this
    dW -= X.T@Temp3

    dW = np.multiply(dW, 1/X.shape[0])
    dW += np.multiply(W, 2*reg)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
