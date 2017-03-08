import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=np.max(y)+1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  for i in range(num_train):
      scores=X[i].dot(W)  #1*C
      C=np.amax(scores)
      only_sum=np.sum(scores)
      sum=np.sum(np.exp(scores -C))  #N*1
      true_class_score=np.exp(scores[y[i]] - C )
      loss+=-1*np.log(true_class_score/sum)
      dW[:,y[i]]+=-1*X[i]
      d=np.reshape(np.exp(scores),(num_classes,1)) *X[i]
      dW+=d.T/np.sum(np.exp(scores))
  loss=loss/num_train +0.5*reg*np.sum(W*W)
  dW=dW/num_train + reg*W
      
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train=X.shape[0]
  num_classes=np.max(y)+1

  scores=X.dot(W) #N*C
  C=np.max(scores,axis=1).reshape((num_train,1))  #N*1
  soft=np.exp(scores-C) #N*C
  sum= np.sum(scores,axis=1) #N*1
  tot_prob=np.sum(soft,axis=1).reshape((num_train,1)) #N*1
  prob=soft/tot_prob #N*C

  loss=prob[range(num_train) , y]
  loss=-1*np.log(loss)/num_train
  loss=np.sum(loss) + 0.5*reg*np.sum(W*W)
  
  prob=prob.T  #C*N
  prob[y,range(num_train)] -=1 
  dW=(prob.dot(X)).T/num_train
  dW= dW+ reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

