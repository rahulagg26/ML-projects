# -*- coding: utf-8 -*-

import numpy as np
# This is the only scipy method you are allowed to use
# Use of scipy is not allowed otherwise
from scipy.linalg import khatri_rao


def createFeatures( X ):
    return np.cumprod( np.flip( 2 * X - 1 , axis = 1 ), axis = 1 )


def get_renamed_labels( y ):

#  Non Editable Region Ending  #
################################

 	# Since the dataset contain 0/1 labels and SVMs prefer -1/+1 labels,
 	# Decide here how you want to rename the labels
 	# For example, you may map 1 -> 1 and 0 -> -1 or else you may want to go with 1 -> -1 and 0 -> 1
 	# Use whatever convention you seem fit but use the same mapping throughout your code
 	# If you use one mapping for train and another for test, you will get poor accuracy
    y_new=2*y-1
     
    
    return y_new.reshape( ( y_new.size, -1) )

###############################
#Non Editable Region Starting #
###############################
def get_features( X ):
################################
#  Non Editable Region Ending  #
################################

 	# Use this function to transform your input features (that are 0/1 valued)
 	# into new features that can be fed into a linear model to solve the problem
 	# Your new features may have a different dimensionality than the input features
 	# For example, in this application, X will be 8 dimensional but your new
 	# features can be 2 dimensional, 10 dimensional, 1000 dimensional, 123456 dimensional etc
 	# Keep in mind that the more dimensions you use, the slower will be your solver too
 	# so use only as many dimensions as are absolutely required to solve the problem
 	X_new=(khatri_rao(khatri_rao(X.T, X.T),X.T)).T
    
    
    
 	return X_new


def processdata (s):
    data= np.loadtxt(s)
    xtemp= createFeatures(data[:,:-1])
    x=np.concatenate((np.ones((xtemp.shape[0],1)),xtemp), axis=1)
    y=get_renamed_labels(data[:,-1])
    X=get_features(x)
    
    return (X,y)

(X_train, y_train)=processdata ("train.dat")

def solver(X_train, Y_train, lr=0.5e-01, epochs=5, la=0.2e-03):
    weights = np.zeros((X_train.shape[1],1))
    X = X_train
    y = Y_train
    # b = 0
    # loss = np.zeros((epochs,))
    
    for _ in range(epochs):
        for idx, x in enumerate(X):
            
            divider = (y[idx]*(np.dot(x, weights)) >= 1)
            
            if divider:
                weights = weights - lr*la*weights
                # b = b
            else:
                
                grad =  la*weights - np.dot(np.reshape(x, (-1, 1)), np.reshape(y[idx], (-1,1)))
                # grad_b = -y[idx]
                weights = weights - lr*grad
                # b  = b - lr*grad_b

    return weights

w= solver(X_train, y_train)

(X_test, y_test)=processdata ("test.dat")

def Predict(X_test, weights):
    
    # print(X.shape)
    return np.sign(np.dot(X_test, weights))

Y_predict=Predict(X_test, w)

mse=np.mean((Y_predict-y_test)**2)
