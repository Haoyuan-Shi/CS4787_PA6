#!/usr/bin/env python3
import os

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program (should be set to 1 for Parts 1 and 3)
implicit_num_threads = 4
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
#matplotlib.use('agg')
from matplotlib import pyplot
import threading
import time

from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


# SOME UTILITY FUNCTIONS that you may find to be useful, from my PA3 implementation
# feel free to use your own implementation instead if you prefer
def multinomial_logreg_error(Xs, Ys, W):
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error

def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    WdotX = numpy.dot(W, Xs[:,ii])
    expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0))
    softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis = 0)
    return numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
# END UTILITY FUNCTIONS


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(4787)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# SGD + Momentum (adapt from Programming Assignment 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    numpy.random.seed(10)
    # TODO students should use their implementation from programming assignment 3
    # or adapt this version, which is from my own solution to programming assignment 3
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
    return W


# SGD + Momentum (No Allocation) => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    numpy.random.seed(10)
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    V = numpy.zeros(W0.shape)
    W = W0
    CTB = numpy.zeros((c,B))
    numpy.ascontiguousarray(CTB)
    CTD = numpy.zeros((c,d))
    numpy.ascontiguousarray(CTD)
    BT = numpy.zeros(B)
    numpy.ascontiguousarray(BT)
    grad = numpy.zeros((c,d))
    numpy.ascontiguousarray(grad)
    XX = []
    YY = []
    for i in range(int(n/B)):
        ii = range(i*B, (i+1)*B)
        XX.append(Xs[:,ii])
        YY.append(Ys[:,ii])
    numpy.ascontiguousarray(XX)
    numpy.ascontiguousarray(YY)
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    start = time.time()
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            numpy.dot(W, XX[ibatch],out=CTB)
            numpy.exp(numpy.subtract(CTB, numpy.amax(CTB, axis=0,out=BT),out=CTB),out=CTB)
            numpy.divide(CTB, numpy.sum(CTB, axis = 0,out = BT),out=CTB)
            numpy.divide(numpy.dot(numpy.subtract(CTB, YY[ibatch],out=CTB), XX[ibatch].transpose(),out=grad),B,out=grad) 
            numpy.add(grad,numpy.multiply(gamma,W,out=CTD),out=grad)
            numpy.multiply(beta,V,out = CTD)
            numpy.multiply(alpha,grad,out=grad)
            numpy.subtract(CTD,grad,out=V)
            numpy.add(W,V,out=W)
    return W

# SGD + Momentum (threaded)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    V = numpy.zeros(W0.shape)
    W = W0
    grad = numpy.zeros((c,d,num_threads))
    numpy.ascontiguousarray(grad)
    CTD = numpy.zeros((c,d))
    numpy.ascontiguousarray(CTD)
    grad_sum = numpy.zeros((c,d))
    numpy.ascontiguousarray(grad_sum)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)
    BB = int(B/num_threads)
    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations
        CTB = numpy.zeros((c,BB))
        numpy.ascontiguousarray(CTB)
        BT = numpy.zeros(BB)
        numpy.ascontiguousarray(BT)
        XX = []
        YY = []
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B + ithread*BB, ibatch*B + (ithread+1)*BB)
            XX.append(Xs[:,ii])
            YY.append(Ys[:,ii])
        numpy.ascontiguousarray(XX)
        numpy.ascontiguousarray(YY)
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                # ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)               
                numpy.dot(W, XX[ibatch],out=CTB)
                numpy.exp(numpy.subtract(CTB, numpy.amax(CTB, axis=0,out=BT),out=CTB),out=CTB)
                numpy.divide(CTB, numpy.sum(CTB, axis = 0,out = BT),out=CTB)
                numpy.dot(numpy.subtract(CTB, YY[ibatch],out=CTB), XX[ibatch].transpose(),out=CTD) 
                grad[:,:,ithread] = CTD
                iter_barrier.wait()
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            numpy.divide(numpy.sum(grad, axis=2, out=grad_sum),B,out = grad_sum)
            numpy.add(grad_sum,numpy.multiply(gamma,W,out=CTD),out=grad_sum)
            numpy.multiply(beta,V,out = CTD)
            numpy.multiply(alpha,grad_sum,out=grad_sum)
            numpy.subtract(CTD,grad_sum,out=V)
            numpy.add(W,V,out=W)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()
    # return the learned model
    return W


# SGD + Momentum (No Allocation) in 32-bits => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    numpy.random.seed(10)
    Xs = Xs.astype(numpy.float32)
    Ys = Ys.astype(numpy.float32)
    W0 = W0.astype(numpy.float32)
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code
    V = numpy.zeros(W0.shape, dtype=numpy.float32)
    W = W0
    CTB = numpy.zeros((c,B), dtype=numpy.float32)
    numpy.ascontiguousarray(CTB)
    CTD = numpy.zeros((c,d), dtype=numpy.float32)
    numpy.ascontiguousarray(CTD)
    BT = numpy.zeros(B, dtype=numpy.float32)
    numpy.ascontiguousarray(BT)
    grad = numpy.zeros((c,d), dtype=numpy.float32)
    numpy.ascontiguousarray(grad)
    XX = []
    YY = []
    for i in range(int(n/B)):
        ii = range(i*B, (i+1)*B)
        XX.append(Xs[:,ii])
        YY.append(Ys[:,ii])
    numpy.ascontiguousarray(XX, dtype=numpy.float32)
    numpy.ascontiguousarray(YY, dtype=numpy.float32)
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            numpy.dot(W, XX[ibatch],out=CTB)
            numpy.exp(numpy.subtract(CTB, numpy.amax(CTB, axis=0,out=BT),out=CTB),out=CTB)
            numpy.divide(CTB, numpy.sum(CTB, axis = 0,out = BT),out=CTB)
            numpy.divide(numpy.dot(numpy.subtract(CTB, YY[ibatch],out=CTB), XX[ibatch].transpose(),out=grad),B,out=grad) 
            numpy.add(grad,numpy.multiply(gamma,W,out=CTD),out=grad)
            numpy.multiply(beta,V,out = CTD)
            numpy.multiply(alpha,grad,out=grad)
            numpy.subtract(CTD,grad,out=V)
            numpy.add(W,V,out=W)
    return W


# SGD + Momentum (threaded, float32)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    Xs = Xs.astype(numpy.float32)
    Ys = Ys.astype(numpy.float32)
    W0 = W0.astype(numpy.float32)
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should implement this by copying and adapting their 64-bit code
    V = numpy.zeros(W0.shape, dtype=numpy.float32)
    W = W0
    grad = numpy.zeros((c,d,num_threads), dtype=numpy.float32)
    numpy.ascontiguousarray(grad, dtype=numpy.float32)
    CTD = numpy.zeros((c,d), dtype=numpy.float32)
    numpy.ascontiguousarray(CTD, dtype=numpy.float32)
    grad_sum = numpy.zeros((c,d), dtype=numpy.float32)
    numpy.ascontiguousarray(grad_sum, dtype=numpy.float32)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)
    BB = int(B/num_threads)
    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations
        CTB = numpy.zeros((c,BB), dtype=numpy.float32)
        numpy.ascontiguousarray(CTB, dtype=numpy.float32)
        BT = numpy.zeros(BB, dtype=numpy.float32)
        numpy.ascontiguousarray(BT, dtype=numpy.float32)
        XX = []
        YY = []
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B + ithread*BB, ibatch*B + (ithread+1)*BB)
            XX.append(Xs[:,ii])
            YY.append(Ys[:,ii])
        numpy.ascontiguousarray(XX, dtype=numpy.float32)
        numpy.ascontiguousarray(YY, dtype=numpy.float32)
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                # ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)               
                numpy.dot(W, XX[ibatch],out=CTB)
                numpy.exp(numpy.subtract(CTB, numpy.amax(CTB, axis=0,out=BT),out=CTB),out=CTB)
                numpy.divide(CTB, numpy.sum(CTB, axis = 0,out = BT),out=CTB)
                numpy.dot(numpy.subtract(CTB, YY[ibatch],out=CTB), XX[ibatch].transpose(),out=CTD) 
                grad[:,:,ithread] = CTD
                iter_barrier.wait()
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            numpy.divide(numpy.sum(grad, axis=2, out=grad_sum),B,out = grad_sum)
            numpy.add(grad_sum,numpy.multiply(gamma,W,out=CTD),out=grad_sum)
            numpy.multiply(beta,V,out = CTD)
            numpy.multiply(alpha,grad_sum,out=grad_sum)
            numpy.subtract(CTD,grad_sum,out=V)
            numpy.add(W,V,out=W)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    (d, n) = Xs_tr.shape
    (c, n) = Ys_tr.shape
    #----------------- Part 1.3 --------------------
    alpha = 0.1
    beta = 0.9
    B = 16
    gamma = 0.0001
    num_epochs = 20
    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    '''
    numpy.random.seed(10)
    W0=numpy.random.rand(c,d)
    start = time.time()
    sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
    end = time.time()
    runningTime = end - start
    start = time.time()
    sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
    end = time.time()
    runningTime_noalloc = end - start
    #print(W)
    #print(W_noalloc)
    print(runningTime)
    print(runningTime_noalloc)

    T1 = [31.645213842391968]
    T2 = [18.75607705116272]
    
    #----------------- Part 1.4 --------------------
    
    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    runningTime_function1 = numpy.zeros(len(BatchSize))
    runningTime_function2 = numpy.zeros(len(BatchSize))
    
    for i,B in enumerate(BatchSize):
        numpy.random.seed(10)
        W0=numpy.random.rand(c,d)
        start = time.time()
        sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        end = time.time()
        runningTime_function1[i] = end-start
        start = time.time()
        sgd_mss_with_momentum_noalloc(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        end = time.time()    
        runningTime_function2[i] = end-start

    print(runningTime_function1)
    print(runningTime_function2)
    
    
    pyplot.plot(numpy.log(BatchSize),runningTime_1Thread_with_allocation,label = "1 thread with allocation")
    pyplot.plot(numpy.log(BatchSize),runningTime_1Thread_without_allocation,label = "1 thread without allocation")
    pyplot.plot(numpy.log(BatchSize),runningTime_function1,label = "4 threads with allocation")
    pyplot.plot(numpy.log(BatchSize),runningTime_function2,label = "4 threads without allocation")

    pyplot.legend()
    pyplot.xlabel("log(Batch Size)")
    pyplot.ylabel("Wall-clock times (sec)")
    pyplot.minorticks_on()
    pyplot.show()
    

    #----------------- Part 3 --------------------
    
    
    # Part 3
    alpha = 0.1
    beta = 0.9
    gamma = 0.0001
    num_epochs = 20
    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    runningTime_function3 = numpy.zeros(len(BatchSize))
    num_threads = 4
    for i,B in enumerate(BatchSize):
        numpy.random.seed(10)
        W0=numpy.random.rand(c,d)
        start = time.time()
        sgd_mss_with_momentum_threaded(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs,num_threads)
        end = time.time()
        runningTime_function3[i] = end-start
    print(runningTime_function3)

    
    '''
    runningTime_1Thread_with_allocation = [52.17557883,31.39143443, 22.97072291, 18.75415611, 14.70796633, 11.96760631,11.91203523]
    runningTime_1Thread_without_allocation = [30.94371533, 19.20824242, 13.574512, 10.58016515, 7.30097246, 7.2607975,6.30259347]
    runningTime_4Thread_with_allocation = [49.1065228,  30.5379777,  24.24255371, 19.32755065, 12.99099326, 11.21421123, 9.21772575]
    runningTime_4Thread_without_allocation = [29.71110582,18.98925781,13.08909869,8.95436907,5.04137897,3.73946047,3.47340536]
    runningTime_4Thread_multithreading = [172.21271634,  84.06371689,  50.96550298,  26.82168555,   8.85591698, 4.24953127,   2.21880579]
    runningTime_1Thread_32float_without_allocation = [27.478374,16.02139592, 11.49432158, 8.70450497, 7.34732604,  6.67453861, 6.44164276]    
    runningTime_4Thread_32float_without_allocation = [25.66877294,  8.45196152,  6.48774958,  5.09150052,  2.62744069,  2.30011964, 2.28426647]
    runningTime_4Thread_32float_multithreading = [160.98149586,81.5410459,44.88501406,23.00822234,8.7163167,4.1080749,2.26823258]

    pyplot.plot(numpy.log(BatchSize),runningTime_1Thread_with_allocation,label = "1 thread with allocation")
    pyplot.plot(numpy.log(BatchSize),runningTime_1Thread_without_allocation,label = "1 thread without allocation")
    #pyplot.plot(numpy.log(BatchSize),runningTime_4Thread_with_allocation,label = "4 threads with allocation")
    #pyplot.plot(numpy.log(BatchSize),runningTime_4Thread_without_allocation,label = "4 threads without allocation")
    #pyplot.plot(numpy.log(BatchSize),runningTime_4Thread_multithreading,label = "4 threads multi-threading")
    #pyplot.plot(numpy.log(BatchSize),runningTime_1Thread_32float_without_allocation,label = "1 thread float32 without allocation")
    #pyplot.plot(numpy.log(BatchSize),runningTime_4Thread_32float_without_allocation,label = "4 threads float32 without allocation")
    #pyplot.plot(numpy.log(BatchSize),runningTime_4Thread_32float_multithreading,label = "4 threads float32 multi-threading")

    pyplot.legend()
    pyplot.xlabel("log(Batch Size)")
    pyplot.ylabel("Wall-clock times (sec)")
    pyplot.minorticks_on()
    pyplot.show()
   
    '''
    
    # Part 4
    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    runningTime_function4 = numpy.zeros(len(BatchSize))
    for i,B in enumerate(BatchSize):
        numpy.random.seed(10)
        W0=numpy.random.rand(c,d)
        start = time.time()
        sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        end = time.time()
        runningTime_function4[i] = end-start
    print(runningTime_function4)

    
     
    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    runningTime_function5 = numpy.zeros(len(BatchSize))
    for i,B in enumerate(BatchSize):
        numpy.random.seed(10)
        W0=numpy.random.rand(c,d)
        start = time.time()
        sgd_mss_with_momentum_noalloc_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs)
        end = time.time()
        runningTime_function5[i] = end-start
    print(runningTime_function5)


    

    BatchSize = numpy.array([8,16,30,60,200,600,3000])
    runningTime_function6 = numpy.zeros(len(BatchSize))
    num_threads = 4
    for i,B in enumerate(BatchSize):
        numpy.random.seed(10)
        W0=numpy.random.rand(c,d)
        start = time.time()
        sgd_mss_with_momentum_threaded_float32(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs,num_threads)
        end = time.time()
        runningTime_function6[i] = end-start
    print(runningTime_function6)


    '''
    
