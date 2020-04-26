import numpy as np
import sys
import random
# Reading input data

a = open(sys.argv[1])
data = np.loadtxt(a)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

print("train=",train)
print("train shape=",train.shape)

b = open(sys.argv[2])
data = np.loadtxt(b)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

print("test=",test)
print("test shape=",test.shape)

row = train.shape[0]
col = train.shape[1]

hidden_node = 3

# Initializing all weights

w = np.random.rand(hidden_node)
print("w=",w)

W = np.random.rand(hidden_node, col)
print("W=",W)

epochs = 10000
eta = .01
prevobj = np.inf
i=0

# Calculating objective

hidden_layer = np.matmul(train, np.transpose(W))
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
print("hidden_layer=",hidden_layer)
print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
print("obj=",obj)

# gradient descent calculation

#stop=0.01
while(prevobj - obj > 0.0000001 and i < epochs):
#while(prevobj - obj > 0):

	#Update previous objective
	prevobj = obj

	#Calculate gradient update for final layer (w)
	#dellw is the same dimension as w

	dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
	for j in range(1, row):
		dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]

	#Update w
	w = w - eta*dellw
	print("w=",w)
	
	#Calculate gradient update for hidden layer weights (W)
	#dellW has to be of same dimension as W

	#Let's first calculate dells. After that we do dellu and dellv.
	#Here s, u, and v are the three hidden nodes
	#dells = df/dz1 * (dz1/ds1, dz1,ds2)
	dells = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
	for j in range(1, row):
		dells += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]

	#dellu
	dellu = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[1] * (hidden_layer[0,1])*(1-hidden_layer[0,1])*train[0]
	for j in range(1, row):
		dellu += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[1] * (hidden_layer[j,1])*(1-hidden_layer[j,1])*train[j]
	#dellv
	dellv = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[2] * (hidden_layer[0,2])*(1-hidden_layer[0,2])*train[0]
	for j in range(1, row):
		dellv += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[2] * (hidden_layer[j,2])*(1-hidden_layer[j,2])*train[j]
	#TODO: Put dells, dellu, and dellv as rows of dellW
		
	dellW=np.vstack((dells,dellu,dellv))

	#Update W
	W = W - eta*dellW

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
	print("hidden_layer=",hidden_layer)

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
	print("hidden_layer=",hidden_layer)

	output_layer = np.matmul(hidden_layer, np.transpose(w))
	print("output_layer=",output_layer)

	obj = np.sum(np.square(output_layer - trainlabels))
	i = i + 1

'''
# final predictions
x=np.matmul(train,np.transpose(W))
predictions=np.sign(np.matmul(sigmoid(x),np.transpose(w)))
print("Final Predictions=",predictions)


err = 0
avgerr=0
for i in range(0, len(predictions), 1):
    if(predictions[i] != trainlabels[i]):
        err += 1
err = err/float(len(trainlabels))
print("Test Error=",err)
avgerr+=err

avgerr/=10
print("Avg error=",avgerr)
'''

def prediction(x):
	hl = np.matmul(x, np.transpose(W))
	hl = np.array([sigmoid(xi) for xi in hl])
	ol = np.matmul(hl, np.transpose(w))
	predictions = np.sign(ol)
	return predictions

trpred = prediction(train)
tepred = prediction(test)

trerr = (1 - (trpred == trainlabels).mean()) * 100
teerr = (1 - (tepred == testlabels).mean()) * 100

print('train predictions: \t\t', trpred)
print('train error: \t\t', trerr)

print('test predictions: \t\t', tepred)
print('test error: \t\t', teerr)
