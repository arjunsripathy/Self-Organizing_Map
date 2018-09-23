import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
import csv

#dataFile = open("seeds.txt",'r')
att = 8

orderedData = []

with open("HTRU_2.csv",'rU') as csvfile:
	reader = csv.reader(csvfile, dialect=csv.excel_tab)
	for row in reader:
		orderedData.append(row)


'''for line in dataFile:
	orderedData.append(line)'''


randomData = random.shuffle(orderedData)

uData = []
names = []

for i in range(len(orderedData)):
	line = (orderedData[i][0]).strip()
	data = line.split(',')
	fD = (np.array(data[:att])).astype(np.float)
	uData.append(fD)
	names.append(data[att])


numData = len(uData)

means = np.zeros(att)

for i in range(numData):
	means = np.add(means,uData[i])

means /= numData


stdDevs = np.zeros(att)

for i in range(numData):
	diff = np.subtract(means,uData[i])
	stdDevs = np.add(stdDevs,np.multiply(diff,diff))

stdDevs /= numData

for i in range(att):
	stdDevs[i] = math.sqrt(stdDevs[i])

data = []

divStdDevs = np.reciprocal(stdDevs)

for i in range(numData):
	diff = uData[i]-means
	normalized = np.multiply(diff,divStdDevs)
	data.append(normalized)

labels = []

classes = 2

classDict = {"0":0,"1":1}

for i in range(numData):
	labels.append(classDict[names[i]])

n = 15
LEARNING_RATE_INIT = 0.3
RADIUS_INIT = n/4
EPOCHS = 2
ITERATIONS = EPOCHS*numData

weightMatrix = np.random.normal(size=[n,n,att])

def sqd(x,y):
	dim = len(x)
	ret = 0
	for i in range(dim):
		diff = x[i]-y[i]
		ret += diff*diff
	return ret

#x,y ints, r float
#2d only
def getCircle(x,y,r):
	mx0 = int(r)
	maxes = np.zeros(mx0+1)
	for i in range(len(maxes)):
		thisY = i
		maxX = int(math.sqrt(r*r-thisY*thisY))
		maxes[i] = maxX

	circle = []
	for i in range(-mx0,mx0+1):
		maxX = int(maxes[abs(i)])
		for j in range(-maxX,maxX+1):
			thisX = x+j
			thisY = y+i
			circle.append([thisX,thisY])

	return circle

def removeInvalid(points):
	validPoints = []
	for p in points:
		x = p[0]
		y = p[1]
		if(not (x<0 or x>=n or y<0 or y>=n)):
			validPoints.append(p)
	return validPoints


def updateW(learningRate,mV,i,r):
	validCircle = removeInvalid(getCircle(mV[0],mV[1],r))
	for p in validCircle:
		squaredDistance = sqd(p,mV)
		distanceMultiplier = np.exp(squaredDistance/(2*r*r))
		netMultiplier = learningRate*distanceMultiplier

		x = p[0]
		y = p[1]
		wV = weightMatrix[x][y]
		uwV = np.add(wV,netMultiplier*(np.subtract(i,wV)))
		weightMatrix[x][y] = uwV

def distanceMatrix(i):
	dm = np.zeros([n,n])
	for x in range(n):
		for y in range(n):
			dm[x][y] = sqd(i,weightMatrix[x][y])
	return dm

def dataMatrixCL():

	dm = np.zeros([n,n,3])

	for i in range(numData):
		inp = data[i]

		initialized = False
		minD = 0
		minV = [0,0]
		for x in range(n):
			for y in range(n):
				weightVector = weightMatrix[x][y]
				squaredDistance = sqd(inp,weightVector)
				if(not initialized or squaredDistance<minD):
					initialized = True
					minD = squaredDistance
					minV = [x,y]

		dataClass = labels[i]
		dm[minV[0]][minV[1]][dataClass] += 1

	return dm

def colorMatrix(dMatrixCL):

	scale = 1.0/20
	raw = scale * dMatrixCL
	ret = np.zeros(np.shape(dMatrixCL))
	for i in range(n):
		for j in range(n):
			for k in range(classes):
				val = raw[i][j][k]
				ret[i][j][k] = min(val,1.0)

	return ret

def showColorMatrix():
	c = colorMatrix(dataMatrixCL())

	plt.imshow(c)
	plt.pause(0.03)

showColorMatrix()

for j in range(EPOCHS):

	for i in range(numData):

		lr = LEARNING_RATE_INIT*np.exp(-3*float(j*numData+i)/ITERATIONS)
		radius = RADIUS_INIT*np.exp(-3*float(j*numData+i)/ITERATIONS)

		inp = data[i]

		initialized = False
		minD = 0
		minV = [0,0]
		for x in range(n):
			for y in range(n):
				weightVector = weightMatrix[x][y]
				squaredDistance = sqd(inp,weightVector)
				if(not initialized or squaredDistance<minD):
					initialized = True
					minD = squaredDistance
					minV = [x,y]

		updateW(lr,minV,inp,radius)

		if(i%30==0):
			showColorMatrix()

