# Find Crossings
#------------------------------------------------------------------------------


# Libraries
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

"""
naiveCross
"""
def naiveCross(xMatrix, yMatrix, zMatrix):
	#simply does iterative n^2 operation for two axes at a time -- 3 choose 2
	crossingsDict = {k: [] for k in [('x','y'),('y','z'),('z','x')]}
	

	for x1, x2 in window(xMatrix):
		for y1, y2 in window(yMatrix):
			value = findIntersect(*x1, *x2, *y1, *y2)
			if (value):
				crossingsDict[('x','y')].append(value)

	for y1, y2 in window(yMatrix):
		for z1, z2 in window(zMatrix):
			value = findIntersect(*y1, *y2, *z1, *z2)
			if (value):
				crossingsDict[('y','z')].append(value)

	for z1, z2 in window(zMatrix):
		for x1, x2 in window(xMatrix):
			value = findIntersect(*z1, *z2, *x1, *x2)
			if (value):
				crossingsDict[('z','x')].append(value)

	return crossingsDict

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

"""
testCrosses
simple testing function to check for all the crosses
"""
def testCrosses(numPoints = 10):
	time = np.random.randint(0, 100, size = numPoints)
	time = np.unique(time)
	xVals = np.random.randint(0, 100, size = len(time))
	xMatrix = np.asarray(list(zip(time, xVals)))

	time = np.random.randint(0, 100, size = numPoints)
	time = np.unique(time)
	yVals = np.random.randint(0, 100, size = len(time))	
	yMatrix = np.asarray(list(zip(time, yVals)))
	
	time = np.random.randint(0, 100, size = numPoints)
	time = np.unique(time)
	zVals = np.random.randint(0, 100, size = len(time))
	zMatrix = np.asarray(list(zip(time, zVals)))

	plt.plot(xMatrix[:,0], xMatrix[:,1], color = 'black')
	plt.plot(yMatrix[:,0], yMatrix[:,1], color = 'black')
	plt.plot(zMatrix[:,0], zMatrix[:,1], color = 'black')
	
	crossingsDict = naiveCross(xMatrix, yMatrix, zMatrix)
	for key, color in zip(crossingsDict.keys(), ['red','blue', 'green']):
		for x in crossingsDict[key]:
			plt.axvline(x = x, color = color)

	plt.show()
	plt.close()





















####
#IGNORE: WORKING ON OPTIMIZATIONS
####



"""
crosses
find the crosses in the original GUPR vectors (e.g. matrix, before rebased)
returns the "base" of the vectors at which they intersect
"""
def asdfcrosses(xMatrix, yMatrix, zMatrix):
	lastTimeStamp = xMatrix[-1][0]
	currentTime = xMatrix[0][0]

	#output is an intersectionLocation dictionary
	#key is timestamp and value are the axes that crossed at that timestamp
	intersectionLocations = dict()
	xStart = xMatrix.pop(0)
	xEnd = xMatrix.pop(0)
	yStart = yMatrix.pop(0)
	zStart = zMatrix.pop(0)

	currentAxis = 'x'
	#keep going until there's no more vectors to be considered (in this case)
	while lastTimeStamp != currentTime:
		if (currentAxis == 'x'):
			yEnd = yMatrix.pop(0)
			zEnd = zMatrix.pop(0)
			possible1 = findIntersect(*xStart, *xEnd, *yStart, *yEnd)
			possible2 = findIntersect(*xStart, *xEnd, *zStart, *zEnd)
			possible3 = findIntersect(*yStart, *yEnd, *zStart, *zEnd)
			


			if yEnd[0] <= xEnd[0]:
				yStart = yEnd
				yEnd = yEnd = yMatrix.pop(0)
			else:
				currentAxis = 'y'

			if zEnd[0] <= xEnd[0]:
				zStart = zEnd
			else:
				pass

			if (currentAxis != 'x'):
				xStart = xEnd
				xEnd = xMatrix.pop(0)

		elif (currentAxis == 'y'):
			xEnd = xMatrix.pop(0)
			zEnd = zMatrix.pop(0)
			findIntersect(*xStart, *xEnd, *yStart, *yEnd)
			findIntersect(*xStart, *xEnd, *zStart, *zEnd)
			findIntersect(*yStart, *yEnd, *zStart, *zEnd)
			
		else:
			xEnd = xMatrix.pop(0)
			yEnd = yMatrix.pop(0)
			findIntersect(*xStart, *xEnd, *yStart, *yEnd)
			findIntersect(*xStart, *xEnd, *zStart, *zEnd)
			findIntersect(*yStart, *yEnd, *zStart, *zEnd)
			

	return intersectionLocations

"""
findIntersect
Given two points on each line, find the point of intersect
https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
(currently we only care about the timestamp - P_x in wiki)
"""
def findIntersect(x1, y1, x2, y2, x3, y3, x4, y4):
	numerator = (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)
	denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

	#if denominator is zero, the two lines are parallel
	if not denominator:
		return False

	P_x = numerator/denominator

	#if the point is actually not in the line segments, also return False
	if (P_x < x1 or P_x > x2 or P_x < x3 or P_x > x4):
		return False

	return P_x