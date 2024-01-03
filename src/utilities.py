import sys
import time
import numpy as np

from shutil import get_terminal_size

'''
Calculates Gini coefficient (inequality)
'''
def GiniCoefficient(x):

	total=0

	for i,xi in enumerate(x[:-1],1):
		total+=np.sum(np.abs(xi-x[i:]))

	return total/(len(x)**2*np.mean(x))

def IsIterable(value):
	return hasattr(value,'__iter__')

def TopNIndices(array,n):
	return sorted(range(len(array)), key=lambda i: array[i])[-n:]

def BottomNIndices(array,n):
	return sorted(range(len(array)), key=lambda i: array[i])[:n]

def FullFact(levels):
	n = len(levels)  # number of factors
	nb_lines = np.prod(levels)  # number of trial conditions
	H = np.zeros((nb_lines, n))
	level_repeat = 1
	range_repeat = np.prod(levels).astype(int)
	for i in range(n):
		range_repeat /= levels[i]
		range_repeat=range_repeat.astype(int)
		lvl = []
		for j in range(levels[i]):
			lvl += [j]*level_repeat
		rng = lvl*range_repeat
		level_repeat *= levels[i]
		H[:, i] = rng
	return H.astype(int)

def Pythagorean(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2+(y1-y2)**2)

#Function for calculating distances between lon/lat pairs
def Haversine(lon1,lat1,lon2,lat2):
	r=6372800 #[m]
	dLat=np.radians(lat2-lat1)
	dLon=np.radians(lon2-lon1)
	lat1=np.radians(lat1)
	lat2=np.radians(lat2)
	a=np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c=2*np.arcsin(np.sqrt(a))
	return c*r

def RMSE(x,y):

	return np.sqrt(((x-y)**2).sum()/len(x))

def CondPrint(message,disp=True,*args,**kwargs):

	if disp:
		print(message,**kwargs)

#Custom progress bar
class ProgressBar():

	def __init__(self, iterable, message_length = None, disp = True, freq = 1):

		if message_length is None:
			message_length = get_terminal_size()[0]

		self.iterable=iterable
		self.total=len(iterable)
		self.message_length=message_length
		self.disp=disp
		self.freq=freq
		
		if self.disp:
			self.update=self.Update
		else:
			self.update=self.Update_Null

	def __iter__(self):

		return PBIterator(self)

	def Update_Null(self,current,rt):
		pass

	def Update(self,current,rt):

		percent=float(current-1)*100/self.total
		itps=current/rt
		projrem=max([0,(self.total-current)/itps])

		str_0 = "\r\033[32m "
		str_1 = "Progress"
		str_3 = f" ({current-1}/{self.total}) {percent:.2f}%,"
		str_4 = f" {itps:.2f} it/s,"
		str_5 = f" {rt:.2f} s elapsed, {projrem:.2f} s remaining"
		str_6 = "\033[0m\r"

		columns_used = len(str_1 + str_3 + str_4 + str_5)

		bar_length = self.message_length - columns_used

		arrow='-'*int(percent/100*bar_length-1)+'>'
		spaces=' '*(bar_length-len(arrow))

		str_2 = f" [{arrow}{spaces}]"

		message = str_0 + str_1 + str_2 + str_3 + str_4 + str_5 + str_6

		sys.stdout.write(message)
		sys.stdout.flush()

#Custom iterator for progress bar
class PBIterator():
	def __init__(self,ProgressBar):

		self.ProgressBar=ProgressBar
		self.index=0
		self.rt=0
		self.t0=time.time()

	def __next__(self):

		if self.index<len(self.ProgressBar.iterable):

			self.index+=1
			self.rt=time.time()-self.t0

			if self.index%self.ProgressBar.freq==0:
				self.ProgressBar.update(self.index,self.rt)

			return self.ProgressBar.iterable[self.index-1]

		else:

			self.index+=1
			self.rt=time.time()-self.t0

			self.ProgressBar.update(self.index,self.rt)

			if self.ProgressBar.disp:
				
				print('\n')

			raise StopIteration