import sys
import time

from shutil import get_terminal_size

default_color = '\033[1m\033[38;5;34m\033[48;5;0m'
default_end_color = "\033[0m\r"

#Custom progress bar
class ProgressBar():

	def __init__(self, iterable, **kwargs):

		kwargs.setdefault('message_length', get_terminal_size()[0])
		kwargs.setdefault('disp', True)
		kwargs.setdefault('freq', 1)
		kwargs.setdefault('color', default_color)
		kwargs.setdefault('end_color', default_end_color)

		self.iterable = iterable
		self.total = len(iterable)
		self.message_length = kwargs['message_length']
		self.disp = kwargs['disp']
		self.freq = kwargs['freq']
		self.color = kwargs['color']
		self.end_color = kwargs['end_color']
		
		if self.disp:

			self.update = self.Update

		else:

			self.update = self.Update_Null

	def __iter__(self):

		return PBIterator(self)

	def Update_Null(self, current, run_time):

		pass

	def Update(self, current, run_time):

		percent = float(current - 1) * 100 / self.total
		itps = current / run_time
		projrem = max([0, (self.total - current) / itps])

		str_0 = f"\r{self.color}"
		str_1 = ""
		str_3 = f" ({current-1}/{self.total}) {percent:.2f}%,"
		str_4 = f" {itps:.2f} it/s,"
		str_5 = f" {run_time:.2f} s elapsed, {projrem:.2f} s remaining"
		str_6 = self.end_color

		columns_used = len(str_1 + str_3 + str_4 + str_5)

		bar_length = self.message_length - columns_used - 5

		arrow='-'*int(percent/100*bar_length-1)+'>'
		spaces=' '*(bar_length-len(arrow))

		str_2 = f" [{arrow}{spaces}]"

		message = str_0 + str_1 + str_2 + str_3 + str_4 + str_5

		sys.stdout.write(message)
		sys.stdout.flush()

#Custom iterator for progress bar
class PBIterator():
	def __init__(self,ProgressBar):

		self.ProgressBar=ProgressBar
		self.index=0
		self.run_time=0
		self.t0=time.time()

	def __next__(self):

		if self.index<len(self.ProgressBar.iterable):

			self.index+=1
			self.run_time=time.time()-self.t0

			if self.index%self.ProgressBar.freq==0:
				self.ProgressBar.update(self.index,self.run_time)

			return self.ProgressBar.iterable[self.index-1]

		else:

			self.index+=1
			self.run_time=time.time()-self.t0

			self.ProgressBar.update(self.index,self.run_time)

			raise StopIteration