import numpy as np
import os, re

class SNOM_File:

	def __init__(self, filename):

		self.fileName = filename
		self.name = self.fileName.split('/')[-1]

		# Header
		self.__HEADR_RAW__ = import_Header(self.fileName+'-HEADR.txt.txt')

		# Catch weird headers
		self.__HEADR_RAW__=re.sub('[Pp]ulse[_\s][Ww]idth','Pulse width=', self.__HEADR_RAW__)
		self.__HEADR_RAW__=re.sub('[Pp]re[_\s][Ss]can=','Pre scan= \n', self.__HEADR_RAW__)

		# regex matches for each row of the header
		regex = [re.findall(r'([\w\s-]+)[=\s]+([-.:/\w\d\s\µ]+)', line) for line in self.__HEADR_RAW__.split('\n')]

		# Avoid certain lines in the header
		avoid = [('Pre Scan',' '),('Laser','Parameters'),('Lock-in','Settings'),('Post Scan',' ')]

		# Do a bit of processing to the labels and filter avoided lines before storing them
		parsed_header={match[0][0].strip().replace(' ','_') : match[0][1].strip() 
		for match in regex if len(match)>0 and match[0] not in avoid}

		# Create an attribute for each of the parsed headings
		[self.__setattr__(k.lower(),v) for k,v in parsed_header.items()]

		images = dict()

		# Cycle through each image type, creating an image for each one
		for im_name in ["-FTOPO.AFM","-FSNOM.AFM","-FZERO.AFM", "-BTOPO.AFM","-BSNOM.AFM","-BZERO.AFM"]:

			try: 
				
				if 'TOPO' in im_name:

					topo = SNOM_Image(self.fileName + im_name).image

					# Invert TOPO
					images[im_name[1:-4]] = 1-topo
					#images[im_name[1:-4]] = 1-(SNOM_Image(self.fileName + im_name).image)


				else:
					images[im_name[1:-4]] = SNOM_Image(self.fileName + im_name).image
					
			except: continue

		[self.__setattr__(k,v) for k,v in images.items()]


class SNOM_Image:

	def __init__(self, image_file_name):

		with open(image_file_name, mode = 'rb') as file:

			data = np.fromfile(file, np.int16)

		self.dimensions = (data[0],data[0])

		self.image = np.reshape(data[3:-2], (self.dimensions[0], self.dimensions[1]))
		self.image = np.rot90(self.image)

def plane_correct(image):
	'''
	Performs plan tilt correction on image.
	Returns corrected image.
	'''

	m, n = image.shape

	X1, X2 = np.mgrid[:m, :n]

	X = np.hstack((np.reshape(X1, (m*n, 1)) , np.reshape(X2, (m*n, 1)) ) )
	X = np.hstack((np.ones((m*n, 1)) , X ))
	YY = np.reshape(image, (m*n, 1))

	theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

	plane = np.reshape(np.dot(X, theta), (m, n));
	Y_corr = image-plane

	return Y_corr



def import_Header(image_file_name):

	with open(image_file_name, mode = "r", errors='ignore') as file:

		return file.read()

	try:

		with open(image_file_name, mode = "r") as file:
	
			return file.read()

	# To catch weirdly formatted file names
	except:

		with open(image_file_name + ".txt", encoding = "cp1252", mode = "r") as file:

			return file.read()