import numpy as np
import os, re

class SNOM_File:

	def __init__(self, filename):

		self.fileName = filename

		# Header
		self.__HEADR_RAW__ = import_Header(self.fileName+'-HEADR.txt.txt')

		# regex matches for each row of the header
		regex = [re.findall(r'([\w\s-]+)[=\s]+([-.:/\w\d\s\µ]+)', line) for line in self.__HEADR_RAW__.split('\n')]

		# Avoid certain lines in the header
		avoid = [('Pre Scan',' '),('Laser','Parameters'),('Lock-in','settings'),('Post Scan',' ')]

		# Do a bit of processing to the labels and filter avoided lines before storing them
		parsed_header={match[0][0].strip().replace(' ','_') : match[0][1].strip() 
		for match in regex if len(match)>0 and match[0] not in avoid}

		# Create an attribute for each of the parsed headings
		[self.__setattr__(k,v) for k,v in parsed_header.items()]

		images = dict()

		# Cycle through each image type, creating an image for each one
		for im_name in ["-FTOPO.AFM","-FSNOM.AFM","-FZERO.AFM", "-BTOPO.AFM","-BSNOM.AFM","-BZERO.AFM"]:

			try: 
				
				if 'TOPO' in im_name:

					# Invert TOPO
					images[im_name[1:-4]] = np.abs(SNOM_Image(self.fileName + im_name).image)

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