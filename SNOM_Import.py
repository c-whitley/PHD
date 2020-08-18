import numpy as np
import os, re

class SNOM_File:

	def __init__(self, filename):

		self.fileName = filename

		# Header

		self.HEADR_RAW = import_Header(self.fileName)
		self.HEADR = self


		#try: self.HEADR = import_Header(self.fileName) + "-HEADR.txt")
		#except: self.HEADR = import_Header(self.fileName + "-HEADR.txt.txt")

		#self.sensitivity = float(re.search(r"[Ss]ensitivity =\s*([^\n\r\t\s]{1,5})" , self.HEADR)[1])
		#self.wavenumber = float(re.search(r"[Ww]avenumber =\s*([^\n\r\t\s]{1,5})" , self.HEADR)[1])

		print(self.fileName)

		#try: self.pH = float(re.search(r"p[Hh]([\d])" , self.HEADR)[1])	
		#except: pass

		
		#self.substrate = re.search(r"[Oo]n\s*([^\s]{1,6})" , self.HEADR)[1]

		self.images = dict()

		for im_name in ["-FTOPO.AFM","-FSNOM.AFM","-FZERO.AFM", "-BTOPO.AFM","-BSNOM.AFM","-BZERO.AFM"]:

			try: 

				self.images[im_name[1:-4]] = SNOM_Image(self.fileName + im_name).image

			except:

				#print("Failed to load: {}".format(self.fileName + im_name))


				continue


class SNOM_Image:

	def __init__(self, image_file_name):

		with open(image_file_name, mode = 'rb') as file:

			data = np.fromfile(file, np.int16)

		self.dimensions = (data[0],data[0])

		self.image = np.reshape(data[3:-2], (self.dimensions[0], self.dimensions[1]))


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

