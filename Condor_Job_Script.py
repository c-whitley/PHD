import numpy as np
import pandas as pd 
import shelve
import os
import datetime
from tqdm.notebook import tqdm
import pysftp
import glob

from scipy.io import loadmat
from scipy.stats import mode
from spectral.io import envi

from skimage.filters import threshold_mean


def pd_split(df, n_each):

	if n_each == 1:

		return [df]
    
	cuts = np.arange(0, df.shape[0], n_each)
	cuts = np.append(cuts, df.shape[0])

	return [df.iloc[cuts[i]: cuts[i+1], :] for i in range(len(cuts) -1)]

def get_mask(im):

    return np.stack([im[:,:,i] > threshold_mean(im[:,:,i]) for i in [370]])


class Condor_Job:

	# Takes an iterable containing the 
	def __init__(self, func, whole_data, job_name):

		# Set up a few variables to do with the job
		self.job_name = job_name
		self.function = func
		self.initialised = False
		self.whole_data = whole_data

		print(f"Job created: {self.job_name}")

				# Create the place to save the input and output data
		if self.job_name == None:

			self.job_name = str(datetime.datetime.now())[:-7]


		self.file_name = os.path.join("/mnt/h/", self.job_name)

		if not os.path.exists(self.file_name):

			os.mkdir(self.file_name)

		print(f"Directory: {self.file_name}")


	def prepareB(self, sub_split):

		i = 0

		# Given the iterable provided go through each entry and create a file associated with that split
		for subset in tqdm(self.whole_data, total  = len(list(self.whole_data))):


			file = loadmat(subset)

			im = file["data"]
			wn = file["wn"].squeeze()

			XX,YY = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))


			subset_df = pd.DataFrame(im.reshape(-1, len(wn))
				, columns = np.array(wn, dtype = np.float))

			min_, max_ = np.argmin(abs(1000-subset_df.columns)) , np.argmin(abs(3800-subset_df.columns))
			
			# Clip the range of wavelenths used
			subset_df = subset_df.iloc[:,min_:max_]


			subset_df["X"] = XX.ravel()
			subset_df["Y"] = YY.ravel()
			#subset_df["Disk"] = subset.split("/")[-1][:-4]
			subset_df["TMA"] = subset.split("/")[-2]

			subset_df.set_index(["X","Y","TMA"], inplace = True)


			if not os.path.exists(os.path.join(self.file_name, "input")):

				os.mkdir(os.path.join(self.file_name, "input"))


			# If the file is to be split up into smaller pieces this loop will do it.
			for subset_piece in pd_split(subset_df, sub_split):

				subset_file_name = os.path.join(self.file_name, "input", "input{}".format(i))

				subset_piece.to_pickle(subset_file_name)


				# Open a file and then store the data inside
				with shelve.open(os.path.join(self.file_name, "input", "input_ref_{}".format(i)), "n") as shelf:

					shelf["mean_ref"] = subset_df.mean(axis = 0).values

				i +=1
				

		self.n_jobs = i
		self.initialised = True


	def prepare(self, sub_split):

		i = 0

		# Given the iterable provided go through each entry and create a file associated with that split
		for subset in tqdm(self.whole_data, total  = len(list(self.whole_data))):

			im = envi.open(subset)
			wn = im.metadata["wn"]
			im = im.load()


			XX,YY = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))


			subset_df = pd.DataFrame(im.reshape(-1, len(wn))
				, columns = np.array(wn, dtype = np.float))

			min_, max_ = np.argmin(abs(1000-subset_df.columns)) , np.argmin(abs(3800-subset_df.columns))
			
			# Clip the range of wavelenths used
			subset_df = subset_df.iloc[:,min_:max_]


			subset_df["X"] = XX.ravel()
			subset_df["Y"] = YY.ravel()
			subset_df["Disk"] = subset.split("/")[-1][:-4]
			subset_df["TMA"] = subset.split("/")[-2]

			subset_df.set_index(["X","Y","TMA", "Disk"], inplace = True)


			if not os.path.exists(os.path.join(self.file_name, "input")):

				os.mkdir(os.path.join(self.file_name, "input"))


			# If the file is to be split up into smaller pieces this loop will do it.
			for subset_piece in pd_split(subset_df, sub_split):

				subset_file_name = os.path.join(self.file_name, "input", "input{}".format(i))

				subset_piece.to_pickle(subset_file_name)


				# Open a file and then store the data inside
				with shelve.open(os.path.join(self.file_name, "input", "input_ref_{}".format(i)), "n") as shelf:

					#shelf["mean_ref"] = subset_df.mean(axis = 0).values

					im = subset_df.values.reshape(im.shape[0], im.shape[1], -1)

					test_im = im.reshape(-1, max(im.shape))
					masks = get_mask(im)

					# Only tissue parts of image remain
					tissue = im[mode(masks, axis = 0)[0].squeeze(), :]
					shelf["mean_ref"] = pd.DataFrame(tissue, columns = subset_df.columns).mean(axis = 0)

				i +=1
				

		self.n_jobs = i
		self.initialised = True


	def submit(self, **kwargs):

		self.submission_file()
		print("Submission file created")


		if self.initialised:

			username = kwargs.get("username", input("Username:"))
			password = kwargs.get("password", input("Password:"))

			with pysftp.Connection("condor.liv.ac.uk",
	                       username = username,
	                       password = password) as sftp:
	    
			    try:
			    	sftp.mkdir("/condor_data/sgcwhitl/data/{}".format(self.job_name))

			    except: pass

			    print("Uploading file segments")

			    if not os.path.exists(os.getcwd() + "/Condor_Dependencies"):
			    	os.mkdir(os.getcwd() + "/Condor_Dependencies")

			    file_names = glob.glob(self.file_name + "/input/*") + glob.glob(os.getcwd() + "/Condor_Dependencies/*")

			    for local_file_name in tqdm(file_names):

			    	#print(local_file_name)

			    	with sftp.cd("/condor_data/sgcwhitl/data/{}".format(self.job_name)):

				    	sftp.put(local_file_name
				    	, confirm = True)

		else:

			print("Not initialised")

			raise


	def retrieve(self, **kwargs):

		username = kwargs.get("username", input("Username:"))
		password = kwargs.get("password", input("Password:"))


		with pysftp.Connection("condor.liv.ac.uk",
                       username = username,
                       password = password) as sftp:
    

			with sftp.cd("/condor_data/sgcwhitl/data/{}".format(self.job_name)):

				print("Downloading files")

				if not os.path.exists(os.path.join(self.file_name + "/output/")):
					os.mkdir(os.path.join(self.file_name + "/output/"))


				for file_name in tqdm([file_name for file_name in sftp.listdir() if "output" in file_name]):

					sftp.get(file_name, os.path.join(self.file_name, "output", file_name))


	def combine(self, **kwargs):

		file_list = glob.glob(self.file_name + "/output/*")

		file_list = sorted(file_list, key = lambda FN: int(FN.split("/")[-1][6:]))

		i = 0
		set_ = []


		while True:


			if i > len(file_list):
				break
			

			try: 
				set_.append(pd.read_pickle(file_list[i]))

			except:

				break

			if len(np.unique([s.size for s in set_])) > 1:

				full = pd.concat(set_)

				set_ = []

				yield full

			i += 1


	def submission_file(self):

		# Generates submission file

		with open(os.path.join(self.file_name, "input/run_product"), "w") as file:

			file.write(f"python_script = {self.function}\n")
			file.write(f"python_version = python_3.7.4\n")
			file.write("indexed_input_files = input, input_ref_.dat, input_ref_.dir, input_ref_.bak\n")
			file.write("indexed_output_files = output\n")
			file.write("indexed_stdout = stdout.txt\n")
			file.write("indexed_stderr = stderr.txt\n")
			file.write("log = log.txt\n")
			file.write(f"total_jobs = {self.n_jobs}\n")