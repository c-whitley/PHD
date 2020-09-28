import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# Library of all methods
transformers = {
    'Binning': {'Mean': FunctionTransofmer(binningmethod)},
    'Normalising': {'Vector' : SklearnScaler, 'MinMax' : .}
}


class Pipeline_Creator:

    def __init__(self):

        pass

    def make_pipeline(self, address):

        pipeline_list=[]

        for k, v in address.items():

            transformer = [v[0]]
            transformer.set_attr(v[1])

            pipeline_list.append(transformer)

        return Pipeline(pipeline_list)


class trial:

    def __init__(self,  pipeline):

        self.data
        self.pipeline = pipeline

    def run_trial(self):
        
        self.score = score(self.pipeline)