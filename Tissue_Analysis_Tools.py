import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import label_binarize

import random
import numbers
from sklearn.decomposition import PCA

import matplotlib as mpl

font = {'family' : 'normal',
        'size'	 : 12}

mpl.rc('font', **font)


def h5py_import(file_name, data_name, labels):

    with h5py.File(file_name, "r") as f:

        data = f[data_name]
        
        data = [f[data[tissue_n][0]].value for tissue_n in range(f[data_name].shape[0])]

        data = [pd.DataFrame(tissue_data, index = np.round(data[-1].ravel(), 0)).T for tissue_data in data[:-1]]

        # Assign the labels to each dataframe.
        for i in range(len(data)): 

            data[i]["Label"] = labels[i]
            data[i]["Number"] = data[i].index

        dataframe = pd.concat(data)

        dataframe.set_index(["Label","Number"], inplace = True)

    return dataframe

def cleandata(data, std = 0.85, iterations = 5, sigma = 3):
    output = data
    
    #input a list of ratios, representing the ratio for a given metric at each pixel.
    for _ in range(iterations):
    
        datastd = output.std()
        output = output[output-output.mean() < sigma*datastd]
        
        if output.std()/datastd > std: break
    
    nremoved = len(data)-len(output)	

    return (output, nremoved)
    
def calculate_stats(predictions, test_labels, probabilities = None,	 n_ROC_curve_points = 100):
    
    y = label_binarize(test_labels, classes = np.unique(test_labels))
    predictions = label_binarize(predictions, classes = np.unique(test_labels))
            
    results = {}
    
    # AUC
    results["AUC"] = roc_auc_score(y, predictions, average = "weighted")

    # Accuracy:
    results["accuracy"] = accuracy_score(y, predictions)

    # Confusion Matrix:
    tn, fp, fn, tp =  confusion_matrix(y, predictions).ravel()

    # Sensitivity
    results["Sensitivity"] = tp/(tp+fn)

    # Specificity
    results["Specificity"] = tn/(tn+fp)

    # F1 Score
    results["F1"] = (2*tp)/(2*tp+fp+fn)

    if probabilities != None:
        
        # ROC curves
        ROC_curve = roc_curve(y, probabilities)

        fpr, tpr, thresholds = ROC_curve

        new_x = np.linspace(min(fpr), max(fpr), n_ROC_curve_points)
        ROC_curve = np.interp(new_x, fpr, tpr)
        
        results["ROC_Curve"] = ROC_curve
        
    return results

def multiclass_kfold_roc_curve(clf, X, y, cv_n = 3, n_ROC_curve_points = 100):
    
    n_classes = len(np.unique(y))
    
    if n_classes == 2:
        n_classes=1
    
    class_names = np.unique(y)
    
    k_split = StratifiedKFold(n_splits = cv_n, random_state = random.randint(0,1E10))
    
    #Binarise the labels
    y = label_binarize(y, classes = class_names)

    # Create onevsall classifier
    OVA_clf = OneVsRestClassifier(clf)
    
    # Arrays to store each statistic
    AUCs = np.zeros([cv_n,n_classes])
    accuracies = np.zeros([cv_n,n_classes])
    Sens = np.zeros([cv_n,n_classes])
    Specs = np.zeros([cv_n,n_classes])
    F1 = np.zeros([cv_n,n_classes])
    ROC_curves = [[] for _ in range(n_classes)]
    
    if n_classes == 1: ROC_curves = []
    
    
    # Iterate over each class in the onevsrest classifier
    for i in range(n_classes):
        
        # To keep track of repeats to store results in array
        k = 0
        
        # Repeat over each class to obtain repeats
        for train_data, test_data in k_split.split(X, y[:,i]):
            
            OVA_clf.fit(X[train_data], y[:,i][train_data])
            
            predictions = OVA_clf.predict(X[test_data])
            
            # AUC
            AUC = roc_auc_score(y[:,i][test_data],predictions, average = "weighted")
            AUCs[k,i] = AUC

            # Accuracy:
            accuracy = accuracy_score(y[:,i][test_data], predictions)
            accuracies[k,i] = accuracy
            
            # Confusion Matrix:
            CF_mat = confusion_matrix(y[:,i][test_data], predictions)
            tn, fp, fn, tp =  confusion_matrix(y[:,i][test_data], predictions).ravel()
            
            # Sensitivity
            Sens[k,i] = tp/(tp+fn)
            
            # Specificity
            Specs[k,i] = tn/(tn+fp)
            
            # F1 Score
            F1[k,i] = (2*tp)/(2*tp+fp+fn)
            
            # ROC curves
            
            ROC_curve = roc_curve(y[:,i][test_data], OVA_clf.predict_proba(X[test_data])[:,1])
            
            fpr, tpr, thresholds = ROC_curve
            
            new_x = np.linspace(min(fpr), max(fpr), n_ROC_curve_points)
            ROC_curve = np.interp(new_x, fpr, tpr)

            try:
                ROC_curves[i].append(ROC_curve)
                
            except:
                ROC_curves.append(ROC_curve)
            
            
            k+=1
    
    if n_classes == 1:

            AUC_DF = pd.Series(AUCs.ravel(), name = class_names[0]).T
            accuracy_DF = pd.Series(accuracies.ravel(), name = class_names[0]).T
            Sensitivity_DF = pd.Series(Sens.ravel(), name = class_names[0]).T
            Specificity_DF = pd.Series(Specs.ravel(), name = class_names[0]).T
            F1_DF = pd.Series(F1.ravel(), name = class_names[0]).T
            
            ROC_curves_DF = pd.DataFrame(ROC_curves)
            
            return {"AUC_DF": AUC_DF, 
                "accuracy_DF": accuracy_DF, 
                "Sensitivity_DF": Sensitivity_DF, 
                "Specificity_DF": Specificity_DF, 
                "F1_DF": F1_DF,
                "ROC_curves_DF": ROC_curves_DF}
    
    
    AUC_DF = pd.DataFrame(AUCs, columns = class_names).T
    accuracy_DF = pd.DataFrame(accuracies, columns = class_names).T
    Sensitivity_DF = pd.DataFrame(Sens, columns = class_names).T
    Specificity_DF = pd.DataFrame(Specs, columns = class_names).T
    F1_DF = pd.DataFrame(F1, columns = class_names).T
    
    ROC_curves_DF = pd.DataFrame(ROC_curves, index = class_names).T
            
    return {"AUC_DF": AUC_DF, 
            "accuracy_DF": accuracy_DF, 
            "Sensitivity_DF": Sensitivity_DF, 
            "Specificity_DF": Specificity_DF, 
            "F1_DF": F1_DF,
            "ROC_curves_DF": ROC_curves_DF}

def plot_ROC_Curves(ROC_curves_DF, colour_dict, title = None, save_place = None, AUCs_DF = None):
    
    means = [ROC_curves_DF["{}".format(label)].values.mean(axis = 0) for label in colour_dict]
    stds = [ROC_curves_DF["{}".format(label)].values.std(axis = 0)/
        np.sqrt(len(output["ROC_curves_DF"]["{}".format(label)].values.std(axis = 0)))
        for label in colour_dict]

    plt.clf()
    plt.figure(figsize = (8, 5))

    for i in range(4):

        tissue_name = "{}".format(list(colour_dict.keys())[i])

        colour = colour_dict[tissue_name]

        if AUCs_DF is not None: 
            
            mean_AUC = AUCs_DF.mean(axis = 1)["{}".format(tissue_name)]
            std_AUC = AUCs_DF.std(axis = 1)["{}".format(tissue_name)]
            
            plt.plot(np.linspace(0,1,100), means[i], c = colour,
                     label = "{}, AUC: {} +/- {}".format(tissue_name, round(mean_AUC,3), round(std_AUC,3)))
            
        else: plt.plot(np.linspace(0,1,100), means[i], c = colour, label = tissue_name)

        plt.plot(np.linspace(0,1,100), means[i] - stds[i], c = colour, linestyle = ":")
        plt.plot(np.linspace(0,1,100), means[i] + stds[i], c = colour, linestyle = ":")

        plt.legend()
        plt.ylabel("Sensitivity")
        plt.xlabel("1 - Specificity")
        
        if save_place != None:
            
            plt.savefig(r"{}\{}_ROC_Curve".format(save_place, title), pad_inches	 = 0)

    plt.show()

# Pointless, use pd.map
def reassign(dataframe, reassignment_list):
    
    for old_label, new_label in list(reassignment_list):
        
        indices = dataframe["Label"] == "{}".format(old_label)
        
        dataframe.loc[indices.values,"New_label"] = new_label
        
    return dataframe#.drop("index", axis = 1)

def find_wn(n, wavenumber_list):
    
    wavenumbers = list(wavenumber_list)
    
    distances = [np.abs(n-int(wn)) for wn in wavenumbers if isinstance(wn, numbers.Number)]
    
    return np.argmin(distances)
    
def PCA_denoise(input_spectra, n_components):
    
    pca = PCA(n_components)
    
    pca_values = pca.fit(input_spectra)
    
    filtered = np.dot(pca.transform(input_spectra)[:,:n_components], pca.components_[:n_components,:])

    values = np.add(filtered, np.mean(input_spectra, axis = 0).values.reshape(1,-1))

    return pd.DataFrame(values, columns = input_spectra.columns, index = input_spectra.index)

def process_data(input_dataframe, start = 1000, end = 1800, paraffin = (1340,1490), balance = False):

    # Find the indices which the start and end positions are closest to.
    start, end = [find_wn(n, input_dataframe.columns) for n in [start,end]]
    
    input_dataframe = input_dataframe.iloc[:,start: end + 1]
    
    if isinstance(paraffin, (tuple, list)):# or isinstance(paraffin, list):
        
        paraffin_index = [find_wn(n, input_dataframe.columns) for n in paraffin]
        
        input_dataframe.drop(input_dataframe.iloc[:,paraffin_index[0]:paraffin_index[1]].columns, axis = 1, inplace = True)
        
    if isinstance(balance, str):
    
        assert balance in input_dataframe.index.names, "Balance label not in dataframe index."
        
        original_index = input_dataframe.index.names

        #for name,df in groups:
        
        #	df.sample(groups.size().min())
        
        #input_dataframe = pd.DataFrame(groups.apply(lambda x: x.sample(groups.size().min())).values, columns = input_dataframe.columns, index = input_dataframe.index)
        
        groups = input_dataframe.reset_index().groupby(balance)
         
        input_dataframe = groups.apply(lambda x: x.sample(groups.size().min())).set_index(original_index)
        
    return input_dataframe

def bin_predict(x_array, classifiers):
    
    content = {name: classifier.predict_proba(x_array)[:,1] for name, classifier in classifiers.items()}
    output = pd.DataFrame(content)
    
    # Normalise probability
    output = output.div(output.sum(axis = 1), axis = 0)

    return output

def sens_spec_etc(y_true, y_pred):
    
    #assert len(np.unique(y_true)) == len(np.unique(y_pred)), "True and predicted labels have different classes present"

    total = dict()
    
    for i, name in enumerate(np.unique(y_true)):
        
        stats = dict()
        #todo add assert statement to ensure at least 1 example of each class
        
        conf_mat = confusion_matrix(y_true, y_pred)

        stats["TP"] = conf_mat[i,i].astype(int)
        stats["FP"] = (np.sum(conf_mat[:,i]) - stats["TP"]).astype(int)
        stats["FN"] = (np.sum(conf_mat[i,:]) - stats["TP"]).astype(int)
        stats["TN"] = (np.sum(conf_mat) - stats["TP"] - stats["FP"] - stats["FN"]).astype(int)
        
        stats["Sensitivity"] = stats["TP"]/(stats["TP"] + stats["FN"])
        stats["Specficity"] = stats["TN"]/(stats["TN"] + stats["FP"])
        stats["F1"] = 2*stats["TP"]/(2*stats["TP"] + stats["FN"] + stats["FP"])
        stats["J"] = stats["Sensitivity"] + stats["Specficity"] - 1

        total[name] = stats

    return pd.DataFrame(total)

def ensemble_prob_predict(X, clf_dict, **kwargs):
    
    votes = {clf_name: pd.DataFrame(bin_predict(X.values, clf)) for clf_name, clf in clf_dict.items()}

    n_classes, n_clfs = pd.DataFrame(clf_dict).shape

    print(pd.DataFrame(clf_dict).column)


    # Try to retrieve weights from kwargs, otherwise use equal weighting
    weights = kwargs.get("weights", np.array([[1/n_clfs for _ in range(n_classes)] for _ in range(n_clfs)]))

    print(weights.shape)

    stacked = np.stack([df.values for df in votes.values()], axis = 2)
    split = np.split(stacked, stacked.shape[1],axis = 1)

    #{cls_: np.tensordot(split[i], weights[:,i]), axes = [1,0] for cls_, split in enumerate(z)}

    print(split.shape)

    weighted = np.tensordot(np.stack([df.values for df in votes.values()], axis = 2), weights, axes = ([2,1], [1,0]))

    print(weighted.shape)

    # Normalise probability
    output = pd.DataFrame(weighted, columns = pd.concat(votes).columns)

    return output.div(output.sum(axis = 1), axis = 0)

from sklearn.base import BaseEstimator, TransformerMixin

class PCA_denoiser( BaseEstimator, TransformerMixin ):

    #Class Constructor 
    def __init__( self, n_components):

        self.n_components = n_components 

    
    #Return self nothing else to do here    
    def transform( self, X):

        pca_values = self.denoiser.fit(X)
        
        filtered = np.dot(self.denoiser.transform(X)[:,:self.n_components], self.denoiser.components_[:self.n_components,:])

        values = np.add(filtered, np.mean(X, axis = 0).reshape(1,-1))

        return values

    
    #Method that describes what we need this transformer to do
    def fit( self, X, y = None ):

        self.denoiser = PCA(self.n_components)

        return self

from scipy.spatial import ConvexHull

class Rubber_Band( BaseEstimator, TransformerMixin ):


    #Class Constructor 
    def __init__(self):

        pass
    
    #Return self nothing else to do here    
    def transform(self, x):

        return self.y - self.baseline
    
    #Method that describes what we need this transformer to do
    def fit( self, x, y):

        if type(y) == pd.DataFrame: y = y.values

        self.x, self.y = x, y

        points = np.column_stack([self.x, self.y])

        self.v = ConvexHull(points).vertices

        # Rotate convex hull vertices until they start from the lowest one
        self.v = np.roll(self.v, -self.v.argmin())
        # Leave only the ascending part
        self.v = self.v[:self.v.argmax()]

        self.baseline = np.interp(self.x, self.x[self.v], self.y[self.v])

        return sel

def perc_plot(im, lower_perc = 1, upper_perc = 99):
    
    upper, lower = np.percentile(im.flatten()
                                 , [upper_perc , lower_perc ])

    return plt.imshow(im, vmin = lower, vmax = upper)