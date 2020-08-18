#-----------------------------------------------------------------------------
# Name:        fitfun.py
# Purpose:     
#
# Author:      Roger Jarvis
#
# Created:     2006/06/20
# RCS-ID:      $Id: fitfun.py,v 1.6 2009/01/13 16:26:55 rmj01 Exp $
# Copyright:   (c) 2006
# Licence:     GNU General Public License
# Description: Fitness functions for use in genetic algorithm optimisation
#-----------------------------------------------------------------------------

import string, copy, scipy
from process import *
from chemometrics import *
from chemometrics import _split
from chemometrics import _slice
from chemometrics import _index
from chemometrics import _diag
from chemometrics import _put
from chemometrics import _flip
from chemometrics import _BW
from genetic import _remdup
from expSetup import valSplit
from scipy import newaxis as nA

def _group(x,mrep):
    grp = []
    for n in range(1,x.shape[0]/mrep+1,1):
        for cnt in range(0,mrep,1):
            grp.append(n)
    return scipy.reshape(scipy.asarray(grp,'i'),(len(grp),1))

def call_dfa(chrom,xdata,DFs,mask,data):
    """Runs DFA on subset of variables from "xdata" as 
    defined by "chrom" and returns a vector of fitness 
    scores to be fed back into the GA
    """
    Y = []
    for x in range(len(chrom)):
        if _remdup(chrom[x]) == 0:
            #extract vars from xdata
            slice = meancent(_slice(xdata,chrom[x]))
            collate = 0
            for nF in range(mask.shape[1]):
                #split in to training and test
                tr_slice,cv_slice,ts_slice,tr_grp,cv_grp,ts_grp,tr_nm,cv_nm,ts_nm=_split(slice,
                      data['class'][:,0],mask[:,nF].tolist(),data['label'])
                
                try:
                    u,v,eigs,dummy = cva(tr_slice,tr_grp,DFs)
                    projU = scipy.dot(cv_slice,v)
                    u = scipy.concatenate((u,projU),0)
                    group2 = scipy.concatenate((tr_grp,cv_grp),0)
            
                    B,W = _BW(u,group2)
                    L,A = scipy.linalg.eig(B,W)
                    order =  _flip(scipy.argsort(scipy.reshape(L.real,(len(L),))))
                    Ls =  _flip(scipy.sort(L.real))
                    eigval = Ls[0:DFs]
                    
                    collate += sum(eigval)
                except:
                    continue
                
            if collate != 0:
                Y.append(float(mask.shape[1])/collate)
            else:
                Y.append(10.0**5)
        else:
            Y.append(10.0**5)
            
    return scipy.array(Y)[:,nA]        


def rerun_dfa(chrom,xdata,mask,groups,names,DFs):
    """Run DFA in min app"""
    #extract vars from xdata
    slice = meancent(_slice(xdata,chrom))
    
    #split in to training and test
    tr_slice,cv_slice,ts_slice,tr_grp,cv_grp,ts_grp,tr_nm,cv_nm,ts_nm=_split(slice,groups,mask,names)
    
    #get indexes
    idx = scipy.arange(xdata.shape[0])[:,nA]
    tr_idx = scipy.take(idx,_index(mask,0),0)
    cv_idx = scipy.take(idx,_index(mask,1),0)
    ts_idx = scipy.take(idx,_index(mask,2),0)
    
    #model DFA on training samples
    u,v,eigs,dummy = cva(tr_slice,tr_grp,DFs)
    
    #project xval and test samples
    projUcv = scipy.dot(cv_slice,v)
    projUt = scipy.dot(ts_slice,v)
    
    uout = scipy.zeros((xdata.shape[0],DFs),'d')
    _put(uout,scipy.reshape(tr_idx,(len(tr_idx),)).tolist(),u)
    _put(uout,scipy.reshape(cv_idx,(len(cv_idx),)).tolist(),projUcv)
    _put(uout,scipy.reshape(ts_idx,(len(ts_idx),)).tolist(),projUt)
    
    return uout,v,eigs      


def call_pls(chrom,xdata,factors,mask,data):
    """Runs pls on a subset of X-variables"""
    scores = []
    
    for i in range(chrom.shape[0]):
        if _remdup(chrom[i]) == 0:
            #extract vars from xdata
            slice = scipy.take(xdata,chrom[i,:].tolist(),1)
            collate = 0
            for nF in range(mask.shape[1]):
                #split in to training and test
                try:
                    pls_output = pls(slice,data['class'][:,0][:,nA],mask[:,nF].tolist(),factors)
                    
                    if min(pls_output['rmsec']) <= min(pls_output['rmsepc']):
                        collate += pls_output['RMSEPC']
                    else:
                        collate += 10.0**5
                except:
                    collate = 0
                
            if collate != 0:
                scores.append(collate/float(mask.shape[1]))
            else:
                scores.append(10.0**5)
        else:
            scores.append(10.0**5)
            
    return scipy.asarray(scores)[:,nA]    

def rerun_pls(chrom,xdata,groups,mask,factors):
    """rerun pls on a subset of X-variables"""
    
    slice = scipy.take(xdata,chrom,1)
    
    return pls(slice,groups,mask,factors)

if __name__=="__main__":
    import fitfun,doctest
    doctest.testmod(fitfun,verbose=True)
