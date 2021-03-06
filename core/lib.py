"""
    Helper Functions to Store and Retrieve Data and Pickle Data
"""
import os
import pickle
from datetime import date
from collections import Counter
import nltk
import deepdish as dd
from pycocotools.coco import COCO
from scripts.build_vocab import Vocabulary

def __getFilePath(iternumber,date):
    filename = 'models/'+date[1]+'_'+date[0]+'_'+date[2]+'_'+str(iternumber)+'.model'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(filename) and os.path.isfile(filename):
        return __getFilePath(iternumber+1,date)
    return filename

def saveModel(parameters):
    iternumber = 1
    date_today = str(date.today()).split('-')
    filename = __getFilePath(iternumber,date_today)
    fptr = open(filename,'wb')
    pickle.dump(parameters,fptr)
    fptr.close()

def loadData(filepath):
    data = dd.io.load(filepath)
    return data

def loadModel(filepath):
    fptr = open(filepath,'rb')
    param = pickle.load(fptr)
    fptr.close()
    return param
