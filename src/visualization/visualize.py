import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree
import logging

def plot_decision_tree(model, feature_names, file_name='tree.png'):
    try:
        plt.figure(figsize=(20,10))
        tree.plot_tree(model, feature_names=feature_names, filled=True)
        plt.savefig(file_name, dpi=300)
        plt.show()  
    
    except Exception as e:
        logging.error(" Error in plot_decision_tree model: {}". format(e)) 
        
        