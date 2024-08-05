#this function is to load data 
import pandas as pd
import numpy as np

import logging

data_path = "src/data/final.csv"
def load_and_preprocess_data(data_path):
    
    try:
        
        # Import the data from 'age_salary.csv'
        df = pd.read_csv(data_path)
        print(df.head())
       
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

""" if __name__=="__main__":
    df = load_and_preprocess_data(data_path)   
    print(df.head())      """