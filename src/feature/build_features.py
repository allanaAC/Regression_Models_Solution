import pandas as pd
import logging

# 
def create_features(df):
    try:
        
        # seperate input features in x
        x = df.drop('price', axis=1)

        # store the target variable in y
        y = df['price']
        
        return x,y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))