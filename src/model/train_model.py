from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging, pickle



# Function to train the model
def split_data(df, X, y):
    try:
        # Splitting the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234, stratify=df['beds'])
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(" Error in splitting data: {}". format(e))

def train_linear_reg(x_train, y_train):
    try:
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model
    
    except Exception as e:
        logging.error(" Error in train linear reg model: {}". format(e))

def train_decision_tree(x_train, y_train, max_depth=3, max_features=10, random_state=567):
    try:
        model = DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, random_state=random_state)
        model.fit(x_train, y_train)
        return model
    
    except Exception as e:
        logging.error(" Error in train_decision_tree model: {}". format(e)) 
        
def train_random_forest(x_train, y_train, n_estimators=200, criterion='absolute_error'):
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
        model.fit(x_train, y_train)
        return model
    
    except Exception as e:
        logging.error(" Error in train_decision_tree model: {}". format(e))
        
def save_model(model, file_name):
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logging.error(" Error in save_model: {}". format(e))
        
def load_model(file_name):
    try:
        with open(file_name, 'rb') as file:
            return pickle.load(file)    
    except Exception as e:
        logging.error(" Error in load_model: {}". format(e))
        

        
