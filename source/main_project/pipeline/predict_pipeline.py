import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging
from sklearn.preprocessing import LabelEncoder


class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        preprocessor_path = 'elements\preprocessor.pkl'
        # loaeding objects
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        data_scaled = preprocessor.transform(features)
        prediction = model.predict(data_scaled)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,
                gender,purchasedvip,income,
                children,age,attractiveness):
        self.gender = gender
        self.vip = purchasedvip
        self.income = income
        self.child = children
        self.age = age
        self.rating = attractiveness
        
    # let's write a function that returns the user input as a numpy array
    def get_data_as_df(self):
        try:
            user_data = {
                'gender':[self.gender],
                "purchasedvip":[self.vip],
                "income":[self.income],
                "children":[self.child],
                "age":[self.age],
                "attractiveness":[self.rating]
            }
            return pd.DataFrame(user_data)
        except Exception as e:
            raise UserException(e,sys)
        