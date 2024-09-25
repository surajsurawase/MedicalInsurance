import pickle 
import json
import numpy as np
import config
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class MedicalInsurance():
    def __init__(self):
        pass

    def load_data(self):
        with open(config.MODEL_FILE_NAME,'rb') as f:
            self.model = pickle.load(f)

        with open(config.COLUMN_DATA_JSON, 'r') as f:
            self.column_data = json.load(f)

        self.column_names = self.model.feature_names_in_
        self.features_count = self.model.n_features_in_

    def get_predicted_charges(self, age,gender,bmi,children,smoker,region):
    
        self.load_data()
        gender = self.column_data['gender'][gender]
        smoker = self.column_data['smoker'][smoker]
        
        region = "region_" + region
        
        region_index = np.where(self.column_names == region)[0][0]

        test_array = np.zeros((1,self.features_count))
        test_array[0,0] = age
        test_array[0,1] = gender
        test_array[0,2] = bmi
        test_array[0,3] = children
        test_array[0,4] = smoker
        test_array[0,region_index] = 1
        predicted_charges = np.around(self.model.predict(test_array)[0],3)
        print("Predicted Charges :",predicted_charges)
        
        return predicted_charges
    
def load_dataset():
    df = pd.read_csv(config.CSV_FILE_NAME)
    # print(df)
    return df


if __name__ == "__main__":
    ins = MedicalInsurance()
    ins.load_data()
    print(list(ins.column_data['smoker'].keys()))