import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split

def data_ingestion(file_path):
    train_data_path =os.path.join('artifacts','train.csv')
    test_data_path =os.path.join('artifacts','test.csv')
    file_path = os.path.join('artifacts','raw.csv')

    df=pd.read_csv(file_path)

    train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
    train_set.to_csv(train_data_path,index=False,header=True)
    test_set.to_csv(test_data_path,index=False,header=True)

    print("data saved to artifacts")

