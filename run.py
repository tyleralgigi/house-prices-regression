import pandas as pd
from BorutaShap import BorutaShap
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import os 
from BorutaShap import BorutaShap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import seaborn as sns
from scipy import stats


dir_path = os.path.dirname(os.path.realpath(__file__))

def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding '+feature)
    return df

 
def main():
    #pd.set_option("display.max_rows", None, "display.max_columns", None)
    #importing data
    test_data = pd.read_csv(dir_path+"/test.csv")
    train_data = pd.read_csv(dir_path+"/train.csv")

    #dropping empty columns
    train_data = train_data.drop(columns=['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature','LotFrontage', 'GarageYrBlt'])
    test_data = test_data.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature','LotFrontage', 'GarageYrBlt'])
    test_data_id = test_data.pop('Id')

    #encoded string values to int
    train_data = Encoder(train_data)
    
    #remove outliers
    train_data = train_data.mask((train_data - train_data.mean()).abs() > 2 * train_data.std()).dropna()

    #filling in NAN values in data frame and splitting dataframe into x and y
    train_data = train_data.fillna(train_data.mean())
    y = train_data.pop('SalePrice')
    X = train_data

    Feature_Selector = BorutaShap(importance_measure='shap', classification=False)

    Feature_Selector.fit(X=X, y=y, n_trials=50, random_state=0)
    Feature_Selector.plot(which_features='all', figsize=(16,12))

    train_data_subset = Feature_Selector.Subset()

    X = train_data_subset

    #Split data into test and train
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

    model = Lasso(alpha=0.99)
    # fit model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)

    #encoded string values to int
    test_data = Encoder(test_data)
    #filling in NAN values in data frame and splitting dataframe into x and y
    test_data = test_data.fillna(test_data.mean())
    X_test = test_data
    
    pred = model.predict(X_test[X_train.columns.tolist()])
    pred_df = pd.DataFrame(pred)

    submission = pd.concat([test_data_id, pred_df], axis=1)
    submission.columns = ["Id", "SalePrice"]
    submission.to_csv("submission.csv", index=False)

main()