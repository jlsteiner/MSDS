import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
 
    df = pd.read_csv('C:/Users/jenni/Documents/MSDS600/churn_data(1).csv', index_col='customerID')
    return df


def make_predictions(df):
   
    model = load_model('pycaret_model_js')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    
    return predictions['Churn']


if __name__ == "__main__":
    df = load_data('C:/Users/jenni/Documents/MSDS600/churn_data(1).csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
