from flask import Flask, render_template, request, send_file, make_response
import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb
import joblib
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from data_processing import data_processing_mmao_transac, data_processing_likes, data_preprocessing_comment, merge_dataframes
from model_preprocessing import drop_columns, date_time_converting, missing_values, target_encoding, encoding, pre_processing

app = Flask(__name__)

# Load the trained models
model1 = joblib.load('models/REPEATER_LightGBM.pkl')
model2 = joblib.load('models/CLTV_lgbm.pkl')
model3 = joblib.load('models/target_encoder_repeater.pkl')
model4 = joblib.load('models/target_encoder_CLTV.pkl')
model5 = joblib.load('models/scaler_repeater.pkl')
model6 = joblib.load('models/scaler_CLTV.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load the input data
   
    data = pd.read_csv(request.files.get('file'))
    input_data = drop_columns(data)
    input_data = date_time_converting(data)
    input_data = missing_values(data)
    input_data = encoding(data)

    # set ID_BUYER as index
    input_data.set_index('ID_BUYER', inplace=True)

    cols_to_encode = ['ID_CATEGORY', 'ID_SUBCATEGORY', 'ID_BRAND', 'ID_PAYMENT_TYPE']

    # Remove the target columns
    X = input_data.drop(['REPEATER', 'CLTV'], axis=1)
    y_REPEATER = input_data['REPEATER']
    y_CLTV = input_data['CLTV']
    X_REPEATER = X_CLTV = X.copy()
    
    # Apply target encoding
   `## to Repeater data
    X_REPEATER_encoded = model3.transform(X_REPEATER[cols_to_encode])
    X_REPEATER = X_REPEATER.drop(cols_to_encode, axis=1)
    X_REPEATER = pd.concat([X_REPEATER, X_REPEATER_encoded], axis=1)
    
    ## to CLTV data
    X_CLTV_encoded = model4.transform(X_CLTV[cols_to_encode])
    X_CLTV = X_CLTV.drop(cols_to_encode, axis=1)
    X_CLTV = pd.concat([X_CLTV, X_CLTV_encoded], axis=1)

    # Scale the data
    X_REPEATER_scaled = model5.transform(X_REPEATER)
    X_CLTV_scaled = model6.transform(X_CLTV)

    # Make predictions using the trained models
    y_pred_proba = model1.predict_proba(X_REPEATER_scaled)[:,1]

    xgtest = xgb.DMatrix(X_CLTV_scaled, label=y_CLTV)
    y_pred = model2.predict(xgtest)

    # Use the predicted Repeater rate & the predicted CLTV
    weight_repeater = 0.8
    weight_CLTV = 0.2

    composite_score = weight_repeater * y_pred_proba + weight_CLTV * y_pred

    WA_df = X_REPEATER.copy()
    WA_df['Composite_Score'] = composite_score

    # normalise the composite score between 0 and 1 using MinMaxScaler
    scaler = MinMaxScaler()

    WA_df['Composite_Score'] = scaler.fit_transform(WA_df[['Composite_Score']])

    # rank the customers by their composite score
    WA_df = WA_df.sort_values(by='Composite_Score', ascending=False)
    WA_df = WA_df.reset_index()
    WA_df = WA_df[['ID_BUYER','Composite_Score']]
    # Convert WA_df to CSV format
    #csv_output = StringIO()
    WA_df.to_csv('WA_df.csv')

    # Download the CSV file
    #response = make_response(csv_output.getvalue())
    #response.headers['Content-Disposition'] = 'attachment; filename=WA_df.csv'
    #response.headers['Content-Type'] = 'text/csv'


    # Convert WA_df to HTML table
    max_value = 0.5
    filtered_df = WA_df[WA_df['Composite_Score'] >= max_value]
    table_html = filtered_df.to_html(header=False)

    return render_template('result.html', table_html=table_html, max_value=max_value)
    #return render_template('result.html', table_html=table_html)
    #return response

@app.route('/filter_result', methods=['POST'])
def filter_result():
    # Get the maximum value from the form submission
    max_value = float(request.form.get('max_value'))

    WA_df = pd.read_csv('WA_df.csv', index_col=0)
    # Filter the WA_df based on the maximum value
    filtered_df = WA_df[WA_df['Composite_Score'] >= max_value]

    # Convert the filtered dataframe to HTML table
    table_html = filtered_df.to_html(header=False)

    return render_template('result.html', table_html=table_html, max_value=max_value)
