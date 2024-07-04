import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import base64
import tempfile
import os

# Load the pre-trained model
model = load_model('model.h5')

# Function to create sequences
def predict_missing_values(df, look_back):
    features = [
        '80m Avg [m/s]', '50m Avg [m/s]', '20m Avg [m/s]', '10m Avg [m/s]',
        'Pressure 5m [mbar]', '98m WV [°]', '78m WV [°]', '48m WV [°]',
        'Temp 5m [°C]', 'Hum 5m'
    ]
    
    # Find rows where the target value is missing
    missing_indices = df.index[df['100m Avg[m/s]'].isna()].tolist()

    for index in missing_indices:
        # Extract the relevant rows to create a sequence
        seq_look_back = min(look_back, index + 1)
        sequence = df[features].iloc[index - seq_look_back + 1: index + 1].values.astype(float)
        
        # Reshape the sequence
        sequence = sequence.reshape(seq_look_back, len(features))
        
        if len(sequence) < look_back:
            # Pad the sequence if its length is less than look_back
            pad_width = ((look_back - len(sequence), 0), (0, 0))
            sequence = np.pad(sequence, pad_width, mode='constant', constant_values=0)
        
        sequence = sequence.reshape(1, look_back, len(features))
        prediction = model.predict(sequence)
        df.at[index, '100m Avg[m/s]'] = prediction[0][0]
    
    return df

# Streamlit app
st.title('Wind Speed Forecasting')

# Read the input CSV file
input_csv = 'input.csv'
df = pd.read_csv('sample.csv')

st.write('Input Data:')
st.dataframe(df)

# Define look_back period
look_back = 10

# Predict the missing 100m Avg[m/s] values
df = predict_missing_values(df, look_back)

st.write('Data with Predictions:')
st.dataframe(df)

# Save the completed dataframe to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp:
    output_csv = temp.name
    df.to_csv(output_csv, index=False)

# Download the temporary file automatically
st.markdown(f'[Download the completed CSV file]({output_csv})', unsafe_allow_html=True)

# Optionally, delete the temporary file after download (commented out for clarity)
# os.remove(output_csv)
