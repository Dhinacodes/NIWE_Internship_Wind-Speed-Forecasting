import sys
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Function to create sequences and predict missing values
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

# Main function to process the CSV
def main(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Define look_back period
    look_back = 10

    # Predict the missing 100m Avg[m/s] values
    df = predict_missing_values(df, look_back)

    # Save the completed dataframe to CSV
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_csv.py <input_csv_path> <output_csv_path>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    main(input_csv, output_csv)
