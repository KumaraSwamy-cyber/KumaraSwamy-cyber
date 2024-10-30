import serial
import time
import numpy as np
from sklearn.externals import joblib  # For loading trained ML model
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Load trained model (SVM or k-NN model saved after training)
model = joblib.load('trained_model.pkl')

# Initialize Arduino serial communication
arduino = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
time.sleep(2)

# Initialize EEG device (replace with your EEG board's settings)
params = BrainFlowInputParams()
params.serial_port = 'COM6'  # Replace with your EEG port
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()

# Start the EEG session
board.start_stream()

# Command dictionary
commands = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right'}

# Function to send command to Arduino
def send_command(command):
    arduino.write(command.encode())
    print(f"Sent command: {command}")

try:
    while True:
        # Collect EEG data in real-time
        data = board.get_board_data()  # Collects EEG data
        features = extract_features(data)  # Process data and extract features
        
        # Predict command using trained model
        command = model.predict([features])[0]
        
        # Map prediction to movement command
        if command == 0:
            send_command('F')  # Forward
        elif command == 1:
            send_command('B')  # Backward
        elif command == 2:
            send_command('L')  # Left
        elif command == 3:
            send_command('R')  # Right
        
        time.sleep(0.5)  # Delay for stable command interpretation

except KeyboardInterrupt:
    print("Stopping...")
    board.stop_stream()
    board.release_session()
    arduino.close()
