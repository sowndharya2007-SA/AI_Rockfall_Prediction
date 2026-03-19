import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load data
df = pd.read_csv("data/rockfall_dataset.csv")

X = df[["vibration","tilt","crack_width","rainfall","temperature"]]
y = df["rockfall"]

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Reshape
X_train = X_train.reshape((X_train.shape[0], 1, 5))
X_test = X_test.reshape((X_test.shape[0], 1, 5))

# LSTM Model (upgraded)
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(1,5)))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=8)

# Save model
model.save("model/rockfall_lstm_model.h5")

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

np.save("model/confusion_matrix.npy", cm)

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {acc*100:.2f}%")
print("Model trained & saved successfully")