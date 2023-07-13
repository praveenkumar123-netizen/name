import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv('crop_recommendation.csv')

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('saved_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('saved_model.pkl', 'rb') as file:
    model = pickle.load(file)

simulation_data = pd.DataFrame({
    'N': [80, 85, 90],
    'P': [40, 45, 50],
    'K': [35, 40, 45],
    'temperature': [20, 25, 30],
    'humidity': [60, 70, 80],
    'ph': [6.5, 7.0, 7.5],
    'rainfall': [50, 60, 70]
})

predictions = model.predict(simulation_data)

plt.bar(simulation_data.index, predictions)
plt.xlabel('Simulation Index')
plt.ylabel('Crop Label')
plt.title('Predicted Crop Labels for Simulation Data')
plt.xticks(simulation_data.index)
plt.show()
