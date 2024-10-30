#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Pobranie danych
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets
X = pd.DataFrame(X, columns=iris.data.feature_names)


# In[16]:


# Importy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Funkcja dyskretyzująca dane
def discretize_data(X):
    X_discretized = pd.DataFrame()
    
    for i in range(X.shape[1]):
        bins = pd.qcut(X.iloc[:, i], q=3, labels=['low', 'medium', 'high'])
        X_discretized[X.columns[i]] = bins
    
    return X_discretized

# Dyskretyzacja danych
X_discretized = discretize_data(X)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_discretized, y.to_numpy().flatten(), test_size=0.3, random_state=42)

# Resetowanie indeksów po podziale
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train = pd.Series(y_train).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# Implementacja klasyfikatora Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        # Upewnienie się, że y jest jednowymiarowe
        y = pd.Series(y)
        
        # Inicjalizacja klas i ich prawdopodobieństw
        self.classes = y.unique()
        self.class_prob = {}
        self.feature_prob = {cls: {} for cls in self.classes}

        total_samples = len(y)
        
        for cls in self.classes:
            # Uzyskanie próbek dla danej klasy z X_train
            cls_samples = X[y == cls].reset_index(drop=True)  # Używamy reset_index, aby upewnić się, że indeksy są zgodne
            self.class_prob[cls] = len(cls_samples) / total_samples
            
            for feature in X.columns:
                feature_counts = cls_samples[feature].value_counts(normalize=True)
                self.feature_prob[cls][feature] = feature_counts.to_dict()
                
                for value in ['low', 'medium', 'high']:
                    if value not in self.feature_prob[cls][feature]:
                        self.feature_prob[cls][feature][value] = 1e-6

    def predict(self, X):
        predictions = []
        
        for _, row in X.iterrows():
            posteriors = {}

            for cls in self.classes:
                prior = self.class_prob[cls]
                likelihood = prior
                
                for feature in X.columns:
                    feature_value = row[feature]
                    likelihood *= self.feature_prob[cls][feature].get(feature_value, 1e-6)
                    
                posteriors[cls] = likelihood
            
            predictions.append(max(posteriors, key=posteriors.get))
        
        return predictions

# Tworzenie i trenowanie modelu Naive Bayes
nb_model = NaiveBayes()

# Trenowanie modelu
nb_model.fit(X_train, y_train)

# Przewidywanie
y_pred = nb_model.predict(X_test)

# Obliczanie metryk ewaluacyjnych
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[17]:


print(conf_matrix)


# In[18]:


print("Rozkład klas w zbiorze treningowym:")
print(y_train.value_counts())

print("Rozkład klas w zbiorze testowym:")
print(y_test.value_counts())


# In[19]:


print(X)


# In[21]:


# Konwertuj dane na DataFrame
df = pd.DataFrame(X, columns=iris.data.feature_names)

# Oblicz średnią, maksimum i minimum dla każdej kolumny
mean_values = df.mean()
max_values = df.max()
min_values = df.min()

# Wyświetlenie wyników
print("Średnie wartości (avg values):\n", mean_values)
print("\nMaksymalne wartości (max values):\n", max_values)
print("\nMinimalne wartości (min values):\n", min_values)


# In[ ]:




