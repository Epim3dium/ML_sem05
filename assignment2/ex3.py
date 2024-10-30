#!/usr/bin/env python
# coding: utf-8

# In[33]:

# Importuj potrzebne biblioteki
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Przygotuj dane
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets

# Konwersja X do DataFrame
X = pd.DataFrame(X, columns=iris.data.feature_names)

def discretize_data(X):
    X_discretized = pd.DataFrame()
    
    for i in range(X.shape[1]):
        # Użycie percentyli do ustalenia granic
        bins = np.percentile(X.iloc[:, i], [0, 33.3, 66.6, 100])
        labels = ['low', 'medium', 'high']
        X_discretized[i] = pd.cut(X.iloc[:, i], bins=bins, labels=labels, include_lowest=True)
    
    return X_discretized


# Dyskretyzuj dane
X_discretized = discretize_data(X)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_discretized, y, test_size=0.3, random_state=42)

# Implementacja Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)  # Unikalne klasy
        self.class_prob = {cls: 0 for cls in self.classes}  # Prawdopodobieństwo klas
        self.feature_prob = {cls: {} for cls in self.classes}  # Prawdopodobieństwo cech dla każdej klasy

        # Obliczanie prawdopodobieństw
        total_samples = len(y)
        
        for cls in self.classes:
            cls_samples = X[y == cls]
            self.class_prob[cls] = len(cls_samples) / total_samples  # P(Class)
            
            for feature in X.columns:
                # P(Xi | Class)
                feature_values = cls_samples[feature].value_counts(normalize=True)
                self.feature_prob[cls][feature] = feature_values.to_dict()

    def predict(self, X):
        predictions = []
        
        for _, row in X.iterrows():
            posteriors = {}

            for cls in self.classes:
                # Oblicz P(Class | X)
                prior = self.class_prob[cls]
                likelihood = 1
                
                for feature in X.columns:
                    feature_value = row[feature]
                    # Wartość prawdopodobieństwa dla cechy
                    likelihood *= self.feature_prob[cls].get(feature_value, 0)
                    
                posteriors[cls] = prior * likelihood
            
            # Klasa z najwyższym prawdopodobieństwem
            predictions.append(max(posteriors, key=posteriors.get))
        
        return predictions

# Tworzenie i trenowanie modelu Naive Bayes
nb_model = NaiveBayes()
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
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# In[34]:


print(conf_matrix)


# In[35]:


print("Rozkład klas w zbiorze treningowym:")
print(y_train.value_counts())

print("Rozkład klas w zbiorze testowym:")
print(y_test.value_counts())


# In[38]:


print(X)


# In[39]:


import pandas as pd
from ucimlrepo import fetch_ucirepo 

# Importuj zbiór danych
iris = fetch_ucirepo(id=53)

# Przygotuj dane
X = iris.data.features 
y = iris.data.targets 

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




