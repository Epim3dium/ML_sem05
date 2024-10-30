#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from ucimlrepo import fetch_ucirepo 

# Wczytanie danych
iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets 

# y reprezentuje etykiety (targety) dla zbioru danych, a przekształcenie go w Series ułatwia późniejsze operacje, 
# takie jak mapowanie czy obliczanie entropii i zysku informacji.
y = y.squeeze()

# Konwersja kolumny `y` na format binarny: 1 dla Iris-setosa, 0 dla pozostałych klas
y_binary = y.map(lambda x: 1 if x == 'Iris-setosa' else 0)

# Dyskretyzacja danych
def discretize_data(X, n_bins=3):
    X_discretized = pd.DataFrame()
    
    for i in range(X.shape[1]):
        # Użycie pd.qcut do dyskretyzacji na podstawie percentyli
        X_discretized[i] = pd.qcut(X.iloc[:, i], q=n_bins, labels=['low', 'medium', 'high'])
    
    return X_discretized

X_discretized = discretize_data(X)

# Tworzenie zbiorów na podstawie wartości pierwszej kolumny
low_data = X_discretized[X_discretized[0] == 'low']
medium_data = X_discretized[X_discretized[0] == 'medium']
high_data = X_discretized[X_discretized[0] == 'high']

# Odpowiednie wartości celu dla każdego zbioru
y_low = y_binary[low_data.index]
y_medium = y_binary[medium_data.index]
y_high = y_binary[high_data.index]

# Sprawdzenie rozmiarów podzbiorów
print("Size of Low subset:", len(low_data))
print("Size of Medium subset:", len(medium_data))
print("Size of High subset:", len(high_data))

# Funkcja obliczająca entropię
def calculate_entropy(y_subset):
    if len(y_subset) == 0:
        return 0  # Zwróć 0, jeśli podzbiór jest pusty

    p_pos = sum(y_subset) / len(y_subset)
    p_neg = 1 - p_pos
    if p_pos == 0 or p_neg == 0:
        return 0
    return - (p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))

# Obliczanie entropii dla całego zbioru i podzbiorów
entropy_full = calculate_entropy(y_binary)
entropy_low = calculate_entropy(y_low)
entropy_medium = calculate_entropy(y_medium)
entropy_high = calculate_entropy(y_high)

# Obliczenie zysku informacji
total_length = len(X_discretized)
weight_low = len(low_data) / total_length
weight_medium = len(medium_data) / total_length
weight_high = len(high_data) / total_length

gain_S_a = entropy_full - (weight_low * entropy_low + weight_medium * entropy_medium + weight_high * entropy_high)

# Wyświetlanie wyników
print("Entropy of full dataset:", entropy_full)
print("Entropy of Low subset:", entropy_low)
print("Entropy of Medium subset:", entropy_medium)
print("Entropy of High subset:", entropy_high)


# In[15]:


# Wyniki z poprzedniego kroku
print("Information gain from splitting on first column:", gain_S_a)

# Interpretacja
if gain_S_a > 0:
    print("The split improves our ability to classify elements of S.")
    print(f"The gain of {gain_S_a} indicates that the split effectively reduces uncertainty in classification.")
else:
    print("The split does not improve our ability to classify elements of S.")
    print("The gain of 0 means the split does not reduce uncertainty in classification.")


# In[16]:


# Inicjalizowanie zmiennych do przechowywania zysków informacji dla każdej cechy
gains = []

# Obliczanie zysku informacji dla każdej cechy
for col in range(X_discretized.shape[1]):
    # Tworzymy podzbiory dla danej cechy
    low_data = X_discretized[X_discretized[col] == 'low']
    medium_data = X_discretized[X_discretized[col] == 'medium']
    high_data = X_discretized[X_discretized[col] == 'high']

    # Odpowiednie wartości celu dla każdego zbioru
    y_low = y_binary[low_data.index]
    y_medium = y_binary[medium_data.index]
    y_high = y_binary[high_data.index]

    # Obliczamy entropię dla każdego zbioru
    entropy_full = calculate_entropy(y_binary)
    entropy_low = calculate_entropy(y_low)
    entropy_medium = calculate_entropy(y_medium)
    entropy_high = calculate_entropy(y_high)

    # Obliczamy zysk informacji
    total_length = len(X_discretized)
    weight_low = len(low_data) / total_length
    weight_medium = len(medium_data) / total_length
    weight_high = len(high_data) / total_length

    gain = entropy_full - (weight_low * entropy_low + weight_medium * entropy_medium + weight_high * entropy_high)
    gains.append((col, gain))

    print(f"Information Gain for feature {col}: {gain}")

# Znalezienie cechy z największym zyskiem informacji
best_feature = max(gains, key=lambda x: x[1])
print(f"The feature with the greatest information gain is feature {best_feature[0]} with a gain of {best_feature[1]}.")


# In[21]:


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# Wczytanie danych
iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets 


y = y.squeeze()

# Konwersja kolumny `y` na format binarny: 1 dla Iris-setosa, 0 dla pozostałych klas
y_binary = y.map(lambda x: 1 if x == 'Iris-setosa' else 0)

# Dyskretyzacja danych
X_discretized = discretize_data(X)

# Funkcja do obliczania entropii
def calculate_entropy(y_subset):
    if len(y_subset) == 0:
        return 0

    p_pos = sum(y_subset) / len(y_subset)
    p_neg = 1 - p_pos
    if p_pos == 0 or p_neg == 0:
        return 0
    return - (p_pos * np.log2(p_pos) + p_neg * np.log2(p_neg))

# Funkcja do obliczania zysku informacji
def information_gain(X, y, feature_index):
    # Obliczamy entropię dla całego zbioru
    entropy_full = calculate_entropy(y)
    
    # Podział danych na podstawie cechy
    subsets = X.iloc[:, feature_index].unique()
    weighted_entropy = 0

    for subset in subsets:
        y_subset = y[X.iloc[:, feature_index] == subset]
        weight = len(y_subset) / len(y)
        weighted_entropy += weight * calculate_entropy(y_subset)

    return entropy_full - weighted_entropy

# Funkcja do budowy drzewa decyzyjnego
class DecisionNode:
    def __init__(self, feature_index=None, value=None, left=None, right=None, class_label=None):
        self.feature_index = feature_index
        self.value = value
        self.left = left
        self.right = right
        self.class_label = class_label

def build_tree(X, y):
    # Sprawdzanie, czy wszystkie etykiety są takie same
    if len(y.unique()) == 1:
        return DecisionNode(class_label=y.iloc[0])

    # Jeśli nie ma więcej cech, zwracamy etykietę klasy
    if X.empty:
        return DecisionNode(class_label=y.mode()[0])

    best_gain = -1
    best_feature = None

    # Przeszukiwanie cech
    for feature_index in range(X.shape[1]):
        gain = information_gain(X, y, feature_index)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index

    if best_gain == 0:
        return DecisionNode(class_label=y.mode()[0])

    # Dzielimy dane
    node = DecisionNode(feature_index=best_feature)

    # Ustalamy unikalne wartości cechy
    subsets = X.iloc[:, best_feature].unique()

    for subset in subsets:
        indices = X.iloc[:, best_feature] == subset
        subset_X = X[indices].drop(best_feature, axis=1)
        subset_y = y[indices]

        # Ustalamy wartość węzła
        node.value = subset

        # Rekurencyjne budowanie drzewa
        child_node = build_tree(subset_X, subset_y)
        if subset == 'low':
            node.left = child_node
        else:
            node.right = child_node

    return node

# Budowanie drzewa
decision_tree = build_tree(X_discretized, y_binary)

# Funkcja do klasyfikacji
def classify(node, sample):
    if node.class_label is not None:
        return node.class_label
    if sample[node.feature_index] == node.value:
        return classify(node.left, sample)
    else:
        return classify(node.right, sample)

# Przykładowe klasyfikowanie
sample = X_discretized.iloc[0]
print("Predykcja dla próbki:", classify(decision_tree, sample))


# # In[22]:
#
#
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
#
# plt.figure(figsize=(12, 8))
# plot_tree(clf, feature_names=list(X_encoded.columns), class_names=['Not Iris-setosa', 'Iris-setosa'], filled=True)
# plt.show()
#
#
# # In[ ]:
#
#
#
#
#
# # In[ ]:
#
#
#
#
