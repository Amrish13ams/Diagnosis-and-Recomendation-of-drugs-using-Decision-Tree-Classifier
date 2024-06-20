import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\amsmc\Downloads\drug200.csv")

df.dropna(inplace=True)

x_cat = df[['Sex', 'BP', 'Cholesterol']]
encoder = OneHotEncoder()
x_cat_encoded = encoder.fit_transform(x_cat).toarray()

x_num = df.drop(['Sex', 'BP', 'Cholesterol', 'Drug'], axis=1)
x_cat_encoded_df = pd.DataFrame(x_cat_encoded, columns=[f"cat_{i}" for i in range(x_cat_encoded.shape[1])]) 
x = pd.concat([x_num, x_cat_encoded_df], axis=1)

y = df['Drug'].map({'DrugA': 1, 'drugA': 1, 'DrugB': 2, 'drugB': 2, 'DrugC': 3, 'drugC': 3, 'DrugX': 4, 'drugX': 4, 'DrugY': 5, 'drugY': 5}).values  # Convert drug names to numerical labels
y_encoder = OneHotEncoder()
y_encoded = y_encoder.fit_transform(y.reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)

param_grid = {'max_depth': [3, 5, 7, 9], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best Parameters:", grid_search.best_params_)

best_classifier = grid_search.best_estimator_
y_pred_best = best_classifier.predict(x_test)
cm_best = confusion_matrix(y_test.argmax(axis=1), y_pred_best.argmax(axis=1))
accuracy_best = accuracy_score(y_test.argmax(axis=1), y_pred_best.argmax(axis=1))
print("Best Model Confusion Matrix:")
print(cm_best)
print("Best Model Accuracy:", accuracy_best)
