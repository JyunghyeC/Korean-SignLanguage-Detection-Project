import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_predict, y_test)

print(f'{score * 100} % of samples were classified correctly!')

# saving the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
