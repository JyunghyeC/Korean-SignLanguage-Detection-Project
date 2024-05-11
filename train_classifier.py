import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

normalized_data = []

# 모든 랜드마크의 개수가 최대 개수
max_landmarks = 21

# Normalization 진행, 진행하지 않을시 데이터 간 랜드마크 개수의 차이가 있어 데이터 추출이 불가능
for sample in data:

    sample = np.array(sample).reshape(-1, 2)

    min_values = np.min(sample, axis=0)
    max_values = np.max(sample, axis=0)

    ranges = max_values - min_values

    normalized_sample = (sample - min_values) / ranges

    if normalized_sample.shape[0] < max_landmarks:
        # 최대 랜드마크 보다 적으면 0으로 대체
        normalized_sample = np.pad(normalized_sample, ((0, max_landmarks - normalized_sample.shape[0]), (0, 0)),
                                   mode='constant')
    elif normalized_sample.shape[0] > max_landmarks:
        # 최대 보다 많으면 삭제
        normalized_sample = normalized_sample[:max_landmarks]

    normalized_data.append(normalized_sample.flatten())

normalized_data = np.array(normalized_data)

X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, shuffle=True,
                                                    stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_predict, y_test)
# 약 99.6%
print(f'{score * 100} % of samples were classified correctly!')

# 모델 저장
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
