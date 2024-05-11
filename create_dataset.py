import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

CONSONANTS_DIR = './data/consonants'
VOWELS_DIR = './data/vowels'

data = []
labels = []


def process_images(directory):
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Unable to read image: {img_path}")
                continue

            data_aux = []
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

            if data_aux:
                data.append(data_aux)
                labels.append(class_folder)


process_images(CONSONANTS_DIR)

process_images(VOWELS_DIR)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
