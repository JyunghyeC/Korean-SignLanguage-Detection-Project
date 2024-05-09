import os
import cv2

# 자음과 모음 데이터 저장 경로
CONSONANTS_DIR = './data/consonants'
VOWELS_DIR = './data/vowels'

for directory in [CONSONANTS_DIR, VOWELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

number_of_consonants = 19
number_of_vowels = 17

dataset_size = 100

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Press "Q" for consonants, "W" for vowels!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        class_type = 'consonants'
        number_of_classes = number_of_consonants
        data_dir = CONSONANTS_DIR
        break
    elif key == ord('w'):
        class_type = 'vowels'
        number_of_classes = number_of_vowels
        data_dir = VOWELS_DIR
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

if 'class_type' not in locals():
    print("Invalid key pressed. Please press 'Q' for consonants or 'W' for vowels.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'When you are ready, press "Q" to start recording', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
