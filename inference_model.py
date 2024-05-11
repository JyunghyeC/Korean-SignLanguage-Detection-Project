# 모델 문제점 : 정확도가 눈에 띄게 좋지 않다, 특정 사인을 했을 때 에러 발생(feature 갯수 불일치), 전체적으로 불안정

import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings

from PIL import ImageFont, ImageDraw, Image

warnings.filterwarnings("ignore")


# 한글 출력을 위한 코드
def draw_text_korean(image, text, position, font_path='./font/Arial Unicode.ttf',
                     font_size=30, font_color=(0, 255, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=font_color)
    return np.array(image_pil)


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static_image_mode를 False로 둔다면 된소리도 추가가 가능할지도 -> 공부 필요
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    # 자음
    'c0': 'ㄱ', 'c1': 'ㄴ', 'c2': 'ㄷ', 'c3': 'ㄹ', 'c4': 'ㅁ',
    'c5': 'ㅂ', 'c6': 'ㅅ', 'c7': 'ㅇ', 'c8': 'ㅈ', 'c9': 'ㅊ',
    'c10': 'ㅋ', 'c11': 'ㅌ', 'c12': 'ㅍ', 'c13': 'ㅎ',
    # 모음
    'v0': 'ㅏ', 'v1': 'ㅑ', 'v2': 'ㅓ', 'v3': 'ㅕ', 'v4': 'ㅗ',
    'v5': 'ㅛ', 'v6': 'ㅜ', 'v7': 'ㅠ', 'v8': 'ㅡ', 'v9': 'ㅣ',
    'v10': 'ㅐ', 'v11': 'ㅒ', 'v12': 'ㅓ', 'v13': 'ㅖ', 'v14': 'ㅢ',
    'v15': 'ㅚ', 'v16': 'ㅟ'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            # for i in range(len(hand_landmarks.landmark)):
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[prediction[0]]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        frame = draw_text_korean(frame, predicted_character, (x1, y1 - 10),
                                 font_path='./font/Arial Unicode.ttf', font_size=30,
                                 font_color=(0, 255, 0))

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
