import cv2
import numpy as np
import time
import math
import mediapipe as mp
import pyautogui

FRAME_REDUCTION = 100
SMOOTHING = 5
PINCH_THRESHOLD = 40
RIGHT_PINCH_THRESHOLD = 40
SCROLL_SENSITIVITY =3.0
CLICK_DEBOUNCE = 0.20

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0
pinch_active= False
dragging=False
last_click_time = 0

prev_x, prev_y = 0, 0
smooth_x, smooth_y = 0, 0
pinch_active=False
dragging = False
last_click_time=0
last_right_click = 0
last_scroll_click = 0
scroll_anchor = None

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

print("AI Virtual Mouse strating.. Press q to quit.")
while True:
    success, frame = cap.read()
    if not success:
        break

    frame= cv2.flip(frame,1)
    image_height, image_width = frame.shape[:2]
    rgb= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks= results.multi_hand_landmarks[0]
        lm_list= []
        for id, lm in enumerate(hand_landmarks.landmark):
            x, y = int(lm.x * cam_w), int(lm.y * cam_h)
            lm_list.append((x, y))

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        thumb_tip = lm_list[4]
        index_tip = lm_list[8]
        middle_tip = lm_list[12]

        x_cam, y_cam = index_tip

        left_bound = FRAME_REDUCTION
        right_bound = cam_w - FRAME_REDUCTION
        top_bound= FRAME_REDUCTION
        bottom_bound = cam_h - FRAME_REDUCTION

        x_cam = np.clip(x_cam, left_bound, right_bound)
        y_cam = np.clip(y_cam, top_bound, bottom_bound)

        screen_x = np.interp(x_cam, (left_bound, right_bound), (0, screen_w))
        screen_y = np.interp(y_cam, (top_bound, bottom_bound), (0, screen_h))

        smooth_x= prev_x + (screen_x - prev_x)
        smooth_y= prev_y + (screen_y - prev_y)
        prev_x, prev_y = smooth_x, smooth_y

        pyautogui.moveTo(screen_x, screen_y, _pause = False)

        pinch_dist =distance(thumb_tip,index_tip)
        now = time.time()

        right_pinch_dist = distance(index_tip, middle_tip)
        if right_pinch_dist < RIGHT_PINCH_THRESHOLD:
            if now - last_right_click > CLICK_DEBOUNCE:
                pyautogui.rightClick()
                last_right_click = now
            cv2.putText(frame, "Right Click", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if pinch_dist < PINCH_THRESHOLD:
            cv2.circle(frame,index_tip, 12, (0, 255, 0), cv2.FILLED)
            if not pinch_active:
                pinch_active = True
                pinch_start_time = now
                pinch_start_pos= (screen_x, screen_y)
                pyautogui.mouseDown()
                dragging = True
                cv2.putText(frame,"Pinch start - potential drag", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(frame,"Dragging", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            if pinch_active:
                pyautogui.mouseUp()
                pinch_active = False
                pinch_duration = now - pinch_start_time
                dx =abs(screen_x - pinch_start_pos[0])
                dy = abs(screen_y - pinch_start_pos[1])
                move_dist = math.hypot(dx, dy)

                if pinch_duration < 0.6 and move_dist <20:
                    if now - last_click_time > CLICK_DEBOUNCE:
                        pyautogui.click()
                        last_click_time = now
                        cv2.putText(frame, "Click", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                else:
                    cv2.putText(frame, "Drag released", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                dragging = False

        fingers_seperated = right_pinch_dist > 80
        if fingers_seperated:
            idx_mid_x, idx_mid_y = (index_tip[0]+middle_tip[0])//2, (index_tip[1]+middle_tip[1])//2
            if scroll_anchor is None:
                scroll_anchor = idx_mid_x
                last_scroll_time = now
            else:
                dy= scroll_anchor - idx_mid_y
                if abs(dy) > 10 and now - last_scroll_time > 0.05:
                    scroll_amount = int(dy/10* SCROLL_SENSITIVITY)
                    if scroll_amount != 0:
                        pyautogui.scroll(scroll_amount)
                        last_scroll_time = now
                        scroll_anchor = idx_mid_y
                        cv2.putText(frame, f"Scrolling{scroll_amount}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        else:
            scroll_anchor = None

        cv2.rectangle(frame, (5,5), (260,160), (0,0,0), cv2.FILLED)
        cv2.putText(frame, "AI Virtual Mouse", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Index -> move", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, "Pinch thumb + index -> click/drag", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame, "Index + Middle pinch -> right click", (10,95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame, "Index + Middle apart & move vertically -> scroll", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.imshow("AI Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()