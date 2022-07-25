
import cv2

#capture webcam video
import numpy as np

web_cam = cv2.VideoCapture(0)

while True:
    #reading webcam video
    _, frame = web_cam.read()
    #cv2.imshow('frame', frame)

    #converting to HSV
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #set range and mask for red
    red_low = np.array([136,87,111])
    red_high =np.array([180,255,255])
    red_mask = cv2.inRange(hsv_frame,red_low, red_high)

    #set range and mask for green
    green_low = np.array([25,52,72],)
    green_high = np.array([102,255,255])
    green_mask = cv2.inRange(hsv_frame, green_low, green_high)

    #set range and mask for blue
    blue_low = np.array([94,80,2])
    blue_high = np.array([120,255,255])
    blue_mask = cv2.inRange(hsv_frame, blue_low, blue_high)

    k = np.ones((5,5))

    red_mask = cv2.dilate(red_mask,k)
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)

    green_mask = cv2.dilate(green_mask,k)
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)

    blue_mask = cv2.dilate(blue_mask, k)
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)

    #contour to track red
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "RED!", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    #contour to track green
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(imageFrame, "Green Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))

    #contour to track blue
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Blue Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

    cv2.imshow("WHAT COLOR?", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            web_cam.release()
            cv2.destroyAllWindows()
            break
