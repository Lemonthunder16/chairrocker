import cv2
import numpy as np
import math

def count_fingers(thresh_img, drawing):
    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    largest_contour_index = -1

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour_index = i

    if largest_contour_index == -1:
        return 0

    # Get the largest contour
    cnt = contours[largest_contour_index]

    # Approximate the contour
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Convex hull
    hull = cv2.convexHull(cnt)
    
    # Convexity defects
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)

    if defects is None:
        return 0

    # Count fingers
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        
        # Use cosine rule to find angle of the far point
        a = math.dist(end, start)
        b = math.dist(far, start)
        c = math.dist(end, far)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / math.pi)

        # If angle is less than 90 degrees, count it as a finger
        if angle <= 90:
            finger_count += 1
            cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            
        cv2.line(drawing, start, end, [0, 255, 0], 2)

    # Finger count is the number of defects plus one
    return finger_count + 1

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest
    roi = frame[100:400, 100:400]

    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a copy of the roi for drawing purposes
    drawing = np.zeros(roi.shape, np.uint8)

    # Count the number of fingers
    finger_count = count_fingers(thresh, drawing)

    # Draw the count on the original frame
    cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    
    # Show the original frame and the thresholded image
    cv2.imshow("Frame", frame)
    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Drawing", drawing)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
