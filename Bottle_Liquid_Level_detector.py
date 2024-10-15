
import cv2
import numpy as np


video_path = '/Users/marcelosponton/Desktop/🍊/IMG_4450.MOV'

cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# ROI
x1, y1 = 300, 750
x2, y2 = 750, 950
aux_line_y = 800  # Auxiliary line at height y = 800


while True:
   
    ret, frame = cap.read()
    
    
    if not ret:
        print("End of video or error reading the frame.")
        break

    # Mask of the same size as the frame with all values set to 0
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # White rectangle on the mask at ROI
    mask[y1:y2, x1:x2] = 255

    # Mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # grayscale frame
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply thresholding
    _, threshold = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Apply Canny edge detection
    edges = cv2.Canny(threshold, 140, 200)
    
    # Apply the Hough Transform to detect lines
    n_max = 40
    rho_res = 1
    theta_res = np.pi / 180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20
    lines = cv2.HoughLinesP(edges, rho_res, theta_res, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    
    if lines is not None:
        count = 0
        for line in lines:
            x1_line, y1_line, x2_line, y2_line = line[0]
            angle = np.arctan2(y2_line - y1_line, x2_line - x1_line) * 180 / np.pi
            # Calculate the length of the line
            length = np.sqrt((x2_line - x1_line) ** 2 + (y2_line - y1_line) ** 2)
            # Filter horizontal lines by angle and lenght
            if abs(angle) < 10 and length >= 200:
                color = (0, 255, 0)  # Default color: green
                if y1_line < aux_line_y and y2_line < aux_line_y:
                    color = (0, 0, 255)  # Red if above the auxiliary line
                    
                    cv2.putText(frame, "Bottle overfilled", (x1, aux_line_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), color, 12)
                count += 1
                if count >= n_max:
                    break

    # ROI rectangle on the original frame for reference
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    cv2.imshow(' Oranje juice level detection', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
