import cv2
import numpy as np
import vehicles

# Video capture and background subtractor
cap = cv2.VideoCapture("../Videos/video1.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200, varThreshold=90)

# Kernels for morphological operations
kernalOp = np.ones((3,3), np.uint8)
kernalCl = np.ones((11,11), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1

# Counters for vehicles per lane
cnt_up = 0    # Lane 1 (vehicles going UP)
cnt_down = 0  # Lane 2 (vehicles going DOWN)

# Lines to detect crossing
line_up = 400
line_down = 250
up_limit = 230
down_limit = int(4.5 * (500 / 5))

print("VEHICLE DETECTION, CLASSIFICATION AND COUNTING")

if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 500))

    # Age all existing cars
    for car in cars:
        car.age_one()

    fgmask = fgbg.apply(frame)

    # Binarization and morphological operations
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True
            if up_limit <= cy <= down_limit:
                for car in cars:
                    if abs(x - car.getX()) <= w and abs(y - car.getY()) <= h:
                        new = False
                        car.updateCoords(cx, cy)

                        if car.going_UP(line_down, line_up):
                            cnt_up += 1
                            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.imwrite("./detected_vehicles/vehicleUP" + str(cnt_up) + ".png", img[y:y+h-1, x:x+w])

                        elif car.going_DOWN(line_down, line_up):
                            cnt_down += 1
                            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.imwrite("./detected_vehicles/vehicleDOWN" + str(cnt_down) + ".png", img[y:y+h-1, x:x+w])

                        break

                # Remove old cars
                for car in cars[:]:
                    if car.getState() == '1':
                        if car.getDir() == 'down' and car.getY() > down_limit:
                            car.setDone()
                        elif car.getDir() == 'up' and car.getY() < up_limit:
                            car.setDone()
                    if car.timedOut():
                        cars.remove(car)

                if new:
                    p = vehicles.Car(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display vehicle IDs and type (basic logic)
    for car in cars:
        cv2.putText(frame, str(car.getId()), (car.getX(), car.getY()), font, 0.3, (255, 255, 0), 1, cv2.LINE_AA)
        if line_down + 20 <= car.getY() <= line_up - 20:
            a = (h + (0.74 * w) - 100)
            if a >= 0:
                cv2.putText(frame, "Truck", (car.getX(), car.getY()), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Car", (car.getX(), car.getY()), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display counting info
    str_up = 'UP: ' + str(cnt_up)
    str_down = 'DOWN: ' + str(cnt_down)

    # Draw lines
    cv2.line(frame, (0, line_up), (900, line_up), (255, 0, 255), 3, 8)  # Magenta
    cv2.line(frame, (0, up_limit), (900, up_limit), (0, 255, 255), 3, 8)  # Cyan
    cv2.line(frame, (0, down_limit), (900, down_limit), (255, 0, 0), 3, 8)  # Yellow
    cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 3, 8)  # Blue

    # Display counts on frame
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SIGNAL TIMING CALCULATION ----------------

# Assume lane 3 and lane 4 vehicle counts are zero; modify as needed
count_lane1 = cnt_up
count_lane2 = cnt_down
count_lane3 = 0
count_lane4 = 0

no_of_vehicles = [count_lane1, count_lane2, count_lane3, count_lane4]

baseTimer = 120
timeLimits = [5, 30]
maxCycleTime = 60  # Max red time

total_vehicles = sum(no_of_vehicles)
if total_vehicles == 0:
    print("No vehicles detected in any lane.")
else:
    # Calculate green time per lane
    t = [(i / total_vehicles) * baseTimer if timeLimits[0] < (i / total_vehicles) * baseTimer < timeLimits[1]
        else min(timeLimits, key=lambda x: abs(x - (i / total_vehicles) * baseTimer)) for i in no_of_vehicles]

    # Calculate red time per lane (assuming maxCycleTime = 60s)
    red_times = [maxCycleTime - green for green in t]

    print("\nVehicle counts per lane:", no_of_vehicles)
    print("Recommended GREEN light times (seconds):", [round(x,2) for x in t])
    print("Recommended RED light times (seconds):", [round(x,2) for x in red_times])
