import cv2
import numpy as np
import winsound
import tkinter as tk

# Initialize GUI
gui = tk.Tk()
gui.title("Drone Detection")
gui.geometry("500x200")

# Create a text box for entering RTSP link
rtsp_entry = tk.Entry(gui, width=40)
rtsp_entry.pack(pady=10)

# Initialize the video capture object
cap = None

# Load Yolo
net = cv2.dnn.readNet("yolo-drone.weights", "yolo-drone.cfg")

# Output layer names
output_layers = net.getLayerNames()
output_layers = [output_layers[i - 1] for i in net.getUnconnectedOutLayers()]

# Name custom object
classes = ["Drone"]


def start_detection():
    global cap
    # Get the RTSP link from the text box
    rtsp_link = rtsp_entry.get()

    # Release any previous capture object
    if cap is not None:
        cap.release()

    # Initialize the video capture object with the RTSP link
    cap = cv2.VideoCapture(rtsp_link)

    if not cap.isOpened():
        print("Error: failed to open video stream.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture frame.")
            break

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []

        # Check if any objects were detected
        if len(outs) == 0:
            print("No objects detected in the input image")
        else:
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.3:
                        # Drone detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indexes) > 0:
                # Drone detected
                print("Drone detected!")
                winsound.PlaySound("beep-warning-6387.wav", winsound.SND_ASYNC)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = [[0, 255, 255]]  # yellow
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    #label = str(classes[class_ids[i]])
                    label = str(classes[class_ids[i]]) + " " + str(round(confidences[i] * 100, 2)) + "%"
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 5), font, 1, color, 2)
                    # Display the resulting frame

        cv2.imshow("Drone Detection", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # Release capture object and destroy windows
    cap.release()
    cv2.destroyAllWindows()


# Create a button for starting the detection process
start_button = tk.Button(gui, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

# Run the GUI
gui.mainloop()
