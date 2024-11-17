import cv2
import time

def main():
    # Define a video capture object
    cap = cv2.VideoCapture(0)

    # Main Loop
    while True:
        # Capture the frame
        ret, frame = cap.read()

        if ret:
            # Display the live video stream
            cv2.imshow("Webcam", frame)
        else:
            print("! No frame")

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # Save an image if Enter (13) or Space (32) is pressed
        if key == 13 or key == 32:
            filename = f'{int(time.time())}.png'  # Save with a timestamp
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")

        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the capture object and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()