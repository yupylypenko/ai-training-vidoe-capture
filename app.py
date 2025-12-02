import streamlit as st
import cv2
import requests
import numpy as np
from datetime import datetime
import os

st.title("Webcam DIAL App")
st.write("This app captures video from your webcam and processes it using DIAL (Device Interaction and Automation Layer) technology.")

# Camera source selection
camera_source = st.radio(
    "Select Camera Source:",
    ["Local Webcam", "Network Camera URL", "Test/Demo Mode"],
    horizontal=True
)

if camera_source == "Network Camera URL":
    camera_url = st.text_input("Enter camera URL (e.g., rtsp://, http://, or IP camera URL):", 
                               placeholder="rtsp://username:password@ip:port/stream")
elif camera_source == "Test/Demo Mode":
    st.info("Test mode will generate a sample image for testing purposes.")

if st.button("Capture Snapshot"):
    frame = None
    success_message = ""
    caption = ""
    
    if camera_source == "Test/Demo Mode":
        # Generate a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (590, 430), (0, 255, 0), 2)
        cv2.putText(test_image, "Test Image", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(test_image, "Webcam DIAL App", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        frame = test_image
        success_message = "Test image generated successfully!"
        caption = "Test/Demo Image"
        
    elif camera_source == "Network Camera URL":
        if not camera_url or camera_url.strip() == "":
            st.error("Please enter a camera URL.")
        else:
            # Try to access network camera
            cap = cv2.VideoCapture(camera_url.strip())
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    success_message = f"Snapshot captured from network camera!"
                    caption = f"Network Camera: {camera_url[:50]}..."
                else:
                    st.error("Failed to capture frame from network camera.")
                cap.release()
            else:
                st.error(f"Unable to connect to camera URL: {camera_url}")
                st.info("Make sure the URL is correct and the camera is accessible.")
    
    else:  # Local Webcam
        # Try to access webcam - try multiple indices
        cap = None
        camera_index = None
        
        # Try indices 0-3 to find an available camera
        for idx in range(4):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None:
                    cap = test_cap
                    camera_index = idx
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if cap is not None and cap.isOpened():
            # Capture a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                success_message = f"Snapshot captured successfully from camera {camera_index}!"
                caption = f"Local Camera {camera_index}"
            else:
                st.error("Failed to capture frame from webcam.")
            
            # Release the VideoCapture object
            cap.release()
        else:
            st.error("Unable to access webcam. Please make sure your webcam is connected and not being used by another application.")
            st.info("**Note for WSL2 users:** WSL2 doesn't have direct access to USB devices. You may need to:")
            st.info("1. Use USB/IP to forward the webcam to WSL2, or")
            st.info("2. Run this app on Windows directly, or")
            st.info("3. Use a network camera stream instead (select 'Network Camera URL' option above)")
    
    # Display the captured frame if available
    if frame is not None:
        # Save the captured frame as an image file
        # Create snapshots directory if it doesn't exist
        snapshots_dir = "snapshots"
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{snapshots_dir}/snapshot_{timestamp}.jpg"
        
        # Save the frame using OpenCV's imwrite
        cv2.imwrite(filename, frame)
        
        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=caption, use_container_width=True)
        if success_message:
            st.success(success_message)
        st.info(f"Snapshot saved as: `{filename}`")
