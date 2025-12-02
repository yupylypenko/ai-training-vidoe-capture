import streamlit as st
import cv2
import requests
import numpy as np
from datetime import datetime
import os

# DIAL API endpoint URL
dial_api_url = "https://api.example.com/dial/inference"


def display_dial_insights(insights: dict, snapshot_file: str = None):
    """
    Display DIAL API insights using Streamlit components.
    
    Args:
        insights: Dictionary containing insights from DIAL API
        snapshot_file: Optional path to the snapshot file for display
    """
    if snapshot_file:
        st.caption(f"From snapshot: {snapshot_file}")
    
    # Display status
    if insights.get('status'):
        status_emoji = "✅" if insights['status'] == 'success' else "⚠️"
        st.text(f"{status_emoji} Status: {insights['status']}")
    
    # Display confidence if available
    if insights.get('confidence') is not None:
        st.text(f"Confidence: {insights['confidence']:.2%}" if isinstance(insights['confidence'], (int, float)) else f"Confidence: {insights['confidence']}")
    
    # Display predictions in a table if available
    if insights.get('predictions') and len(insights['predictions']) > 0:
        st.subheader("Predictions")
        if isinstance(insights['predictions'], list):
            predictions_data = []
            for pred in insights['predictions']:
                if isinstance(pred, dict):
                    predictions_data.append(pred)
                else:
                    predictions_data.append({"prediction": str(pred)})
            if predictions_data:
                st.table(predictions_data)
        else:
            st.text(str(insights['predictions']))
    
    # Display detections in a table if available
    if insights.get('detections') and len(insights['detections']) > 0:
        st.subheader("Detections")
        if isinstance(insights['detections'], list):
            detections_data = []
            for det in insights['detections']:
                if isinstance(det, dict):
                    detections_data.append(det)
                else:
                    detections_data.append({"detection": str(det)})
            if detections_data:
                st.table(detections_data)
        else:
            st.text(str(insights['detections']))
    
    # Display labels if available
    if insights.get('labels') and len(insights['labels']) > 0:
        st.subheader("Labels")
        if isinstance(insights['labels'], list):
            labels_text = ", ".join(str(label) for label in insights['labels'])
            st.text(labels_text)
        else:
            st.text(str(insights['labels']))
    
    # Display results if available
    if insights.get('results'):
        st.subheader("Results")
        if isinstance(insights['results'], (dict, list)):
            st.json(insights['results'])
        else:
            st.text(str(insights['results']))
    
    # Display analysis if available
    if insights.get('analysis'):
        st.subheader("Analysis")
        if isinstance(insights['analysis'], (dict, list)):
            st.json(insights['analysis'])
        else:
            st.text(str(insights['analysis']))
    
    # Display objects if available
    if insights.get('objects') and len(insights['objects']) > 0:
        st.subheader("Objects")
        if isinstance(insights['objects'], list):
            objects_data = []
            for obj in insights['objects']:
                if isinstance(obj, dict):
                    objects_data.append(obj)
                else:
                    objects_data.append({"object": str(obj)})
            if objects_data:
                st.table(objects_data)
        else:
            st.text(str(insights['objects']))
    
    # Display metadata if available
    if insights.get('metadata') and insights['metadata']:
        st.subheader("Metadata")
        st.json(insights['metadata'])
    
    # Option to view raw response
    with st.expander("View Raw API Response"):
        st.json(insights.get('raw_response', {}))


def run_dial_inference(image_path: str):
    """
    Send a POST request to the DIAL API endpoint with an image file and extract insights.
    
    Args:
        image_path: Path to the image file to send
        
    Returns:
        Dictionary containing extracted insights from the API response
    """
    try:
        # Open the image file in binary mode
        with open(image_path, 'rb') as image_file:
            # Prepare the file for upload
            files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
            
            # Send POST request to DIAL API endpoint
            response = requests.post(dial_api_url, files=files)
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Process the API response and extract insights from JSON data
            try:
                json_data = response.json()
                
                # Extract relevant insights from the JSON response
                insights = {
                    'status': json_data.get('status', 'unknown'),
                    'predictions': json_data.get('predictions', []),
                    'confidence': json_data.get('confidence', None),
                    'detections': json_data.get('detections', []),
                    'labels': json_data.get('labels', []),
                    'metadata': json_data.get('metadata', {}),
                    'raw_response': json_data  # Keep full response for reference
                }
                
                # Extract any additional relevant fields that might be present
                if 'results' in json_data:
                    insights['results'] = json_data['results']
                if 'analysis' in json_data:
                    insights['analysis'] = json_data['analysis']
                if 'objects' in json_data:
                    insights['objects'] = json_data['objects']
                
                return insights
                
            except ValueError as e:
                # Response is not valid JSON
                raise ValueError(f"API response is not valid JSON: {e}. Response text: {response.text[:200]}")
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"HTTP error from DIAL API: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error sending request to DIAL API: {e}")


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

# DIAL API inference option
enable_dial_inference = st.checkbox("Enable DIAL API Inference", value=False)

# Initialize session state for insights
if 'dial_insights' not in st.session_state:
    st.session_state.dial_insights = None
if 'last_snapshot_file' not in st.session_state:
    st.session_state.last_snapshot_file = None

if st.button("Capture Snapshot"):
    # Clear previous insights when capturing a new snapshot
    st.session_state.dial_insights = None
    st.session_state.last_snapshot_file = None
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
        
        # Store the snapshot filename in session state
        st.session_state.last_snapshot_file = filename
        
        # Run DIAL inference if enabled
        if enable_dial_inference:
            st.divider()
            st.subheader("DIAL API Insights")
            
            with st.spinner("Sending image to DIAL API and processing..."):
                try:
                    insights = run_dial_inference(filename)
                    
                    # Store insights in session state
                    st.session_state.dial_insights = insights
                    
                    # Display insights using helper function
                    display_dial_insights(insights, filename)
                    
                except Exception as e:
                    st.error(f"Error running DIAL inference: {str(e)}")
                    st.info("Make sure the DIAL API endpoint is correctly configured and accessible.")
                    st.session_state.dial_insights = None

# Display stored insights if available (from previous capture)
if st.session_state.dial_insights is not None and st.session_state.last_snapshot_file:
    st.divider()
    st.subheader("Latest DIAL API Insights")
    display_dial_insights(st.session_state.dial_insights, st.session_state.last_snapshot_file)
