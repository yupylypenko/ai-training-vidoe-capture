import streamlit as st
import cv2
import requests
import numpy as np
from datetime import datetime
import os
import time

# DIAL API endpoint URL
dial_api_url = "https://api.example.com/dial/inference"


def display_dial_insights(insights: dict, snapshot_file: str = None):
    """
    Display DIAL API insights using Streamlit components.
    
    Args:
        insights: Dictionary containing insights from DIAL API
        snapshot_file: Optional path to the snapshot file for display
    """
    try:
        if not isinstance(insights, dict):
            st.error("Invalid insights data: Expected a dictionary.")
            return
        
        if snapshot_file:
            st.caption(f"From snapshot: {snapshot_file}")
        
        # Display status
        try:
            if insights.get('status'):
                status_emoji = "✅" if insights['status'] == 'success' else "⚠️"
                st.text(f"{status_emoji} Status: {insights['status']}")
        except Exception as e:
            st.error(f"Error displaying status: {str(e)}")
        
        # Display confidence if available
        try:
            if insights.get('confidence') is not None:
                st.text(f"Confidence: {insights['confidence']:.2%}" if isinstance(insights['confidence'], (int, float)) else f"Confidence: {insights['confidence']}")
        except Exception as e:
            st.error(f"Error displaying confidence: {str(e)}")
    
        # Display predictions in a table if available
        try:
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
        except Exception as e:
            st.error(f"Error displaying predictions: {str(e)}")
    
        # Display detections in a table if available
        try:
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
        except Exception as e:
            st.error(f"Error displaying detections: {str(e)}")
        
        # Display labels if available
        try:
            if insights.get('labels') and len(insights['labels']) > 0:
                st.subheader("Labels")
                if isinstance(insights['labels'], list):
                    labels_text = ", ".join(str(label) for label in insights['labels'])
                    st.text(labels_text)
                else:
                    st.text(str(insights['labels']))
        except Exception as e:
            st.error(f"Error displaying labels: {str(e)}")
        
        # Display results if available
        try:
            if insights.get('results'):
                st.subheader("Results")
                if isinstance(insights['results'], (dict, list)):
                    st.json(insights['results'])
                else:
                    st.text(str(insights['results']))
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
        
        # Display analysis if available
        try:
            if insights.get('analysis'):
                st.subheader("Analysis")
                if isinstance(insights['analysis'], (dict, list)):
                    st.json(insights['analysis'])
                else:
                    st.text(str(insights['analysis']))
        except Exception as e:
            st.error(f"Error displaying analysis: {str(e)}")
        
        # Display objects if available
        try:
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
        except Exception as e:
            st.error(f"Error displaying objects: {str(e)}")
        
        # Display metadata if available
        try:
            if insights.get('metadata') and insights['metadata']:
                st.subheader("Metadata")
                st.json(insights['metadata'])
        except Exception as e:
            st.error(f"Error displaying metadata: {str(e)}")
        
        # Option to view raw response
        try:
            with st.expander("View Raw API Response"):
                st.json(insights.get('raw_response', {}))
        except Exception as e:
            st.error(f"Error displaying raw response: {str(e)}")
    
    except Exception as e:
        st.error(f"Error displaying DIAL insights: {str(e)}")


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
    try:
        # Clear previous insights when capturing a new snapshot
        st.session_state.dial_insights = None
        st.session_state.last_snapshot_file = None
        frame = None
        success_message = ""
        caption = ""
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if camera_source == "Test/Demo Mode":
            try:
                status_text.info("🔄 Generating test image...")
                progress_bar.progress(20)
                
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
                progress_bar.progress(50)
                status_text.success("✅ Test image generated successfully!")
            except Exception as e:
                st.error(f"Error generating test image: {str(e)}")
                frame = None
                progress_bar.progress(0)
                status_text.empty()
        
        elif camera_source == "Network Camera URL":
            if not camera_url or camera_url.strip() == "":
                st.error("Please enter a camera URL.")
                progress_bar.progress(0)
                status_text.empty()
            else:
                cap = None
                try:
                    status_text.info("🔄 Connecting to network camera...")
                    progress_bar.progress(10)
                    
                    # Try to access network camera
                    cap = cv2.VideoCapture(camera_url.strip())
                    progress_bar.progress(30)
                    
                    if cap.isOpened():
                        status_text.info("📸 Capturing frame from network camera...")
                        progress_bar.progress(50)
                        ret, frame = cap.read()
                        progress_bar.progress(70)
                        
                        if ret and frame is not None:
                            success_message = f"Snapshot captured from network camera!"
                            caption = f"Network Camera: {camera_url[:50]}..."
                            progress_bar.progress(80)
                            status_text.success("✅ Frame captured successfully!")
                        else:
                            st.error("Failed to capture frame from network camera.")
                            progress_bar.progress(0)
                            status_text.empty()
                    else:
                        st.error(f"Unable to connect to camera URL: {camera_url}")
                        st.info("Make sure the URL is correct and the camera is accessible.")
                        progress_bar.progress(0)
                        status_text.empty()
                except cv2.error as e:
                    st.error(f"OpenCV error while accessing network camera: {str(e)}")
                    progress_bar.progress(0)
                    status_text.empty()
                except Exception as e:
                    st.error(f"Unexpected error accessing network camera: {str(e)}")
                    progress_bar.progress(0)
                    status_text.empty()
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception as e:
                            st.error(f"Error releasing camera: {str(e)}")
    
        else:  # Local Webcam
            # Try to access webcam - try multiple indices
            cap = None
            camera_index = None
            
            try:
                status_text.info("🔍 Searching for available cameras...")
                progress_bar.progress(5)
                
                # Try indices 0-3 to find an available camera
                for idx in range(4):
                    test_cap = None
                    try:
                        status_text.info(f"🔍 Checking camera {idx}...")
                        progress_bar.progress(10 + (idx * 15))
                        test_cap = cv2.VideoCapture(idx)
                        if test_cap.isOpened():
                            ret, test_frame = test_cap.read()
                            if ret and test_frame is not None:
                                cap = test_cap
                                camera_index = idx
                                status_text.success(f"✅ Camera {idx} found and ready!")
                                progress_bar.progress(70)
                                break
                            else:
                                test_cap.release()
                        else:
                            test_cap.release()
                    except cv2.error as e:
                        st.error(f"OpenCV error accessing camera {idx}: {str(e)}")
                        if test_cap is not None:
                            try:
                                test_cap.release()
                            except:
                                pass
                    except Exception as e:
                        st.error(f"Unexpected error accessing camera {idx}: {str(e)}")
                        if test_cap is not None:
                            try:
                                test_cap.release()
                            except:
                                pass
                
                if cap is not None and cap.isOpened():
                    try:
                        status_text.info("📸 Capturing frame from webcam...")
                        progress_bar.progress(75)
                        # Capture a frame
                        ret, frame = cap.read()
                        progress_bar.progress(80)
                        
                        if ret and frame is not None:
                            success_message = f"Snapshot captured successfully from camera {camera_index}!"
                            caption = f"Local Camera {camera_index}"
                            status_text.success("✅ Frame captured successfully!")
                        else:
                            st.error("Failed to capture frame from webcam.")
                            progress_bar.progress(0)
                            status_text.empty()
                    except cv2.error as e:
                        st.error(f"OpenCV error capturing frame: {str(e)}")
                        progress_bar.progress(0)
                        status_text.empty()
                    except Exception as e:
                        st.error(f"Unexpected error capturing frame: {str(e)}")
                        progress_bar.progress(0)
                        status_text.empty()
                else:
                    st.error("Unable to access webcam. Please make sure your webcam is connected and not being used by another application.")
                    st.info("**Note for WSL2 users:** WSL2 doesn't have direct access to USB devices. You may need to:")
                    st.info("1. Use USB/IP to forward the webcam to WSL2, or")
                    st.info("2. Run this app on Windows directly, or")
                    st.info("3. Use a network camera stream instead (select 'Network Camera URL' option above)")
                    progress_bar.progress(0)
                    status_text.empty()
            except Exception as e:
                st.error(f"Unexpected error during webcam access: {str(e)}")
                progress_bar.progress(0)
                status_text.empty()
            finally:
                # Release the VideoCapture object
                if cap is not None:
                    try:
                        cap.release()
                    except Exception as e:
                        st.error(f"Error releasing camera: {str(e)}")
    
        # Display the captured frame if available
        if frame is not None:
            try:
                # Save the captured frame as an image file
                # Create snapshots directory if it doesn't exist
                snapshots_dir = "snapshots"
                try:
                    status_text.info("📁 Preparing to save snapshot...")
                    progress_bar.progress(82)
                    if not os.path.exists(snapshots_dir):
                        os.makedirs(snapshots_dir)
                        status_text.info("📁 Created snapshots directory")
                except OSError as e:
                    st.error(f"Error creating snapshots directory: {str(e)}")
                    frame = None
                    progress_bar.progress(0)
                    status_text.empty()
                    return
                
                # Generate filename with timestamp
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{snapshots_dir}/snapshot_{timestamp}.jpg"
                    progress_bar.progress(85)
                except Exception as e:
                    st.error(f"Error generating filename: {str(e)}")
                    frame = None
                    progress_bar.progress(0)
                    status_text.empty()
                    return
                
                # Save the frame using OpenCV's imwrite
                try:
                    status_text.info("💾 Saving snapshot to disk...")
                    progress_bar.progress(88)
                    if not cv2.imwrite(filename, frame):
                        st.error(f"Failed to save image to {filename}. Check file permissions and disk space.")
                        frame = None
                        progress_bar.progress(0)
                        status_text.empty()
                        return
                    status_text.success("✅ Snapshot saved successfully!")
                    progress_bar.progress(90)
                except cv2.error as e:
                    st.error(f"OpenCV error saving image: {str(e)}")
                    frame = None
                    progress_bar.progress(0)
                    status_text.empty()
                    return
                except Exception as e:
                    st.error(f"Unexpected error saving image: {str(e)}")
                    frame = None
                    progress_bar.progress(0)
                    status_text.empty()
                    return
                
                # Convert BGR to RGB for Streamlit display
                try:
                    status_text.info("🖼️ Processing image for display...")
                    progress_bar.progress(92)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=caption, use_container_width=True)
                    progress_bar.progress(95)
                except cv2.error as e:
                    st.error(f"OpenCV error converting image color: {str(e)}")
                    st.warning("Image saved but could not be displayed.")
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    st.warning("Image saved but could not be displayed.")
                
                if success_message:
                    st.success(success_message)
                st.info(f"Snapshot saved as: `{filename}`")
                
                # Store the snapshot filename in session state
                st.session_state.last_snapshot_file = filename
                
                # Run DIAL inference if enabled
                if enable_dial_inference:
                    st.divider()
                    st.subheader("DIAL API Insights")
                    
                    # Create separate progress bar for inference
                    inference_progress = st.progress(0)
                    inference_status = st.empty()
                    
                    try:
                        inference_status.info("📤 Sending image to DIAL API...")
                        inference_progress.progress(10)
                        
                        insights = run_dial_inference(filename)
                        inference_progress.progress(60)
                        
                        inference_status.info("🔄 Processing API response...")
                        inference_progress.progress(80)
                        
                        # Store insights in session state
                        st.session_state.dial_insights = insights
                        
                        inference_status.info("📊 Extracting insights...")
                        inference_progress.progress(90)
                        
                        # Display insights using helper function
                        display_dial_insights(insights, filename)
                        
                        inference_progress.progress(100)
                        inference_status.success("✅ DIAL inference completed successfully!")
                        
                        # Clear progress after a short delay
                        time.sleep(0.5)
                        inference_progress.empty()
                        inference_status.empty()
                        
                    except FileNotFoundError as e:
                        st.error(f"Image file not found: {str(e)}")
                        st.info("The snapshot file may have been deleted or moved.")
                        st.session_state.dial_insights = None
                        inference_progress.progress(0)
                        inference_status.empty()
                    except requests.exceptions.HTTPError as e:
                        st.error(f"HTTP error from DIAL API: {str(e)}")
                        st.info("Make sure the DIAL API endpoint is correctly configured and accessible.")
                        st.session_state.dial_insights = None
                        inference_progress.progress(0)
                        inference_status.empty()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Network error connecting to DIAL API: {str(e)}")
                        st.info("Check your internet connection and API endpoint URL.")
                        st.session_state.dial_insights = None
                        inference_progress.progress(0)
                        inference_status.empty()
                    except ValueError as e:
                        st.error(f"Invalid API response: {str(e)}")
                        st.info("The API response may not be in the expected format.")
                        st.session_state.dial_insights = None
                        inference_progress.progress(0)
                        inference_status.empty()
                    except Exception as e:
                        st.error(f"Unexpected error running DIAL inference: {str(e)}")
                        st.info("Make sure the DIAL API endpoint is correctly configured and accessible.")
                        st.session_state.dial_insights = None
                        inference_progress.progress(0)
                        inference_status.empty()
                else:
                    # Complete progress bar if inference is not enabled
                    progress_bar.progress(100)
                    status_text.success("✅ Snapshot capture completed!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
            except Exception as e:
                st.error(f"Unexpected error processing captured frame: {str(e)}")
    
    except Exception as e:
        st.error(f"Unexpected error during snapshot capture: {str(e)}")
        st.info("Please try again or check the application logs for more details.")

# Display stored insights if available (from previous capture)
try:
    if st.session_state.dial_insights is not None and st.session_state.last_snapshot_file:
        st.divider()
        st.subheader("Latest DIAL API Insights")
        display_dial_insights(st.session_state.dial_insights, st.session_state.last_snapshot_file)
except Exception as e:
    st.error(f"Error displaying stored insights: {str(e)}")
    # Clear invalid insights from session state
    st.session_state.dial_insights = None
