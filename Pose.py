import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
from time import perf_counter

class PoseTracker:
   def __init__(self):
      """
      Initializing webcamera and mediapipe pose related 
      """
      
      # init for opencv webcam instance
      self.webcam = cv2.VideoCapture(0)
      self.start_time = perf_counter()
      
      # init for mediapipe pose model
      self.annotated_frame = None
      self.latest_results = None
      # self.latest_frame = None # This is only here if I need to check for things/need an image without pose
      self.base_options = mp_python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
      self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True,
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback = self.update_result)
      self.detector = vision.PoseLandmarker.create_from_options(self.options)

   def draw_landmarks_on_image(self, rgb_image, detection_result):
      """
      Draws pose points onto the given image

      Args:
        rgb_image: Image to be drawn on in RGB format
        detection_result: Result of model detection used to obtain pose coordinates
    
      Returns:
        Image with pose points drawn onto it
      """
      pose_landmarks_list = detection_result.pose_landmarks
      annotated_image = np.copy(rgb_image)
      pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
      pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

      for pose_landmarks in pose_landmarks_list:
         drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

      return annotated_image


   def update_result(self, result, output_image, timestamp_ms):
      """
      Updates variables based on results

      Args:
        result: Result of model detection on an image
        output_image: Image that detection was performed on
        timestamp_ms: Timestamp of the image in ms
    
      Returns:
        Nothing

      """
      rgb_frame = output_image.numpy_view()
      self.latest_results = result
      self.annotated_frame = self.draw_landmarks_on_image(rgb_frame, result)

   def async_detection(self, frame, timestamp):
      """
      Converts image frame to proper format and uses detect_async

      Args:
         frame: Image to perform detection on
         timestamp: Timestamp of the image in ms

      Returns:
         Nothing

      """
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_frame = mp.Image(image_format= mp.ImageFormat.SRGB, data= rgb_frame)
      self.detector.detect_async(mp_frame, timestamp)
   
   def draw(self):
       """
       Displays the webcamera feed + pose points
       """
       if self.webcam.isOpened(): # try to get the first frame
          rval, frame = self.webcam.read()

       else:
          rval = False

       while rval:
          rval, frame = self.webcam.read()
          frame_timestamp = int((perf_counter() - self.start_time) * 1000)
          self.async_detection(frame, int(frame_timestamp))
          if self.annotated_frame is not None:
             cv2.imshow("preview", self.annotated_frame)

          # Using key to exit webcam; remove later in favor for the UI   
          key = cv2.waitKey(20)
          if key == 27: # exit on ESC
            break

       cv2.destroyWindow("preview")
       self.webcam.release()


# Testing purposes
# pose_landmarker = PoseTracker()
# pose_landmarker.draw()

