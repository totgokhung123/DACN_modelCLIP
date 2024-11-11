import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

def extract_frames_from_video(video_path, output_dir, threshold=30.0):
    os.makedirs(output_dir, exist_ok=True)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        middle_frame = (start_frame + end_frame) // 2
        print(middle_frame)
        cap = cv2.VideoCapture(video_path)
        
        #frames_to_capture = {'begin': start_frame, 'middle': middle_frame, 'end': end_frame}
        frames_to_capture = {middle_frame}
        
        for frame_idx in frames_to_capture:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_dir, f"{frame_idx}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Đã lưu {frame_filename}")

        cap.release()

    video_manager.release()
    print("Trích xuất hoàn tất.")

video_path = "E:\\THIHE\\testfitty one\\videotesst.mp4" 
output_dir = "E:\\THIHE\\testfitty one\\SegmentVideo\\seg1\\SegmentVideo"  

# Gọi hàm trích xuất khung hình
extract_frames_from_video(video_path, output_dir)