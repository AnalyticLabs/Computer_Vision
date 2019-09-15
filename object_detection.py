from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")


def object_detection(input_file,out_name,model='ResNet'):
    video_detector = VideoObjectDetection()

    if model == "ResNet":
        video_detector.setModelTypeAsRetinaNet()
        video_detector.setModelPath(os.path.join(execution_path, "pretranined_models/resnet50_coco_best_v2.0.1.h5"))

    elif model == "Yolo":
        video_detector.setModelTypeAsYOLOv3()
        video_detector.setModelPath(os.path.join(execution_path, "pretranined_models/yolo.h5"))

    else:
        video_detector.setModelTypeAsTinyYOLOv3()
        video_detector.setModelPath(os.path.join(execution_path, "pretranined_models/yolo-tiny.h5"))

    video_detector.loadModel()

    vi = video_detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, input_file),
        output_file_path=os.path.join(execution_path, out_name),
        frames_per_second=10,
        per_second_function=forSeconds,
        per_frame_function=forFrame,
        per_minute_function=forMinute,
        minimum_percentage_probability=30
    )

input_file = 'traffic-mini.mp4'
out_name = 'traffic_detected'
# object_detection(input_file,out_name)

# model = 'ResNet'
# model = 'Yolo'
model = 'Tiny-Yolo'
object_detection(input_file,out_name,model)