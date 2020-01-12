# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-e", "--embeddings", required=True,
#	help="path to serialized db of facial embeddings")
#ap.add_argument("-r", "--recognizer", required=True,
#	help="path to output model trained to recognize faces")
#ap.add_argument("-l", "--le", required=True,
#	help="path to output label encoder")
#args = vars(ap.parse_args())

args_embeddings = "C:\\Users\\arnab\\Desktop\\AL\\face_recognition\\opencv-face-recognition\\dataset_3i_out\\embeddings.pickle"
args_recognizer = "C:\\Users\\arnab\\Desktop\\AL\\face_recognition\\opencv-face-recognition\\dataset_3i_out\\recognizer.pickle"
args_le = "C:\\Users\\arnab\\Desktop\\AL\\face_recognition\\opencv-face-recognition\\dataset_3i_out\\le.pickle"

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args_embeddings, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args_recognizer, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args_le, "wb")
f.write(pickle.dumps(le))
f.close()