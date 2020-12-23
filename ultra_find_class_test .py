#USAGE  :python ultra_find_class_test .py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk
import time
import cv2
from itertools import count
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import os


class   Application(Tk):
	def __init__(self):
		Tk.__init__(self)
		self.title("GE.ENSA.KHOURIBGA")
		self.geometry("1250x520")
		self.config(bg = "#660033")
		#self.iconbitmap('oo.ico')
		
		self.vs = My_image()

		self.fram1 = Frame(self, width = 900,bg = 'black')
		text = Label(self.fram1, text = "ULTRA_FIND", bg = "black", fg = "white", font = ("Algerian", 30))
		text2 = Label(self.fram1, text = "Created By: El-khamsaoui Ousama", bg = "black", fg = "white", font = ("Algerian", 10))
		text2.grid(row = 1, column  = 3, ipadx = 600)
		text.grid(row =1, column = 1, ipadx = 20)

		self.frame_t = Frame(self, width = 900,bg = 'black')
		text_t = Label(self.frame_t, text = '"El-khamsaoui Ousama"', bg = "black", fg = "red", font = ("Algerian", 12))
		text_t.grid(row =1, column = 1)

		self.fram3  = Frame(self, bg ="#660033")
		self.can = Canvas(self.fram3, width = 480, height = 370, bg = "black")
		self.can.grid(row = 2, column = 1, rowspan = 5)
		
		self.can2 = Canvas(self.fram3, width = 480, height = 370, bg = "black")
		self.can2.grid(row = 2, column = 3, rowspan = 5)

		self.can3 = Canvas(self.fram3, width = 220, height = 370, bg = "black")
		self.can3.grid(row = 2, column = 2, rowspan = 5 , padx = 15)
		self.load()

		
		self.frame4 = Frame(self, bg = "#660033", width = 1060, height = 110)

		self.but2 = Button(self.frame4, text = "face recognition",bg = "#800060", width = 15,fg = "white" , height= 0 , font = ("Berlin Sans FB Demi", 0), command = self.update)
		self.but2.grid(row = 1, column = 1, padx = 10)

		self.but3 = Button(self.frame4, text = "Tack snap",bg = "#800060",fg = "white" , width = 15, height= 0 , font = ("Berlin Sans FB Demi", 0), command =self.snap)
		self.but3.grid(row = 1, column = 2, padx = 10)


		self.frame5 = Frame(self, bg = "#660033", width = 1060, height = 110)
		self.but4 = Button(self.frame5, text = "snap mask",bg = "#800060",fg = "white" , width = 15, height= 0 , font = ("Berlin Sans FB Demi", 0) ,command = self.snap_mask)
		self.but4.grid(row = 1, column = 4, padx = 10)

		self.but5 = Button(self.frame5, text="Mask detector", bg="#800060", fg="white", width=15, height=0,font = ("Berlin Sans FB Demi", 0), command=self.update_mask)
		self.but5.grid(row=1, column=3, padx=10)

		self.frame6 = Frame(self, bg = "#660033", width = 1060, height = 110)
		self.but6 = Button(self.frame6, text="Stop Camera", bg="#800060", fg="white", width=15, height=0 ,font = ("Berlin Sans FB Demi", 0), command=self.vs.stopcamera)
		self.but6.grid(row=1, column=5, padx=10) 

		self.but7 = Button(self.frame6, text="Exit", bg="red", fg="white", width=10, height=0 ,font = ("Berlin Sans FB Demi", 0), command=exit)
		self.but7.grid(row=2, column=5, padx=10, pady = 5)


		self.fram1.place(relx = 0, rely = 0)
		self.frame_t.place(relx = 0.77, rely = 0.026)
		self.fram3.place(relx = 0.01, rely = 0.12)
		self.frame4.place(relx = 0.045, rely = 0.90)
		self.frame5.place(relx = 0.64, rely = 0.90)
		self.frame6.place(relx = 0.425, rely = 0.85)


	def snap(self):
		frame = self.vs.get_frame()
		cv2.imwrite("image-" + time.strftime("%d-%m-%Y") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	def snap_mask(self):
		frameee = self.vs.get_frame_mask()
		cv2.imwrite("image_mask-" + time.strftime("%d-%m-%Y") + ".jpg", cv2.cvtColor(frameee, cv2.COLOR_RGB2BGR))



	### animation de gif

	def load(self):
		im = Image.open("image.gif")
		self.loc = 0
		self.frames = []

		try:
			for i in count(1):
				self.frames.append(ImageTk.PhotoImage(im.copy()))
				im.seek(i)
		except EOFError:
			pass

		try :
			self.delay = im.info["duration"]
		except :
			self.delay = 100

		#if len(self.frames) == 1:
			#self.can2.create_image(200, 50, image = self.frames[0], anchor = NW)
		#else:
		self.next_frame()

	def next_frame(self):
		if self.frames:
			self.loc += 1
			self.loc %= len(self.frames)
			self.can3.create_image(-90, 0, image = self.frames[self.loc], anchor = NW)

			self.after(self.delay, self.next_frame)

	##################

	def update(self):
		
		self.frame2 = self.vs.get_frame()

		self.photo = ImageTk.PhotoImage(image = Image.fromarray(self.frame2))
		self.can.create_image(0, 0, image = self.photo, anchor = NW)
		self.after(self.delay, self.update)

	###########

	def update_mask(self):

		self.frame3 = self.vs.get_frame_mask()

		self.photo1 = ImageTk.PhotoImage(image = Image.fromarray(self.frame3))
		self.can2.create_image(0, 0, image = self.photo1, anchor = NW)
		print("ok")
		self.after(self.delay, self.update_mask)

	
	#################

class   mask():
	def __init__(self):
		

		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--face", type=str,default="face_detector",help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
		self.args = vars(ap.parse_args())

		# load our serialized face detector model from disk
		print("[INFO] loading face mask detector model...")
		prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
		weightsPath = os.path.sep.join([self.args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
		self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		self.maskNet = load_model(self.args["model"])


	def get_frame_mask(self):
		
		print("[INFO] starting video stream of mask...")
		self.vs = VideoStream(src=0 +cv2.CAP_DSHOW).start()


		self.frame = self.vs.read()
		self.frame = imutils.resize(self.frame, width=510)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = self.detect_and_predict_mask(self.frame, self.faceNet, self.maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "With mask" if mask > withoutMask else "No mask"
			color = (0, 255, 0) if label == "No mask" else (255, 0, 255)
			color1 = (255, 0, 0)

			# include the probability in the label
			#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
			label = "{}:".format(label)
			label2 = "{:.2f}%".format(max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			#cv2.putText(self.frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.putText(self.frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
			cv2.putText(self.frame, label2, (startX + 120, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
			cv2.rectangle(self.frame, (startX, startY), (endX, endY), color1, 2)

		return(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))


	def detect_and_predict_mask(self, frame, faceNet, maskNet):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		faceNet.setInput(blob)
		detections = faceNet.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > self.args["confidence"]:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			preds = maskNet.predict(faces)

		# return a 2-tuple of the face locations and their corresponding
		# locations
		return (locs, preds)


	def stopcamera(self):
		self.vs.stop()
		print("c'est arrtete")
		#time.sleep(2.0)


	def __del__(self):
		self.vs.stop()
		cv2.destroyAllWindows()




class   My_image(mask):
	def __init__(self):
		mask.__init__(self)

		print("[INFO] starting video stream...")
		self.vs = VideoStream(0 + cv2.CAP_DSHOW).start()
		time.sleep(2.0)

		ap = argparse.ArgumentParser()

		ap.add_argument("-d", "--detector", type = str, default= "face_detection_model" , help="path to OpenCV's deep learning face embedding model")
		ap.add_argument("-m", "--embedding-model", type = str, default= "openface_nn4.small2.v1.t7", help="path to OpenCV's deep learning face embedding model")
		ap.add_argument("-r", "--recognizer", type = str, default= "output/recognizer.pickle", help="path to model trained to recognize faces")
		ap.add_argument("-l", "--le", type = str, default= "output/le.pickle", help="path to label encoder")
		ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

		self.args = vars(ap.parse_args())

		
			
		# load our serialized face detector from disk
		print("[INFO] loading face detector...")
		protoPath = os.path.sep.join([self.args["detector"], "deploy.prototxt"])
		modelPath = os.path.sep.join([self.args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
		self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

		# load our serialized face embedding model from disk
		print("[INFO] loading face recognizer...")
		self.embedder = cv2.dnn.readNetFromTorch(self.args["embedding_model"])

		# load the actual face recognition model along with the label encoder
		self.recognizer = pickle.loads(open(self.args["recognizer"], "rb").read())
		self.le = pickle.loads(open(self.args["le"], "rb").read())

		#######



	

	def get_frame(self):

		self.frame = self.vs.read()

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		self.frame = imutils.resize(self.frame, width=500)
		(h, w) = self.frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		self.detector.setInput(imageBlob)
		detections = self.detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):

		# extract the confidence (i.e., probability) associated with
		# the prediction
			confidence = detections[0, 0, i, 2]

		# filter out weak detections
			if confidence > self.args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for
		# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
				face = self.frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

			# perform classification to recognize the face
				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = self.le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
				#print(startX)
				text = "{}:".format(name)
				text2 = "{:.2f}%".format(proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				color2 = (255, 0, 0)
				'''if name == "Ousama":
					color = (0,255,0)
				elif name == "Meriem":
					color = (102, 255, 255)
				elif name == "Unknown":
					color = (0,0,255)
				if name == "Kroos":
					color = (255, 0, 255)'''
				color = self.name_choise(name)
				cv2.rectangle(self.frame, (startX, startY), (endX, endY), color2, 2)
				cv2.putText(self.frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
				cv2.putText(self.frame, text2, (startX + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		return(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))


	def name_choise(self, name):
		if name == "Ousama":
			color = (0,255,0)
			return(color)
		elif name == "Unknown":
			color = (51, 0, 51)
			return(color)
		elif name == "Kroos":
			color = (255, 0, 255)
			return(color)
		elif name == "D.Slaoui":
			#color = (102, 255, 153)
			color = (0, 0, 204)

			return(color)



	def __del__(self):
		self.vs.stop()
		cv2.destroyAllWindows()


root = Application()

root.mainloop()



