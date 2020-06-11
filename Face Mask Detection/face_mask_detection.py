import cv2
import tensorflow as tf
print(tf.__version__)
from mtcnn.mtcnn import MTCNN
import numpy as np


def pred_mask(img, model):
	pred = model.predict(img.reshape(1, 224, 224, 3))
	mask = np.argmax(pred)

	confidence = round(pred[0][mask], 2)

	if mask == 1:
		return 'Mask', confidence
	else:
		return 'No Mask', confidence


def detect_mask(image, detector, model):
	original_image = image
	faces = detector.detect_faces(original_image)

	if len(faces) == 0:
		print('No faces found')
		return original_image

	else:

		faces = [tuple(face['box']) for face in faces]

		for face in faces:
			x1, y1, width, height = face
			x2, y2 = x1 + width, y1 + height

			face_roi = original_image[y1:y2, x1:x2]
			resized_img = cv2.resize(face_roi, (224, 224))

			input_img = tf.keras.applications.mobilenet_v2.preprocess_input(resized_img)

			mask, confidence = pred_mask(input_img, model)
			print('Mask: {}, Confidence: {}'.format(mask, confidence))

			original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 204, 204), 1)
			cv2.putText(original_image, mask + ' ' + str(confidence), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 204, 204), 2)

		return original_image



model = tf.keras.models.load_model('D:/Code/Resources/models/face_mask_detection/mask_model.h5')
detector = MTCNN()
video = cv2.VideoCapture('C:/Users/yasho/Desktop/mask_demo.mp4')


width, height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
fourcc_video = cv2.VideoWriter_fourcc(*'XVID')
print(fourcc_video)
video_writer = cv2.VideoWriter('C:/Users/yasho/Desktop/mask_demo_output.avi', fourcc_video, 30, (int(width//1.5), int(height//1.5)))

while True:
	ret, frame = video.read()

	if ret is not True:
		video.release()
		break

	else:

		try:
			frame = detect_mask(frame, detector, model)
		except Exception as e:
			print(e)

		video_writer.write(frame)
		# cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

video.release()
cv2.destroyAllWindows()