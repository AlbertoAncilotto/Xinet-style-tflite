import cv2
import numpy as np
import tensorflow.lite as tflite
import os
import postprocess

STYLE_TRANSFORM_PATH = 'tflite_int8\\picasso_muse_a75_160_nomp.tflite'
IMAGE_PATH = 'demo_images\\tree_coco.jpg'
IMG_SZ = (160, 160) #hardcoded for easier translation to C code
SCALE_INPUT = False 

# Load TFLite model, create interpreter, get input and output names
interpreter = tflite.Interpreter(model_path=os.path.join(STYLE_TRANSFORM_PATH))
interpreter.allocate_tensors()
input_name = interpreter.get_input_details()[0]['index']
output_name = interpreter.get_output_details()[0]['index']

# Read and resize input image
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, IMG_SZ)
cv2.imshow('input', img)

# Preprocess image: Reshape to (channels, width, heigth) -> normalize in [0,1]
content_tensor = np.transpose(img, (2, 0, 1))
content_tensor = np.expand_dims(content_tensor, axis=0).astype(np.float32) 
content_tensor = content_tensor/ 255.0 if SCALE_INPUT else content_tensor
#/ 255.0

# Run model
np.save('input_tensor.npy', content_tensor)
interpreter.set_tensor(input_name, content_tensor)
interpreter.invoke()
generated_tensor = interpreter.get_tensor(output_name)
np.save('output_tensor.npy', generated_tensor)

# Postprocess image: Reshape to (width, heigth, channels) -> output is already in [0,255], just convert to int
generated_image = generated_tensor.squeeze()
generated_image = generated_image.transpose(1, 2, 0).astype(np.uint8)

#optional: apply smoothing and contrast normalization to remove conv artifacts
generated_image = postprocess.process_frame(generated_image) 

# Show webcam
cv2.imshow('output', generated_image)
cv2.waitKey(-1)