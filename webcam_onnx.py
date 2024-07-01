import cv2
import numpy as np
import onnxruntime as ort
import os

STYLE_TRANSFORM_PATH =  'onnx_fp32/' 

def webcam(style_transform_paths, out_size=(460,460)):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 
    """
    # Load ONNX model
    ort_sessions=[]
    for path in os.listdir(STYLE_TRANSFORM_PATH):
        ort_sessions.append(ort.InferenceSession(os.path.join(STYLE_TRANSFORM_PATH, path)))
    input_name = ort_sessions[0].get_inputs()[0].name
    output_name = ort_sessions[0].get_outputs()[0].name
    input_shapes = [session.get_inputs()[0].shape[2:] for session in ort_sessions]

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    model_id = 0

    # Main loop
    while True:
        # Get webcam input
        _, img = cam.read()

        # Mirror and resize
        img = cv2.flip(img, 1)
        img = cv2.resize(img, input_shapes[model_id])

        # Generate image
        content_tensor = np.transpose(img, (2, 0, 1))
        content_tensor = np.expand_dims(content_tensor, axis=0).astype(np.float32)/255.0
        
        generated_tensor = ort_sessions[model_id].run([output_name], {input_name: content_tensor})[0]
        generated_image = generated_tensor.squeeze()
        generated_image = generated_image.transpose(1, 2, 0)

        generated_image = generated_image.astype(np.uint8)
        generated_image = cv2.resize(generated_image, out_size, cv2.INTER_NEAREST)

        # Show webcam
        cv2.imshow('Demo webcam', generated_image)
        key = cv2.waitKey(1)
        if key != -1: 
            if key == 27:
                break  # esc to quit
            else:
                model_id += 1
                model_id %= len(ort_sessions)

    cam.release()
    cv2.destroyAllWindows()


webcam(STYLE_TRANSFORM_PATH)