import cv2
import numpy as np
import onnxruntime as ort
import os
import postprocess
from tqdm import tqdm

STYLE_TRANSFORM_PATH =  'onnx_fp32/' 
IMAGE_FOLDER = 'demo_images'

def process_images(style_transform_paths, image_folder, out_size=(320,320)):
    """
    Processes images from a folder using style transfer models.
    Shows the styled images in separate windows.
    """
    # Load ONNX models
    ort_sessions = []
    ort_names = []
     # each model has its own input shape to keep feature size consistent and achieve target MACC complexity
    for path in os.listdir(style_transform_paths):
        if '.onnx' in path:
            ort_names.append(path.split('.')[0])
            sess = ort.InferenceSession(os.path.join(style_transform_paths,path))
            ort_sessions.append(sess)
            print('model loaded:', path)

    input_names = [session.get_inputs()[0].name for session in ort_sessions]
    output_names = [session.get_outputs()[0].name for session in ort_sessions]
    input_shapes = [session.get_inputs()[0].shape for session in ort_sessions]

    # Iterate through the folder of images
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        cv2.imshow('Input', cv2.resize(img, out_size))
        if img is None:
            continue
        
        # Apply each model to the image
        for (session, input_name, output_name, in_shape, model_name) in tqdm(zip(ort_sessions, input_names, output_names, input_shapes, ort_names)):

            img = cv2.resize(img, (in_shape[2], in_shape[3]))
            content_tensor = np.transpose(img, (2, 0, 1))
            content_tensor = np.expand_dims(content_tensor, axis=0).astype(np.float32) / 255.0
            generated_tensor = session.run([output_name], {input_name: content_tensor})[0]
            generated_image = generated_tensor.squeeze()
            generated_image = generated_image.transpose(1, 2, 0).astype(np.uint8)
            
            # Show the generated image
            window_name = f'Model: {model_name}'
            generated_image = cv2.resize(generated_image, out_size)
            generated_image = postprocess.process_frame(generated_image)
            cv2.imshow(window_name, generated_image)
        
        # Wait for a key press to proceed to the next image
        key = cv2.waitKey(0)
        if key == 27:  # esc to quit
            break
    
    cv2.destroyAllWindows()


# Run the image processing
process_images(STYLE_TRANSFORM_PATH, IMAGE_FOLDER)
