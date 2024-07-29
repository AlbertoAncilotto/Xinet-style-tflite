import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

onnx_models_dir = 'onnx_fp32'
base_dir = 'demo_images' #calibration data

def convert_onnx(onnx_model_path, model_name='model'):
    # Step 1: Convert ONNX to TensorFlow
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Extract input shape
    input_tensor = onnx_model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]

    print('model input:', input_shape)

    # Convert the ONNX model to a TensorFlow model
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("tensorflow_model/style_transfer")

    # Step 2: Convert TensorFlow Model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("tensorflow_model/style_transfer")

    def representative_dataset():
        for img_path in tqdm(os.listdir(base_dir)):
            img = cv2.imread(os.path.join(base_dir, img_path))
            img = cv2.resize(img, (input_shape[2], input_shape[3]))
            content_tensor = np.transpose(img, (2, 0, 1))
            content_tensor = np.expand_dims(content_tensor, axis=0).astype(np.float32)/255.0
            yield [content_tensor] 

    converter.representative_dataset = representative_dataset
    converter._experimental_disable_per_channel = True
    # converter.experimental_new_converter = False #for per tensor quantization

    # Convert the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant = converter.convert()

    # Save the quantized model
    with open(f"tflite_int8/{model_name}.tflite", "wb") as f:
        f.write(tflite_model_quant)
        
    shutil.rmtree('tensorflow_model')

if __name__=="__main__":
    for model_path in os.listdir(onnx_models_dir):
        convert_onnx(os.path.join(onnx_models_dir, model_path), model_name=model_path.split('.')[0])
