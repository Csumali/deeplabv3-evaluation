import tensorflow as tf
import cv2
import numpy as np
import os
import glob

# Load the model
model_path = "deeplabv3_mobilenetv2_ade20k.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
height, width = input_shape[1], input_shape[2]

# Load the data
image_folder = "ADEChallengeData2016/images/validation"
annotation_folder = "ADEChallengeData2016/annotations/validation"

image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Create output directory
output_dir = "ADE20K_results"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path, model_shape, model_dtype):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = model_shape[1], model_shape[2]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    if model_dtype == np.float32:
        image = np.float32(image) / 255.0
    else:
        image = np.uint8(image)

    return np.expand_dims(image, axis=0)

# Process all validation images
for image_path in image_paths:
    filename = os.path.basename(image_path)

    sample_image = preprocess_image(image_path, input_shape, input_dtype)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sample_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    output = np.squeeze(output)
    output = np.argmax(output, axis=-1)

    output_filename = os.path.join(output_dir, f"segmented_{filename.replace('.jpg', '.png')}")
    cv2.imwrite(output_filename, output)