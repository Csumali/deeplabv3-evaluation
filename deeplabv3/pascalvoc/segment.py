import tensorflow as tf
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Load the model
model_path = "deeplabv3_plus_mobilenet_quantized.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
height, width = input_shape[1], input_shape[2]

# Load the data
image_folder = r"VOCdevkit/VOC2012/JPEGImages"
val_list = r"VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

with open(val_list, "r") as file:
    image_names = file.read().splitlines()

image_paths = [os.path.join(image_folder, f"{name}.jpg") for name in image_names]

# Create output directory
output_dir = "VOC2012_val_results"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path, model_shape, model_dtype):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = model_shape[1], model_shape[2]
    image = cv2.resize(image, (width, height))

    image = np.expand_dims(image, axis=0)

    if model_dtype == np.float32:
        image = np.float32(image) / 255.0
    else:
        image = image.astype(np.uint8)
    
    return image

def voc_colormap(num_classes=21):
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        for j in range(8):
            r |= ((i >> (0 + j * 3)) & 1) << (7 - j)
            g |= ((i >> (1 + j * 3)) & 1) << (7 - j)
            b |= ((i >> (2 + j * 3)) & 1) << (7 - j)
        cmap[i] = [r, g, b]
    return cmap

def apply_colormap(segmentation_mask):
    cmap = voc_colormap()
    return cmap[segmentation_mask]

# Process all validation images
for image_path in image_paths:
    filename = os.path.basename(image_path)
    
    sample_image = preprocess_image(image_path, input_shape, input_dtype)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sample_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if output.shape[0] == 1:
        output = np.squeeze(output)

    if len(output.shape) == 3 and output.shape[-1] > 1:
        output = np.argmax(output, axis=-1)

    if output.shape != (height, width):
        output = output.reshape((height, width))

    colored_output = apply_colormap(output)

    # Save results
    output_filename = os.path.join(output_dir, f"segmented_{filename}")
    cv2.imwrite(output_filename, cv2.cvtColor(colored_output, cv2.COLOR_RGB2BGR))
