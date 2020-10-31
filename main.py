import argparse
import cv2
import imutils
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


def load_image(img_path, width=500):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=width)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    return image

def deprocess(image):
    image = 255 * (image + 1.0) / 2.0
    return tf.cast(image, tf.uint8)

def calculate_loss(image, model):
    image = tf.expand_dims(image, axis=0)
    layer_activations = model(image)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = []
    # Find mean of activations in the chosen layers
    for activation in layer_activations:
        loss = tf.reduce_mean(activation)
        losses.append(loss)
    # Loss is the sum of all means
    return tf.reduce_sum(losses)

@tf.function
def deepdream(image, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = calculate_loss(image, model)

    # Calcaluate gradient of loss wrt. input image
    gradients = tape.gradient(loss, image)
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    # Gradient ascent to amplify activations
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)
    return loss, image

def run_deepdream(image, model, steps=100, step_size=0.01):
    image = preprocess_input(image)
    image = tf.convert_to_tensor(image)

    for step in range(steps):
        loss, image = deepdream(image, model, step_size)
        print(f"[INFO] Step {step}: {loss:.4f}")
    return deprocess(image)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()

    # Load image
    image = load_image(args.image)

    # There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'.
    # Using different layers will result in different dream-like images.
    # Deeper layers respond to higher-level features (such as eyes and faces),
    # while earlier layers respond to simpler features (such as edges, shapes, and textures).
    base_model = InceptionV3(weights="imagenet", include_top=False)
    layer_names = ["mixed3", "mixed5"]
    layers = [base_model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    image = tf.constant(image)
    base_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

    # Octave scale
    scale = 1.30
    # Apply gradient acsent at different scales (multiple octaves)
    for n in range(-2, 5):
        print(f"[INFO] Octave {n} ...")
        new_shape = tf.cast(base_shape * (scale ** n), tf.int32)
        image = tf.image.resize(image, new_shape).numpy()
        image = run_deepdream(image, model, steps=100, step_size=0.001)

    image = np.array(image)
    Image.fromarray(image).save("output.png")


if __name__ == '__main__':
    main()
