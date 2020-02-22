import tensorflow as tf
keras, L = tf.keras, tf.keras.layers

import matplotlib.pyplot as plt
import numpy as np


def choose_images(model, directory_path=None, imagegen=None, X_val=None, y_val=None):
    if directory_path:
        attack_gen = imagegen.flow_from_directory(directory_path, class_mode="categorical",
                                                  shuffle=True, batch_size=1, target_size=(224, 224))
        while True:
            attacked_img, attacked_class = attack_gen.next()
            changed_img, changed_class = attack_gen.next()

            attacked_img = tf.cast(attacked_img, tf.float32)
            changed_img = tf.cast(changed_img, tf.float32)

            attacked_class = np.argmax(attacked_class[0])
            changed_class = np.argmax(changed_class[0])

            attacked_pred = np.argmax(model(attacked_img).numpy()[0])
            changed_pred = np.argmax(model(changed_img).numpy()[0])

            if attacked_class == attacked_pred and changed_class == changed_pred and attacked_pred != changed_pred:
                break

    else:
        while True:
            att_i, ch_i = np.random.randint(len(X_val), size=2)

            attacked_img = X_val[att_i]
            changed_img = X_val[ch_i]

            attacked_img = tf.cast(tf.reshape(attacked_img, [1, *attacked_img.shape]), tf.float32)
            changed_img = tf.cast(tf.reshape(changed_img, [1, *changed_img.shape]), tf.float32)

            attacked_class = y_val[att_i]
            changed_img = y_val[ch_i]

            attacked_pred = np.argmax(model(attacked_img).numpy()[0])
            changed_pred = np.argmax(model(changed_img).numpy()[0])

            if attacked_class == attacked_pred and changed_class == changed_pred and attacked_pred != changed_pred:
                break


    print("Attacked image: ")
    print("Logits: ", model(attacked_img).numpy()[0])
    print("Predicted: ", attacked_pred)
    print("Real: ", attacked_class)
    plt.imshow(attacked_img[0])
    plt.show()

    print("Changing image: ")
    print("Logits: ", model(changed_img).numpy()[0])
    print("Predicted: ", changed_pred)
    print("Real: ", changed_class)
    plt.imshow(changed_img[0])
    plt.show()
    
    return attacked_img, changed_img


def attack(model, attacked_img, changed_img, eps=1, fn=tf.identity, iterations=40, with_jpeg=False):  
    y = model(attacked_img)
    first_img = tf.identity(changed_img)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(changed_img)
            y_pred = model(changed_img)
            loss = tf.reduce_mean((y - y_pred) ** 2)

        gradients, = tape.gradient(loss, changed_img)
        
        changed_img = tf.clip_by_value(changed_img - eps * fn(gradients), 0, 1)

        to_uint_img = tf.cast((changed_img * 255), tf.uint8)
        if with_jpeg:
            val_img = tf.cast(tf.io.decode_jpeg(tf.io.encode_jpeg(to_uint_img[0])), tf.float32) / 255. 
            val_img = tf.expand_dims(val_img, 0)
        else:
            val_img = tf.cast(to_uint_img, tf.float32) / 255.
        val_pred = model(val_img)
        val_loss = tf.reduce_mean((y - val_pred) ** 2)

        if i % 10 == 0:
            print("Loss : ", val_loss.numpy())
            print("Predict of changing image: ", np.argmax(val_pred[0]))
            print("Logits of changing image: ", val_pred.numpy()[0])
            print("Logits of attacked image: ", y.numpy()[0])
            
            print("Changed image:")
            plt.imshow(val_img[0, :])
            plt.show()

            print("Source image to be changed:")
            plt.imshow(first_img[0, :])
            plt.show()
