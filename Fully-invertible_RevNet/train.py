import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import ops
import revnet


def get_hparams_imagenette():
    config = ops.HParams()
    config.add_hparam("init_filters", 32)
    config.add_hparam("n_classes", 10)
    config.add_hparam("n_rev_blocks", 22)
    config.add_hparam("ratio", ([2] + [1] * 10) * 2)
    config.add_hparam("batch_size", 32)
    config.add_hparam("bottleneck", False)
    config.add_hparam("fused", True)
    config.add_hparam("input_shape", (224, 224, 3))
    config.add_hparam("data_format", "channels_last")
    config.add_hparam("dtype", tf.float32)

    config.add_hparam("epochs", 20)
    config.add_hparam("weight_decay", 1e-4)

    return config


if __name__ == "__main__":
    config = get_hparams_imagenette()

    imagegen_train = ImageDataGenerator(rescale=1 / 255)
    imagegen_val = ImageDataGenerator(rescale=1 / 255)

    train = imagegen_train.flow_from_directory("imagenette2/train/", class_mode="sparse", shuffle=True,
                                               batch_size=config.batch_size, target_size=(224, 224))
    val = imagegen_val.flow_from_directory("imagenette2/val/", class_mode="sparse", shuffle=False,
                                           batch_size=config.batch_size, target_size=(224, 224))

    model = revnet.RevNet(config=config)
    model_name = "model_with_two_ratios"

    max_val_acc = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + model_name + '/' + current_time + '/train'
    test_log_dir = 'logs/' + model_name + '/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

    for epoch in range(config.epochs):
        train_mean_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

        for step in range(int(np.ceil(train.samples / train.batch_size))):
            x_batch_train, y_batch_train = train.next()

            grads, vars_, loss = model.compute_gradients(x_batch_train, y_batch_train, training=True)
            optimizer.apply_gradients(zip(grads, vars_))

            logits, _ = model(x_batch_train, training=False)

            train_mean_loss(loss)
            train_acc_metric(y_batch_train, logits)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_mean_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch)

        val_mean_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

        for step in range(int(np.ceil(val.samples / val.batch_size))):
            x_batch_val, y_batch_val = val.next()

            logits, _ = model(x_batch_val, training=False)
            loss = model.compute_loss(logits=logits, labels=y_batch_val)

            val_mean_loss(loss)
            val_acc_metric(y_batch_val, logits)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_mean_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc_metric.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1,
                              train_mean_loss.result(),
                              train_acc_metric.result(),
                              val_mean_loss.result(),
                              val_acc_metric.result()))

        if max_val_acc is None or float(val_acc_metric.result()) > max_val_acc:
            model.save_weights("models/"+model_name+".hdf5")
            max_val_acc = float(val_acc_metric.result())

        model.save_weights("models/curr_"+model_name+".hdf5")
