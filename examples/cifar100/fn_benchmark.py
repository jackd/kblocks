from time import time

import tensorflow as tf

from simple import cifar100_problem, simple_cnn

# tf.config.optimizer.set_jit(True)

problem = cifar100_problem()
image_spec = problem.features_spec
image = tf.keras.Input(shape=image_spec.shape, dtype=image_spec.dtype)
model = simple_cnn(image, problem.outputs_spec)
loss = problem.loss
metrics = problem.metrics
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss_val = loss(labels, predictions)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    for m in metrics:
        m.update_state(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    for m in metrics:
        m.update_state(labels, predictions)


batch_size = 16
train_ds, val_ds = (problem.get_base_dataset(s) for s in ("train", "validation"))

train_ds = train_ds.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(-1)
val_ds = val_ds.batch(batch_size, drop_remainder=True).prefetch(-1)
EPOCHS = 10

for epoch in tf.range(EPOCHS):
    for m in metrics:
        m.reset_states()

    t = time()
    for images, labels in train_ds:
        train_step(images, labels)

    dt = time() - t
    print("Finished epoch {} in {:.3f} s".format(epoch.numpy(), dt))
    for m in metrics:
        print(m.name, m.result().numpy())

    for m in metrics:
        m.reset_states()

    for test_images, test_labels in val_ds:
        test_step(test_images, test_labels)
    print("Test metrics:")
    for m in metrics:
        print(m.name, m.result().numpy())
