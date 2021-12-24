import tensorflow as tf
import time
import numpy as np
def train(model,depth,prob,train_data,validation_data,x_test,y_test,epochs=24):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,nesterov=True)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    history = [[],[],[]]
    time_0 = time.time()
    for epoch in range(epochs):
        print("Epoch %d/%d" % (epoch+1,epochs))
        if (epoch==12) | (epoch==18):
            optimizer.learning_rate = optimizer.learning_rate/10
            
        start_time = time.time()
        step = 0
        for x_batch_train, y_batch_train in train_data:
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y_batch_train, logits)
            if step % 200 == 0:
                print(
                    "Training loss at step %d: %.4f"
                    % (step, float(loss_value))
                )
            step += 1

        history[0].append(loss_value.numpy())
        train_acc = train_acc_metric.result()
        train_acc_metric.reset_states()

        step = 0
        for x_batch_val, y_batch_val in validation_data:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
            step += 1
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        history[1].append(train_acc.numpy())
        history[2].append(val_acc.numpy())
        print("Training accuracy: %.4f" % (float(train_acc),)
              ,"Validation accuracy: %.4f" % (float(val_acc),),"Time taken: %.2fs" % (time.time() - start_time))
    time_total = time.time()-time_0
    val_acc_metric.update_state(y_test, model.predict(x_test))
    accurucy =  val_acc_metric.result().numpy()
    log = np.array(history)
    return log, accurucy, time_total