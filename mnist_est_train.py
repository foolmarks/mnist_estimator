###########################################
# TensorFlow Estimator example
###########################################
import os
import sys
import tensorflow as tf
import numpy as np
import shutil


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


#############################
# Hyperparameters
##############################
LEARN_RATE = 0.0001
BATCH_SIZE = 50
EPOCHS = 5


# checkpoints, graph & TensorBoard data
# are automatically written to this folder
MODEL_DIR = 'est_data'


def model(inputs, is_training):
  '''
  Build the convolution neural network
  arguments:
    inputs: the input tensor - shape must be [None,28,28,1]
  '''
  net = tf.layers.conv2d(inputs, 16, 3, activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
  net = tf.layers.conv2d(net, 32, 3, activation=tf.nn.relu)
  net = tf.layers.flatten(net)
  if is_training:
    net = tf.layers.dropout(net, rate=0.25, training=True)
  else:
    net = tf.layers.dropout(net, rate=0.0, training=False)
  net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
  logits = tf.layers.dense(net, units=10, activation=None)
  return logits
 

def model_fn(features, labels, mode):
    '''
    Standard template for tf.Estimators model functions
    Defines model behaviour in train, eval and predict phases
    '''
    
    global_step=tf.train.get_global_step()
    images = tf.reshape(features, [-1, 28, 28, 1])
    
    #PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
      logits = model(images, False)
      predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
      predict = {'predicted_logit': predicted_logit
                }
      
      # write out inference graph      
      tf.train.write_graph(tf.get_default_graph().as_graph_def(), MODEL_DIR, 'inference_graph.pb', as_text=False)
      print('Saved inference graph to %s' %MODEL_DIR)

      return tf.estimator.EstimatorSpec(mode=mode, predictions=predict)


    #EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
      logits = model(images, False)
      predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
      ground_truth = tf.argmax(input=labels, axis=1, output_type=tf.int32)
      with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, scope='loss')
        tf.summary.scalar('loss', loss)
      with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=ground_truth, predictions=predicted_logit, name='acc')
        tf.summary.scalar('accuracy', accuracy[1])

      return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={'accuracy/accuracy': accuracy},
        evaluation_hooks=None)
       
  
    # TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
      logits = model(images, True)
      predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
      ground_truth = tf.argmax(input=labels, axis=1, output_type=tf.int32)
      with tf.name_scope('loss'):
         loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, scope='loss')
         tf.summary.scalar('loss', loss)
      with tf.name_scope('accuracy'):
         accuracy = tf.metrics.accuracy(labels=ground_truth, predictions=predicted_logit, name='acc')
         tf.summary.scalar('accuracy', accuracy[1])

      # optimizer 
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE, name='Adam')
      train_op = optimizer.minimize(loss,global_step=global_step)
      
      # Create a hook to print acc, loss & global step every 100 iter.   
      train_hook_list= []
      train_tensors_log = {'accuracy'   : accuracy[1],
                           'loss'       : loss
                           }
      train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=100))

      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        training_hooks=train_hook_list)


def mnist_classifier(_):
  
    mnist_dataset = tf.keras.datasets.mnist.load_data('mnist_data')
    (x_train, y_train), (x_test, y_test) = mnist_dataset

    # cast to 32bit float
    x_train = (x_train).astype(np.float32)    
    x_test = (x_test).astype(np.float32)


    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)


    # take 5000 images from test set to make a dataset for prediction
    x_predict = x_train[55000:]
    y_predict = y_train[55000:]

    # reduce test dataset to 55000 images
    y_test = y_test[:55000]
    x_test = x_test[:55000]
    
    # Create a input function for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_train,
                                                        y=y_train,
                                                        batch_size=BATCH_SIZE,
                                                        num_epochs=1,
                                                        shuffle=True)
    # Create a input function for evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_test,
                                                       y=y_test,
                                                       batch_size=BATCH_SIZE,
                                                       num_epochs=1,
                                                       shuffle=False)

   # Create a input function for prediction
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_predict,
                                                       y=y_predict,
                                                       batch_size=1,
                                                       num_epochs=1,
                                                       shuffle=False)

    # keep only the latest checkpoint
    chkpt_config = tf.estimator.RunConfig(
      save_checkpoints_steps=1,
      keep_checkpoint_max = 1
      )


    # Create a estimator with model_fn
    image_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR, config=chkpt_config)

    # Train the model
    print ('-------------------------------------------------------------')
    print ('TRAINING PHASE')
    print ('-------------------------------------------------------------')
    for i in range(EPOCHS):
      print ('  Epoch: {} / {}'.format(i+1,EPOCHS))
      image_classifier.train(input_fn=train_input_fn)

    print ('-------------------------------------------------------------')
    print ('FINISHED TRAINING')
    print('Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % MODEL_DIR)
    print ('-------------------------------------------------------------')
    
    # evaluate the model
    metrics = image_classifier.evaluate(input_fn=eval_input_fn)


    print ('EVALUATION PHASE:')
    print(metrics)
    print ('-------------------------------------------------------------')


    # make some predictions
    predictions = image_classifier.predict(input_fn=pred_input_fn)
    correct_predictions = 0
    wrong_predictions = 0

    print ('PREDICTION PHASE:')
    for i in range(len(x_predict)):
      next_prediction = next(predictions)
      if (next_prediction['predicted_logit'] == np.argmax(y_predict[i])):
        correct_predictions += 1
      else:
        wrong_predictions += 1
    val_accuracy = correct_predictions/len(x_predict)

    print('Validation dataset size:',len(x_predict), ' Correct Predictions:',correct_predictions, ' Wrong Predictions: ',wrong_predictions, 'Validation Accuracy:',val_accuracy)
    print ('-------------------------------------------------------------')
   


if __name__ == '__main__':
    
    dir_list = [MODEL_DIR] 
    # delete existing checkpoints & tensorboard data
    for dir in dir_list:
      if (os.path.exists(dir)):
        shutil.rmtree(dir)

    tf.app.run(mnist_classifier)


