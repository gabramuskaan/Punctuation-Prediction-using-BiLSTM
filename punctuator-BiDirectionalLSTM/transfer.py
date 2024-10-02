from __future__ import division
import main
import models
import pickle
import data
import numpy as np

from time import time

import models, data
import pickle
import sys
import os.path
from datetime import datetime

import tensorflow as tf
import numpy as np

MAX_EPOCHS = 50
MINIBATCH_SIZE = 32
CLIPPING_THRESHOLD = 2.0
PATIENCE_EPOCHS = 1
learning_rate = 0.02

def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = models.cost(y_pred, y)       
    gradients = tape.gradient(loss, model.params)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIPPING_THRESHOLD)
    optimizer.apply_gradients(zip(gradients, model.params))
    return loss

vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

model_path = r"C:\Users\Dell\OneDrive\Desktop\MAJOR\majorV3\punctuator2-BiDirectionalRNN\Model_biLSTMv3INTELDNN_h256_lr0.02.pcl"

trained_model = models.load(model_path, x)

rng = np.random
rng.seed(1)

n_hidden = 256

# Initialize a new instance of your Bi-LSTM model
# new_model = models.LSTM(rng=rng, x=x, n_hidden=n_hidden)

# for i in range(len(trained_model["params"])):
#     new_model.params[i] = trained_model["params"][i]

# Set the weights of the new model to the weights loaded from the trained model
#new_model.set_weights(trained_model.get_weights())

net = trained_model[0]
optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=1e-6)

starting_epoch = 0
best_ppl = np.inf
validation_ppl_history = []

#print(f"Total number of trainable parameters: {sum(np.prod([dim for dim in param.get_shape()]) for param in net.params)}")

print("Training...")
for epoch in range(starting_epoch, MAX_EPOCHS):
    t0 = time()
    total_neg_log_likelihood = 0
    total_num_output_samples = 0
    iteration = 0 
    for X, Y in main.get_minibatch(data.TRAIN_FILE, MINIBATCH_SIZE, shuffle=True):
        loss = train_step(net, X, Y)
        total_neg_log_likelihood += loss
        total_num_output_samples += np.prod(Y.shape)
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("PPL: %.4f; Speed: %.2f sps\n" % (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100)))
            sys.stdout.flush()
    print(f"Total number of training labels: {total_num_output_samples}")

    total_neg_log_likelihood = 0
    total_num_output_samples = 0
    for X, Y in main.get_minibatch(data.DEV_FILE, MINIBATCH_SIZE, shuffle=False):
        total_neg_log_likelihood += models.cost(net(X, training=True), Y)
        print(f"Total neg log likelihood in dev iteration {iteration}, epoch {epoch}: {total_neg_log_likelihood}")
        total_num_output_samples += np.prod(Y.shape)
        print(f"Total num output samples in dev iteration {iteration}, epoch {epoch}: {total_num_output_samples}")
    print(f"Total number of validation labels: {total_num_output_samples}")

    ppl = np.exp(total_neg_log_likelihood / total_num_output_samples)
    validation_ppl_history.append(ppl)

    print(f"Validation perplexity is {np.round(ppl, 4)}")

    if ppl <= best_ppl:
        model_file_name = "Transfered_Modelv3IntelDNN_h256_0.02"
        best_ppl = ppl
        file_path = r"C:\Users\Dell\OneDrive\Desktop\MAJOR\majorV3\punctuator2-BiDirectionalRNN\Transfered_Modelv3INTELDNN"
        models.saveOLD(net, model_file_name, learning_rate=learning_rate, validation_ppl_history=validation_ppl_history, best_validation_ppl=best_ppl, epoch=epoch, random_state=rng.get_state())
        #models.save(net, file_path)

    elif best_ppl not in validation_ppl_history[-PATIENCE_EPOCHS:]:
        print("Finished!")
        print(f"Best validation perplexity was {best_ppl}")
        #print(f"Total time: {time() - main.starting_time}")
        break

end_time = datetime.now()
elapsed_time = end_time - main.start_time
print(elapsed_time.seconds)
