import time
import pickle
import model
import os
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
print('TensorFlow Version: {}'.format(tf.__version__))

def load_pickle(filepath):
    """Load pickled data"""
    documents_f = open('data/'+filepath+'.pickle', 'rb')
    data = pickle.load(documents_f)
    documents_f.close()
    
    return data

#==============================================================================
# # Load data
#==============================================================================
print('Loading and preparing data for training...')

enc_inputs = load_pickle('sorted_inputs')
dec_targets = load_pickle('sorted_targets')
vocab2int = load_pickle('vocab2int')
int2vocab = load_pickle('int2vocab')
word_embedding_matrix = load_pickle('word_embedding_matrix')
assert len(enc_inputs) == len(dec_targets)
assert len(vocab2int) == len(int2vocab)

#==============================================================================
# # Set the Hyperparameters
#==============================================================================
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
learning_rate = 0.005
keep_probability = 0.8     
beam_width = 20

print('Building graph')
# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    
    # Load the model inputs    
    input_data, targets, lr, keep_prob, target_length, max_target_length, input_length = model.model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = model.seq2seq_model(tf.reverse(input_data, [-1]),
                                                            targets, 
                                                            keep_prob,   
                                                            input_length,
                                                            target_length,
                                                            max_target_length,
                                                            len(vocab2int)+1,
                                                            rnn_size, 
                                                            num_layers, 
                                                            vocab2int,
                                                            word_embedding_matrix,
                                                            batch_size,
                                                            beam_width)
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.predicted_ids, name='predictions')
    
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")

#==============================================================================
# Train the model
#==============================================================================
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every 20 batches
stop_early = 0 
stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(enc_inputs)//batch_size//per_epoch)-1

update_loss = 0 
batch_loss = 0
# Record the update losses for saving improvements in the model
question_update_loss = [] 
checkpoint_dir = 'ckpt' 
checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
restore = 0

print('Initializing session and training')
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() 
    
    # If we want to continue training a previous session
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and restore:
        print('Restoring old model parameters from %s...' % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (targets_batch, inputs_batch, targets_lengths, inputs_lengths) in enumerate(
                model.get_batches(dec_targets, enc_inputs, vocab2int, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: inputs_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_length: targets_lengths,
                 input_length: inputs_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(enc_inputs) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check, 3))
                question_update_loss.append(update_loss)
                
                # If the update loss is at a new minimum, save the model
                if update_loss <= min(question_update_loss):
                    print('New Record! Saving the model.') 
                    stop_early = 0
                    saver.save(sess, checkpoint_path)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
            
        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training.")
            break
        
