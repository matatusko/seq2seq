import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

#==============================================================================
# Building the model
#==============================================================================
def model_inputs():
    """Create palceholders for inputs to the model"""
    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_length = tf.placeholder(tf.int32, (None,), name='target_length')
    max_target_length = tf.reduce_max(target_length, name='max_dec_len')
    input_length = tf.placeholder(tf.int32, (None,), name='input_length')

    return input_data, targets, lr, keep_prob, \
           target_length, max_target_length, input_length
           
def process_encoding_input(target_data, vocab2int, batch_size):
    """
    Remove the last word id from each batch and concat the <GO> to the 
    begining of each batch
    """
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab2int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    """Create the encoding layer"""
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                    input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                    input_keep_prob=keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)
    
    return enc_output, enc_state

def training_decoding_attention(rnn_size, enc_output, enc_state, input_length, 
                                dec_cell, batch_size):
    """Prepare the training attention and initial state"""
    attn_mech_training = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                              enc_output,
                                                              input_length,
                                                              normalize=False,
                                                              name='BahdanauAttention')
    dec_cell_training = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                            attention_mechanism=attn_mech_training,
                                                            attention_layer_size=rnn_size)
    initial_state_training = dec_cell_training.zero_state(batch_size, tf.float32)
    initial_state_training = initial_state_training.clone(cell_state=enc_state[0])
    
    return dec_cell_training, initial_state_training

def training_decoding_layer(dec_embed_input, target_length, dec_cell, initial_state, 
                            output_layer, vocab_size, max_target_length):
    """Create the training logits"""
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=target_length,
                                                        time_major=False)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer) 
    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_target_length)
    return training_logits[0]

def inference_decoding_attention(enc_output, enc_state, input_length, rnn_size, dec_cell,
                                batch_size, beam_width):
    """Prepare the inferenec attention and initial state"""
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(enc_state[0], multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(input_length, multiplier=beam_width)
    
    attn_mech_sample = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                            memory=tiled_encoder_outputs,
                                                            memory_sequence_length=tiled_sequence_length)
    dec_cell_inference = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                             attention_mechanism=attn_mech_sample,
                                                             attention_layer_size=rnn_size)
    decoder_initial_state_inference = dec_cell_inference.zero_state(dtype=tf.float32, 
                                                                   batch_size=batch_size*beam_width)
    decoder_initial_state_inference = decoder_initial_state_inference.clone(cell_state=tiled_encoder_final_state)
    
    return dec_cell_inference, decoder_initial_state_inference

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, 
                             output_layer, max_target_length, batch_size, beam_width):
    """
    Create the inference logits with beam search, which will be used during
    the sampling stage in order to find a bunch of questions with highest predictions
    """
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), 
                           [batch_size], 
                           name='start_tokens')    
    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=dec_cell,
                                                             embedding=embeddings,
                                                             start_tokens=start_tokens,
                                                             end_token=end_token,
                                                             initial_state=initial_state,
                                                             beam_width=beam_width,
                                                             output_layer=output_layer)
    inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                         output_time_major=False,
                                                         impute_finished=False,
                                                         maximum_iterations=max_target_length)
    
    return inference_logits[0]

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, 
                   input_length, target_length, max_target_length, rnn_size, 
                   vocab2int, keep_prob, batch_size, num_layers, beam_width):
    """
    Create the decoding cell and attention for the training and inference 
    decoding layers. There are different initial state for training and inference
    due to the use of beam_search
    """
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    #==========================================================================
    # # Training decode using standard decoder
    #==========================================================================   
    dec_cell_training, initial_state_training = training_decoding_attention(rnn_size, 
                                                                            enc_output, 
                                                                            enc_state, 
                                                                            input_length, 
                                                                            dec_cell, 
                                                                            batch_size)
    
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  target_length, 
                                                  dec_cell_training, 
                                                  initial_state_training,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_target_length)
    
    #==========================================================================
    # # Inference decoding using beam search
    #==========================================================================
    dec_cell_inference, decoder_init_state_inference = inference_decoding_attention(enc_output, 
                                                                                    enc_state, 
                                                                                    input_length, 
                                                                                    rnn_size, 
                                                                                    dec_cell,
                                                                                    batch_size, 
                                                                                    beam_width)
        
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab2int['<GO>'], 
                                                    vocab2int['<EOS>'],
                                                    dec_cell_inference, 
                                                    decoder_init_state_inference, 
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size,
                                                    beam_width)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, input_length, target_length, 
                  max_target_length, vocab_size, rnn_size, num_layers, vocab2int, 
                  word_embedding_matrix, batch_size, beam_width):
    """Use the previously defined functions to create the training and inference logits"""
    
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, input_length, num_layers, 
                                           enc_embed_input, keep_prob)
    
    dec_input = process_encoding_input(target_data, vocab2int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    training_logits, inference_logits = decoding_layer(dec_embed_input, 
                                                       embeddings,
                                                       enc_output,
                                                       enc_state, 
                                                       vocab_size, 
                                                       input_length, 
                                                       target_length, 
                                                       max_target_length,
                                                       rnn_size, 
                                                       vocab2int, 
                                                       keep_prob, 
                                                       batch_size,
                                                       num_layers,
                                                       beam_width)
    
    return training_logits, inference_logits

def pad_text_batch(data_batch, vocab2int):
    """Pad text with <PAD> so that each text of a batch has the same length"""
    max_text = max([len(text) for text in data_batch])
    return [text + [vocab2int['<PAD>']] * (max_text - len(text)) for text in data_batch]

def get_batches(targets, inputs, vocab2int, batch_size):
    """Batch targets, inputs, and the their lengths of their together"""
    for batch_i in range(0, len(inputs)//batch_size):
        start_i = batch_i * batch_size
        targets_batch = targets[start_i:start_i + batch_size]
        inputs_batch = inputs[start_i:start_i + batch_size]
        pad_targets_batch = np.array(pad_text_batch(targets_batch, vocab2int))
        pad_inputs_batch = np.array(pad_text_batch(inputs_batch, vocab2int))
        
        # Need the lengths for the _lengths parameters
        pad_targets_lenghts = []
        for target in pad_targets_batch:
            pad_targets_lenghts.append(len(target))
        
        pad_inputs_lenghts = []
        for text in pad_inputs_batch:
            pad_inputs_lenghts.append(len(text))
        
        yield pad_targets_batch, pad_inputs_batch, pad_targets_lenghts, pad_inputs_lenghts
        