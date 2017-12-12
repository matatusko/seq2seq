import pickle
import spacy
import os
from random import randint
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

# Load Spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import STOP_WORDS
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

def load_pickle(filename):
    """Loads up the pickled dataset for further parsing and preprocessing"""
    documents_f = open('data/'+filename+'.pickle', 'rb')
    data = pickle.load(documents_f)
    documents_f.close()
    
    return data
        
def clean_text(text, replace_entities=True):
    """Cleans the text in the same way as in data preprocessing part before training"""
    if replace_entities:
        spacy_text = nlp(text)
        text_ents = [(str(ent), str(ent.label_)) for ent in spacy_text.ents]
        
        text = text.lower()
        # Replace entities
        for ent in text_ents:
            replacee = str(ent[0].lower())
            replacer = str(ent[1])
            try:
                text = text.replace(replacee, replacer)
            except:
                pass
    else:
        text = text.lower()
        
    spacy_text = nlp(text)
    spacy_text = [str(token.orth_) for token in spacy_text 
                  if not token.is_punct and not token.is_stop]
    spacy_text = ' '.join(spacy_text)

    return spacy_text
        
def text_to_seq(input_sequence):
    """Prepare the text for the model"""
    text = clean_text(input_sequence)
    return [vocab2int.get(word, vocab2int['<UNK>']) for word in text.split()]

int2vocab = load_pickle('int2vocab')
vocab2int = load_pickle('vocab2int')
dev_squad_paragraphs = load_pickle('dev_squad_paragraphs')
dev_squad_paragraphs = list(set(dev_squad_paragraphs))

random_example = randint(0, len(dev_squad_paragraphs))
input_sequence = dev_squad_paragraphs[random_example]

# Set hyperparameters (same as training)
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75     
beam_width = 3

text = text_to_seq(input_sequence)
checkpoint_path = 'ckpt/model.ckpt'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    try:
        print('Restoring old model from %s...' % checkpoint_path)
        loader = tf.train.import_meta_graph(checkpoint_path + '.meta')
        loader.restore(sess, checkpoint_path)
    except: 
        raise 'Checkpoint directory not found!'

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    input_length = loaded_graph.get_tensor_by_name('input_length:0')
    target_length = loaded_graph.get_tensor_by_name('target_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      target_length: [25], 
                                      input_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})

# Remove the padding from the tweet
pad = vocab2int["<PAD>"] 
new_logits = []
for i in range(batch_size):
    new_logits.append(answer_logits[i].T)

print('Original Text:', input_sequence.encode('utf-8').strip())

print('\nGenerated Questions:')
for index in range(beam_width):
    print(' -- : {}'.format(" ".join([int2vocab[i] for i in new_logits[1][index] if i != pad and i != -1])))
