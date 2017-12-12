import pickle
import spacy
import numpy as np
import pandas as pd

# Load up spacy model and import stop words
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import STOP_WORDS
for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True

def save_pickle(data, filename):
    """Saves the data into pickle format"""
    save_documents = open('data/'+filename+'.pickle', 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()
    
def load_pickle(data_filepath):
    """Loads up the pickled dataset for further parsing and preprocessing"""
    documents_f = open('data/'+data_filepath+'.pickle', 'rb')
    data = pickle.load(documents_f)
    documents_f.close()
    
    return data

def preprocess_data(data, remove_stopwords=True, replace_entities=False):
    """
    Preprocesses the data by changing all the entities in text into their
    respective form (PERSON, LOC, GPE etc) as well as removes stopwords and
    punctuations from the text is asked for. For body text stopwords should
    be removed, but questions should stay in the original form as we would 
    want the model to generate proper looking questions.
    
    Inputs:
        data: a list of texts to preprocess
        remove_stopwords: as name suggests, boolean, default True
        replace_entities: change entities into their tags, default False
    Returns:
        parsed_data: a list of input texts, but preprocessed
    """
    parsed_data = []
    for index in range(len(data)):
        # Text being parsed right now
        text = data[index]
        
        if replace_entities:
            # That will take a while for big corpuses
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
            
        # Deal with stopwords and punctuation
        text = nlp(text)
        if remove_stopwords:
            text = [str(token.orth_) for token in text 
                    if not token.is_stop and not token.is_punct]
            text = ' '.join(text)
        else:
            text = [str(token.orth_) for token in text if not token.is_punct]
            text = ' '.join(text)
            
        parsed_data.append(text)
        
        if index % 100 == 0 and index > 0:
            print('Preprocessing {}/{}'.format(index, len(data)))
            
        if index % 1000 == 0 and index > 0:
            print('Pickling progress so far.')
            save_pickle(parsed_data, 'parsed_data')
         
        if index % 2000 == 0:
            try:
                print('Sanity check, currently parsed text is:')
                print(text)
            except:
                pass

    return parsed_data
    
def load_embeddings(embeddings_index, filepath):
    """Load Numberbatch word embeddings"""
    print('Loading Conceptnet Numberbatch word embeddings')
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    
    print('Word embeddings:', len(embeddings_index))
    
def count_word_frequency(word_frequency, data):
    """calculate the world frequency for each token in the corpus"""
    for text in data:
        for token in text.split():
            if token not in word_frequency:
                word_frequency[token] = 1
            else:
                word_frequency[token] += 1
                
def create_conversion_dictionaries(word_frequency, embeddings_index, threshold=10):
    """
    Cleans the dataset by removing the words from a corpus which appear below
    a predefined threshold (default 20).
    
    Input:
        word_frequency: dictionary with word frequencies in the corpus
        embeddings_index: dictionary of words and their corresponding vectors
        threshold: frequency threshold under which words will be discarded
    Returns:
        vocab2int: dictionary to convert vocabulary to integers
        int2vocab: dictionary to reverse conversion, integers and their vocab
    """
    print('Removing token which frequency in the corpus is under specified threshold')
    missing_words = 0
    
    for token, freq in word_frequency.items():
        if freq > threshold:
            if token not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_frequency), 4) * 100
    print('Number of words missing from Conceptnet Numberbatch:', missing_words)
    print('Percent of words that are missing from vocabulary: ', missing_ratio, '%')

    # Dictionary to convert words to integers
    print('Creating vocab_to_int dictionary')
    vocab2int = {}
    
    value = 0
    for token, freq in word_frequency.items():
        if freq >= threshold or token in embeddings_index:
            vocab2int[token] = value
            value += 1
    
    # Special tokens that will be added to our vocab. Those tokens will guide the
    # sequence to sequence model
    codes = ['<UNK>', '<PAD>', '<EOS>', '<GO>']   
    
    print('Adding special tokens to vocab_to_int dictionary.')
    # Add the codes to the vocab list
    for code in codes:
        vocab2int[code] = len(vocab2int)
    
    # Dictionary to convert integers to words
    print('Creating int_to_vocab dictionary.')
    int2vocab = {}
    for token, index in vocab2int.items():
        int2vocab[index] = token
    
    usage_ratio = round(len(vocab2int) / len(word_frequency), 4) * 100
    print("Total number of unique words:", len(word_frequency))
    print("Number of words we will use:", len(vocab2int))
    print("Percent of words we will use: {}%".format(usage_ratio))
    
    return vocab2int, int2vocab

def create_embedding_matrix(vocab2int, embeddings_index, embedding_dimensions=300):
    """
    Creates embedding matrix for each token left in the corpus, as denoted by 
    vocab to int. If the word vector is not available in the embeddings_index
    then we create a random embedding for it.
    
    Input:
        vocab2int: dictionary which contains vocab to integer conversions for corpus
        embeddings_index: word vectors
        embedding_dimensions: dimensions of word vectors. 300 by default to match Conceptnet
    Returns:
        word_embedding_matrix: final word vectors for our corpus
    """
    # Number of words in total in the corpus
    num_words = len(vocab2int)
    
    # Create a default matrix with all values set to zero and fill it out
    print('Creating word embedding matrix with all the tokens and their corresponding vectors.')
    word_embedding_matrix = np.zeros((num_words, embedding_dimensions), dtype=np.float32)
    for token, index in vocab2int.items():
        if token in embeddings_index:
            # If the token is pretrained in CN's vectors, use that
            word_embedding_matrix[index] = embeddings_index[token]
        else:
            # Else, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dimensions))
            word_embedding_matrix[index] = new_embedding
            
    return word_embedding_matrix
    
def convert_data_to_ints(data, vocab2int, word_count, unk_count, eos=True):
    """
    Converts the words in the data into their corresponding integer values.
    
    Input:
        data: a list of texts in the corpus
        vocab2list: conversion dictionaries
        word_count: an integer to count the words in the dataset
        unk_count: an integer to count the <UNK> tokens in the dataset
        eos: boolean whether to append <EOS> token at the end or not (default true)
    Returns:
        converted_data: a list of corpus texts converted to integers
        word_count: updated word count
        unk_count: updated unk_count
    """    
    converted_data = []
    for text in data:
        converted_text = []
        for token in text.split():
            word_count += 1
            if token in vocab2int:
                # Convert each token in the paragraph to int and append it
                converted_text.append(vocab2int[token])
            else:
                # If it's not in the dictionary, use the int for <UNK> token instead
                converted_text.append(vocab2int['<UNK>'])
                unk_count += 1
        if eos:
            # Append <EOS> token if specified
            converted_text.append(vocab2int['<EOS>'])
            
        converted_data.append(converted_text)
    
    assert len(converted_data) == len(data)
    return converted_data, word_count, unk_count

def build_summary(data):
    """
    Build pandas data frame summary for dataset, useful for finding the length
    our sequence should be
    """
    summary = []
    for text in data:
        summary.append(len(text))
    return pd.DataFrame(summary, columns=['counts'])

def unk_counter(data, vocab2int):
    """Count <UNK> tokens in data"""
    unk_count = 0
    for token in data:
        if token == vocab2int['<UNK>']:
            unk_count += 1
    return unk_count

def remove_wrong_length_data(coverted_inputs, converted_targets, vocab2int,
                             start_inputs_length, max_inputs_length, max_targets_length, 
                             min_inputs_length=10, min_targets_lengths=5,
                             unk_inputs_limit=1, unk_targets_limit=0):
    """
    Sort the paragraphs and questions by the length of their texts, shortest to longest
    Limit the length of summaries and texts based on the min and max ranges.
    This step is important especially if some of the texts in the corpus provided
    are very long, compared to others. For long texts, it would take too much
    memory to learn sequence, thus we want to avoid those. Short sequences
    might act as unneccesary noise in the learning process.
    """
    sorted_inputs = []
    sorted_targets = []
    
    print('Doing final preprocessing - sorting the texts and keeping only those ' + \
          'of appropriate length.')
    for length in range(start_inputs_length, max_inputs_length): 
        for index, words in enumerate(converted_targets):
            if (len(converted_targets[index]) >= min_targets_lengths and
                len(converted_targets[index]) <= max_targets_length and
                len(coverted_inputs[index]) >= min_inputs_length and
                unk_counter(converted_targets[index], vocab2int) <= unk_targets_limit and
                unk_counter(coverted_inputs[index], vocab2int) <= unk_inputs_limit and
                length == len(coverted_inputs[index])
               ):
                sorted_targets.append(converted_targets[index])
                sorted_inputs.append(coverted_inputs[index])
        
    # Ensure the lenght os sorten paragraph and questions match
    assert len(sorted_inputs) == len(sorted_targets)
    print('Got {} inputs/targets pairs!'.format(len(sorted_inputs)))
    
    return sorted_inputs, sorted_targets

#==============================================================================
# # Load the dataset
#==============================================================================            
data_inputs = load_pickle('train_squad_paragraphs')
data_targets = load_pickle('train_squad_questions') 
assert len(data_targets) == len(data_inputs)
print('Loaded {} question/answer pairs.'.format(len(data_inputs)))

#==============================================================================
# # Load up parsed dataset if found. Else preprocess it
#==============================================================================
try:
    parsed_inputs = load_pickle('parsed_inputs')
    parsed_targets = load_pickle('parsed_targets')
except:
    print('Preprocessing inputs, this may take a while...')
    parsed_inputs = preprocess_data(data_inputs, remove_stopwords=True,
                                    replace_entities=True)
    save_pickle(parsed_inputs, 'parsed_inputs')
    print('Preprocessing targets, this may take a while...')
    parsed_targets = preprocess_data(data_targets, remove_stopwords=False,
                                     replace_entities=True)
    save_pickle(parsed_targets, 'parsed_targets')
    
    assert len(parsed_inputs) == len(parsed_targets)
    print('Loaded up {} parsed inputs/targets pairs'.format(len(parsed_inputs)))

pickle_parsed_data = 1
if pickle_parsed_data:
    save_pickle(parsed_inputs, 'parsed_inputs')
    save_pickle(parsed_targets, 'parsed_targets')

#==============================================================================
# # Load Numberbatch word embeddings
#==============================================================================
filepath = '../numberbatch-en-17.06.txt'
embeddings_index = {}
load_embeddings(embeddings_index, filepath)

#==============================================================================
# # Calculate word frequency
#==============================================================================
word_frequency = {}
count_word_frequency(word_frequency, parsed_targets)
count_word_frequency(word_frequency, parsed_inputs)

#==============================================================================
# # Get the usable only tokens and their integer conversion
#==============================================================================
vocab2int, int2vocab = create_conversion_dictionaries(word_frequency, embeddings_index)
save_pickle(vocab2int, 'vocab2int')
save_pickle(int2vocab, 'int2vocab')

#==============================================================================
# Create embedding matrix
#==============================================================================
word_embedding_matrix = create_embedding_matrix(vocab2int, embeddings_index)
del embeddings_index
save_pickle(word_embedding_matrix, 'word_embedding_matrix')

#==============================================================================
# Convert words to integers and pickle the data
#==============================================================================
word_count = 0
unk_count = 0

print('Converting text to integers')
converted_inputs, word_count, unk_count = convert_data_to_ints(parsed_inputs, 
                                                               vocab2int,
                                                               word_count, 
                                                               unk_count)
converted_targets, word_count, unk_count = convert_data_to_ints(parsed_targets, 
                                                                vocab2int,
                                                                word_count, 
                                                                unk_count)
assert len(converted_inputs) == len(converted_targets)

unk_percent = round(unk_count/word_count, 4) * 100
print('Total number of words:', word_count)
print('Total number of UNKs:', unk_count)
print('Percent of words that are UNK:', unk_percent)

save_pickle(converted_inputs, 'converted_inputs')
save_pickle(converted_targets, 'converted_targets')

#==============================================================================
# Build summary and sort the data to keep only the appropriate length
#==============================================================================
assert len(converted_inputs) == len(converted_targets)

summary_inputs = build_summary(converted_inputs)
summary_targets = build_summary(converted_targets)

print('Inputs:')
print(summary_inputs.describe())
print('#' * 50)
print('Targets')
print(summary_targets.describe())

sorted_inputs, sorted_targets = remove_wrong_length_data(converted_inputs,
                                                         converted_targets,
                                                         vocab2int,
                                                         start_inputs_length=min(summary_inputs.counts),
                                                         max_inputs_length=int(np.percentile(summary_inputs.counts, 100)),
                                                         max_targets_length=int(np.percentile(summary_targets.counts, 100)),
                                                         min_inputs_length=10,
                                                         min_targets_lengths=5,
                                                         unk_inputs_limit=1,
                                                         unk_targets_limit=0)

print('Pickling the final files.')
save_pickle(sorted_inputs, 'sorted_inputs')
save_pickle(sorted_targets, 'sorted_targets')

print('Preprocessing data finished!')
