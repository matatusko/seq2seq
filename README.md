## seq2seq

Universal sequence-to-sequence model with attention and beam search for inference decoding. Can be used for text summarization, 
neural machine translation, question generation etc. 

In my case I've used it for question generation - I've trained the model on reversed [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
dataset with paragraphs as my input and questions as my targets. The SQuAD parsing script is also included in the depository. 

The data and checkpoints are not included as they simply weight too much, but feel free to experiment around with the model and let me know
your results! :) 

## Requirements

- Python 3.5
- Tensorflow 1.4
- [Conceptnet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
- [spaCy 2.0](https://spacy.io/)
- and of course all the standards like numpy, pandas, pickle etc.

## Workflow

Model uses LSTM cells, Bahdanau attention, Conceptnet Numberbatch for word vector and spaCy for data preprocessing. I believe every function
is nicely commented, so I won't go too much into details here as the code speaks for itself.

For the data preprocessing I'm using spaCy as I've came to rely heavily on it when it comes to any NLP tasks - it's simply brilliant.
In this case, spaCy is used to remove stopwords and punctuation from the dataset as well as exchange any entities in the text into their 
corresponding labels, such as LOC, PERSON, DATE etc, so the model learns dependencies between paragraph and question and does not overfit.

## Sample

coming soon
