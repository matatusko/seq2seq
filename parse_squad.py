import json
import pickle

def parse_squad(dataset):
    """
    Parses SQUAD database into more readable format. In this case I only care
    about question/answers pairs in order to make a seq2seq model that would
    generate questions out of a paragraph.
    
    Inputs:
        dataset: squad dataset in json format
    Returns:
        squad_json: parsed squad dataset in json format
    """
    total_topics = 0
    total_questions = 0
    squad_json = []
    
    # Iterate through every topic in the dataset
    for topic in dataset:
        total_topics += 1
        # Iterate through every text passage in the topic
        for passage in topic['paragraphs']:     
            # Iterate through every question/answer pairs in the passage
            for qas in passage['qas']:
                total_questions += 1
                text_question_pair = {}
                # Append the title
                text_question_pair['topic'] = topic['title']   
                # Append the text paragraph
                text_question_pair['paragraph'] = passage['context']
                # Append the question
                text_question_pair['question'] = qas['question']
                # Iterate through available answers
                answers = []
                for answer in qas['answers']:
                    answers.append(answer['text'])
                # And append them all together
                text_question_pair['answers'] = answers 
                
                # Append found dictionary to the full parsed dataset array
                squad_json.append(text_question_pair)
    
    print('Found ' + str(total_topics) + ' topics in total.')
    print('Found ' + str(total_questions) + ' questions in total.')
    return squad_json

#==============================================================================
# # PARSE AND SAVE TRAIN DATA
#==============================================================================
squad_train_filepath = 'data/train-v1.1.json'
save_path = 'data/parsed_train_data.json'

# Load squad train dataset
train_json = json.load(open(squad_train_filepath, 'r'))
train_dataset = train_json['data']

parsed_train_squad = parse_squad(train_dataset)
json.dump(parsed_train_squad, open(save_path, 'w'))

#==============================================================================
# # PARSE AND SAVE DEV DATA
#==============================================================================
# Filepath to squad dataset and path where to save the parsed dataset  
squad_dev_filepath = 'data/dev-v1.1.json'
save_path = 'data/parsed_dev_data.json'

# Load squad dev dataset
dev_json = json.load(open(squad_dev_filepath, 'r'))
dev_dataset = dev_json['data']

parsed_dev_squad = parse_squad(dev_dataset)
json.dump(parsed_dev_squad, open(save_path, 'w'))

#======================================================================
# Extract paragraph/questions pairs from Stanford's QUAD database
#======================================================================
train_filepath = 'data/parsed_train_data.json'
dev_filepath = 'data/parsed_dev_data.json'

train_set = json.load(open(train_filepath, 'r'))
dev_set = json.load(open(dev_filepath, 'r'))

train_paragraphs = []
train_questions = []
for section in train_set:
    train_paragraphs.append(section['paragraph'])
    train_questions.append(section['question'])
    
dev_paragraphs = []
dev_questions = []
for section in dev_set:
    dev_paragraphs.append(section['paragraph'])
    dev_questions.append(section['question'])
    
assert len(train_paragraphs) == len(train_questions)
assert len(dev_paragraphs) == len(dev_questions)

#==============================================================================
# # Pickle up the extracted lists of questions/anserws pairs
#==============================================================================
def save_pickle(data, filename):
    """Saves the data into pickle format"""
    save_documents = open('data/'+filename+'.pickle', 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()
    
save_pickle(train_paragraphs, 'train_squad_paragraphs')
save_pickle(train_questions, 'train_squad_questions')
save_pickle(dev_paragraphs, 'dev_squad_paragraphs')
save_pickle(dev_questions, 'dev_squad_questions')