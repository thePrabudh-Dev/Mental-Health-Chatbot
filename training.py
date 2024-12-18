# import nltk
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import json
# import pickle
# import numpy as np
# # import tensorflow as TF
# import tensorflow as TF
# TF.get_logger().setLevel('ERROR')#change
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # set TF_ENABLE_ONEDNN_OPTS=0 #added
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers import SGD
# import random
# words=[]
# classes = []
# documents = []
# ignore_words = ['?', '!']
# data_file = open('intents.json').read()
# intents = json.loads(data_file)

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         #tokenize each word
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         #add documents in the corpus
#         documents.append((w, intent['tag']))
#         # add to our classes list
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])
# # lemmatize and lower each word and remove duplicates
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))
# # sort classes
# classes = sorted(list(set(classes)))
# # documents = combination between patterns and intents
# print (len(documents), "documents")
# # classes = intents
# print (len(classes), "classes", classes)
# # words = all words, vocabulary
# print (len(words), "unique lemmatized words", words)
# pickle.dump(words,open('texts.pkl','wb'))
# pickle.dump(classes,open('labels.pkl','wb'))
# # create our training data
# training = []
# # create an empty array for our output
# output_empty = [0] * len(classes)
# # training set, bag of words for each sentence
# for doc in documents:
#     # initialize our bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = doc[0]
#     # lemmatize each word - create base word, in attempt to represent related words
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     # create our bag of words array with 1, if word match found in current pattern
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
    
#     # output is a '0' for each tag and '1' for current tag (for each pattern)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
    
#     training.append([bag, output_row])
# # shuffle our features and turn into np.array
# random.shuffle(training)
# training = np.array(training, dtype = object)
# # create train and test lists. X - patterns, Y - intents
# train_x = list(training[:,0])
# train_y = list(training[:,1])
# print("Training data created")
# # Create TF.keras.models - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
# TF.keras.models = Sequential()
# TF.keras.models.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# TF.keras.models.add(Dropout(0.5))
# TF.keras.models.add(Dense(64, activation='relu'))
# TF.keras.models.add(Dropout(0.5))
# TF.keras.models.add(Dense(len(train_y[0]), activation='softmax'))
# # Compile TF.keras.models. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this TF.keras.models
# sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# TF.keras.models.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# #fitting and saving the TF.keras.models 
# hist = TF.keras.models.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# TF.keras.models.save('TF.keras.models.keras', hist)
# print("TF.keras.models created")


import json
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# Load intents.json
with open('intents.json') as file:
    data = json.load(file)

# Preprocessing
patterns = []
responses = []
stop_words = set(stopwords.words('english'))
response_map = {}  # Map index to response text

response_index = 0
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = [word for word in word_tokenize(pattern.lower()) if word not in stop_words]
        patterns.append(' '.join(tokens))
        responses.append(response_index)  # Use response index as the output
    response_map[response_index] = intent['responses']  # Map intent responses
    response_index += 1

# Vectorization and Label Encoding
vectorizer = CountVectorizer().fit(patterns)
X = vectorizer.transform(patterns).toarray()
y = np.array(responses)

# Model Architecture
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(response_map), activation='softmax')  # Outputs match response indices
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save the Model
model.save('chatbot_model.keras')

# Save the vectorizer and response map
with open('vectorizer.pkl', 'wb') as v:
    pickle.dump(vectorizer, v)

with open('response_map.json', 'w') as rm:
    json.dump(response_map, rm)

# Chat Function
# def chat():
#     print("Start chatting! (type 'quit' to stop)")
#     while True:
#         user_input = input('You: ').lower()
#         if user_input == 'quit':
#             break
        
#         # Process input
#         tokens = [word for word in word_tokenize(user_input) if word not in stop_words]
#         input_data = vectorizer.transform([' '.join(tokens)]).toarray()
        
#         # Predict response
#         prediction = model.predict(input_data)
#         response_index = np.argmax(prediction)
#         response = random.choice(response_map[str(response_index)])
        
#         print(f"Bot: {response}")
def chat():
    print("Start chatting! (type 'quit' to stop)")
    while True:
        user_input = input('You: ').lower()
        if user_input == 'quit':
            break
        
        # Process input
        tokens = [word for word in word_tokenize(user_input) if word not in stop_words]
        input_data = vectorizer.transform([' '.join(tokens)]).toarray()
        
        # Predict response
        prediction = model.predict(input_data)
        response_index = np.argmax(prediction)
        
        # Lookup response using integer key
        if response_index in response_map:
            response = random.choice(response_map[response_index])
            print(f"Bot: {response}")
        else:
            print("Bot: Sorry, I didn't understand that.")

# Uncomment below to start the chatbot
# chat()
