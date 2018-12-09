"""Declaration: This code has been used from an online tutorial 
https://www.hackerearth.com/blog/machine-learning/jazz-music-using-deep-learning/
and has been modified to suit my dataset.
"""
import sys
import re 
import numpy as np 
import pandas as pd
import music21
from glob import glob
import IPython
from tqdm import tqdm
import pickle
from keras.utils import np_utils
import play

from music21 import converter, instrument, note, chord, stream

songs = glob('Bach/*.mid')
songs = songs[:3]

def get_notes():
    notes = []
    for file in songs:
        # converting .mid file to stream object
        midi = converter.parse(file)
        notes_to_parse = []
        try:
            # Given a single stream, partition into a part for each unique instrument
            parts = instrument.partitionByInstrument(midi)
        except:
            pass
        if parts: # if parts has instrument parts 
            notes_to_parse = parts.parts[1].recurse()
        else:
            notes_to_parse = midi.flat.notes
    
        for element in notes_to_parse: 
            if isinstance(element, note.Note):
                # if element is a note, extract pitch
                notes.append(str(element.pitch))
            elif(isinstance(element, chord.Chord)):
                # if element is a chord, append the normal form of the 
                # chord (a list of integers) to the list of notes. 
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    return notes

def prepare_sequences(notes, n_vocab): 
    sequence_length = 50

    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    
    # reshape the input into a format comatible with LSTM layers 
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input = network_input / float(n_vocab)
    
    # one hot encode the output vectors
    network_output = np_utils.to_categorical(network_output)
    
    return (network_input, network_output)

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
def create_network(network_in, n_vocab): 
    """Create the model architecture"""
    model = Sequential()
    model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

from keras.callbacks import ModelCheckpoint
def train(model, network_input, network_output, epochs): 
    """
    Train the neural network
    """
    # Create checkpoint to save the best model weights.
    filepath = 'weights2.best.music3.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
    
    model.fit(network_input, network_output, epochs=epochs, batch_size=50, callbacks=[checkpoint])

def train_network():
    """
    Get notes
    Generates input and output sequences
    Creates a model 
    Trains the model for the given epochs
    """
    
    epochs = 200
    
    notes = get_notes()
    print('Notes processed')
    
    n_vocab = len(set(notes))
    print('Vocab generated')
    
    network_in, network_out = prepare_sequences(notes, n_vocab)
    print('Input and Output processed')
    
    model = create_network(network_in, n_vocab)
    print('Model created')
    
    print('Training in progress')
    train(model, network_in, network_out, epochs)
    print('Training completed')

    return model

### Train the model 
#train_network()

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))
    
    print('Initiating music generation process.......')
    
    network_input = get_inputSequences(notes, pitchnames, n_vocab)
    model = create_network(network_input, n_vocab)
    print('Loading Model weights.....')
    model.load_weights('weights2.best.music3.hdf5')
    print('Model Loaded')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def get_inputSequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 50
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)
    print ('n_patterns ', n_patterns)
    print ('n_vocab ', n_vocab)
    
    # reshape the input into a format comatible with LSTM layers 
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input = network_input / float(n_vocab)

    return (network_input)

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # Pick a random integer
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    # pick a random sequence from the input as a starting point for the prediction
    pattern = network_input[start]
    print ('pattern.shape', pattern.shape)
    prediction_output = []
    
    print('Generating notes........')
    
    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
    
        prediction = model.predict(prediction_input, verbose=0)
        
        # Predicted output is the argmax(P(h|D))
        index = np.argmax(prediction)
        # Mapping the predicted interger back to the corresponding note
        result = int_to_note[index]
        # Storing the predicted output
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        # Next input to the model
        pattern = pattern[1:len(pattern)]

    print('Notes Generated...')
    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    
    print('Saving Output file as midi....')

    midi_stream.write('midi', fp='test_output6.mid')

#### Generate a new jazz music 
generate()

### Play the Jazz music
play.play_midi('test_output6.mid')
