import sys 
sys.path.append("..")
import numpy as np 
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import CallbackList
from utils.train_utils import config_model, validate, load_data, open_config


def get_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu1'))
    model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu2'))
    model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu3'))
    model.add(Dense(5, name='output', kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))
    return model


def main():
    model = get_model()

    download =  False
    if download:
        config = open_config("../config/config_w4a6.yml")["data"]
        print(config['preprocess'])
        
        train_data = load_data("../../datasets/jets/train", 1024, 1, config)
        test_data = load_data("../../datasets/jets/val", 1024, 1, config)
        
        X_train_val, y_train_val = train_data[:]
        X_test, y_test = test_data[:]

        np.save('X_train_val.npy', X_train_val)
        np.save('X_test.npy', X_test)
        np.save('y_train_val.npy', y_train_val)
        np.save('y_test.npy', y_test)
    else:
        X_train_val = np.load('X_train_val.npy')
        X_test = np.ascontiguousarray(np.load('X_test.npy'))
        y_train_val = np.load('y_train_val.npy')
        y_test = np.load('y_test.npy', allow_pickle=True)


    train = True
    if train:
        adam = Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
        model.fit(X_train_val, y_train_val, batch_size=1024,
                epochs=100, validation_split=0.2, shuffle=True)
    else:
        from tensorflow.keras.models import load_model
        model = load_model('model_1/KERAS_check_best_model.h5')

    y_keras = model.predict(X_test)
    print("Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))


if __name__ == '__main__':
    main()

# Accuracy: 0.76385
