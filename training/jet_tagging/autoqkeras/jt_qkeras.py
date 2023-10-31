import sys 
sys.path.append("..")
import numpy as np 
from sklearn.metrics import accuracy_score

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu, stochastic_ternary, stochastic_binary, ternary
from qkeras.utils import _add_supported_quantized_objects

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from utils.train_utils import config_model, validate, load_data, open_config


def get_callbacks():
    return [
            ModelCheckpoint(
            '../checkpoints/qkeras/model_best_tr.h5',
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        ),
        ReduceLROnPlateau(patience=10, min_delta=10**-6)
    ]

def get_model():
    model = Sequential()
    model.add(QDense(64, input_shape=(16,), name='fc1',
                    kernel_quantizer=quantized_bits(4,0,alpha=1), bias_quantizer=quantized_bits(4,0,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(4,2), name='relu1'))
    model.add(QDense(32, name='fc2',
                    kernel_quantizer=stochastic_binary(), bias_quantizer=stochastic_binary(),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(4,2), name='relu2'))
    model.add(QDense(32, name='fc3',
                    kernel_quantizer=ternary(), bias_quantizer=ternary(),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(QActivation(activation=quantized_relu(3,1), name='relu3'))
    model.add(QDense(5, name='output',
                    kernel_quantizer=stochastic_binary(), bias_quantizer=stochastic_binary(),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))
    return model


def train(model, X_train_val, y_train_val):
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.fit(X_train_val, y_train_val, batch_size=1024,
              epochs=100, validation_split=0.20, shuffle=True, 
              callbacks=get_callbacks())
    return model


def load_checkpoint(filename):
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(filename, custom_objects=co)
    return model 


def verify_checkpoint(X_test, y_test):
    print('Loading best checkpoint...')
    model = load_checkpoint('../checkpoints/qkeras/model_best_tr.h5')

    y_keras = model.predict(X_test)
    print("Checkpoint Test Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))


def main():
    model = get_model()

    X_train_val = np.load('X_train_val.npy')
    X_test = np.ascontiguousarray(np.load('X_test.npy'))
    y_train_val = np.load('y_train_val.npy')
    y_test = np.load('y_test.npy', allow_pickle=True)

    model = train(model, X_train_val, y_train_val)
    y_keras = model.predict(X_test)
    print("Test Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))

    verify_checkpoint(X_test, y_test)


if __name__ == '__main__':
    main()

# best Accuracy: 0.7087458333333333 (30 epochs)
# best Accuracy: 0.731 (100 epochs)
# best Accuracy: 0.7129375 (100 epochs run #2)
