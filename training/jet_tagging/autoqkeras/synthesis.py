import sys 
sys.path.append("..")
import numpy as np 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import hls4ml

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.utils import _add_supported_quantized_objects

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from utils.train_utils import config_model, validate, load_data, open_config



def load_checkpoint(filename):
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(filename, custom_objects=co, compile=False)
    return model 


def main():
    X_test = np.ascontiguousarray(np.load('X_test.npy'))
    y_test = np.load('y_test.npy', allow_pickle=True)

    model = load_checkpoint('../checkpoints/qkeras/model_best_tr.h5')
    y_keras = model.predict(X_test)
    print("Checkpoint Test Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = 'ap_fixed<32,16>'
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True
    
    print("-----------------------------------")
    print("Configuration")
    print(config)
    print("-----------------------------------")
    
    hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                           hls_config=config,
                                                           output_dir='hls4ml_prj_qkeras',
                                                           part='xcu250-figd2104-2L-e')
    
    hls_model.compile()
    y_qkeras = model.predict(X_test)
    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
    print("Accuracy quantized: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))))
    print("Accuracy hls4ml: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

    # profiling 
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='hls4ml_prj_qkeras/qkeras_model.png')
    hls4ml.model.profiling.numerical(model=model, hls_model=hls_model, X=X_test[:1000])

    y_hls = hls_model.predict(np.ascontiguousarray(X_test))
    print("Accuracy hls4ml: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

    for ax in plt.gcf().get_axes():
        ax.figure.savefig(ax.get_title().replace('/', '_')+'.png')

    # tracing 
    _, hls4ml_trace = hls_model.trace(X_test[:1000])
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test[:1000])

    print("Keras layer 'fc1', first sample:")
    print(keras_trace['fc1'][0])
    print("hls4ml layer 'fc1', first sample:")
    print(hls4ml_trace['fc1'][0])

    print((keras_trace['fc1'].reshape(-1)-hls4ml_trace['fc1'].reshape(-1)).sum())
    print((keras_trace['relu1'].reshape(-1)-hls4ml_trace['relu1'].reshape(-1)).sum())
    print((keras_trace['fc2'].reshape(-1)-hls4ml_trace['fc2'].reshape(-1)).sum())
    print((keras_trace['relu2'].reshape(-1)-hls4ml_trace['relu2'].reshape(-1)).sum())
    print((keras_trace['fc3'].reshape(-1)-hls4ml_trace['fc3'].reshape(-1)).sum())
    print((keras_trace['relu3'].reshape(-1)-hls4ml_trace['relu3'].reshape(-1)).sum())
    print((keras_trace['output'].reshape(-1)-hls4ml_trace['output'].reshape(-1)).sum())
    
    print("Accuracy quantized: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))))
    print("Accuracy hls4ml: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
    print("Accuracy trace[output]: {}".format(accuracy_score(np.argmax(y_test[:1000], axis=1), np.argmax(hls4ml_trace['output'], axis=1))))

    layers = ['fc1', 'relu1', 'fc2', 'relu2', 'fc3', 'relu3', 'output', 'softmax']
    for idx, layer in enumerate(layers):
        keras_layer, hls_layer = layer, layer
        try:
            diff = np.average(np.abs(keras_trace[keras_layer] - hls4ml_trace[hls_layer] ))
            print(f'{keras_layer}-{hls_layer}', '\t\t', diff)

            onnx_min, onnx_max = keras_trace[keras_layer].flatten().min(), keras_trace[keras_layer].flatten().max()
            hls_min, hls_max = hls4ml_trace[hls_layer].flatten().min(), hls4ml_trace[hls_layer].flatten().max()
            print(f'hls/keras min: {hls_min}/{onnx_min}')
            print(f'hls/keras max: {hls_max}/{onnx_max}')

            plt.figure(figsize=(7, 5))
            plt.figure()
            plt.scatter(keras_trace[hls_layer].flatten(), keras_trace[keras_layer].flatten())
            min_x = min(keras_trace[keras_layer][0].min(), hls4ml_trace[hls_layer][0].min())
            max_x = min(keras_trace[keras_layer][0].max(), hls4ml_trace[hls_layer][0].max())
            plt.plot([min_x, max_x], [min_x, max_x], c='green')
            plt.title(f'(hls) {hls_layer} -- (keras) {keras_layer}')
            plt.xlabel('hls4ml')
            plt.ylabel('keras')
            plt.savefig(f'../results/{idx}_(hls){hls_layer} -- (keras){keras_layer}.png')
            plt.close()
        except Exception as e:
            print(e)
    
    # hls_model.build(synth=True, vsynth=True)

if __name__ == '__main__':
    main()

# best Accuracy: 0.7087458333333333 (30 epochs)
# best Accuracy: 0.731 (100 epochs)
# best Accuracy: 0.7129375 (100 epochs run #2)
