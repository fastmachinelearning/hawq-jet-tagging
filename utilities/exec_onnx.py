"""
    Run inference on qonnx model & validate accuracy.
"""
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from qonnx.core.onnx_exec import execute_onnx
from finn.core.modelwrapper import ModelWrapper

parser = argparse.ArgumentParser(description="Load Checkpoints")
parser.add_argument("--eval", action="store_true", help="Evaluate model.")
parser.add_argument(
    "--save-path",
    type=str,
    help="Path to save the quantized model.",
)
parser.add_argument(
    "--scale",
    required=False,
    type=float,
    help="Scale input by value.",
)
parser.add_argument(
    "--samples",
    default=500,
    type=int,
    help="Number of samples to run through (Default: 500).",
)
args = parser.parse_args()


def run_onnx_graph(onnx_model, x_test, y_test, max=500, verbose=True):
    if max > len(y_test):
        raise RuntimeError("Max is greater than number of available samples.")

    onnx_correct = 0
    total = 0
    print_freq = int(np.floor(max / 20))

    pred_list = list()
    true_list = list()

    for idx in range(0, max):
        input_dict = {"global_in": x_test[idx].reshape(1, -1)}
        onnx_trace = execute_onnx(
            onnx_model,
            input_dict,
            return_full_exec_context=True,
            start_node=None,
            end_node=None,
        )
        onnx_pred = onnx_trace["global_out"]
        onnx_pred = np.argmax(onnx_pred, axis=1)[0]

        pred_list.append(onnx_pred)
        y_pred = np.argmax(y_test[idx])
        true_list.append(y_pred)

        if y_pred == onnx_pred:
            onnx_correct += 1
        total += 1
        if verbose and (idx + 1) % 200 == 0:
            print(f"[{idx+1}] Ground Truth: {y_test[idx]}, {y_pred}")
            print(f"[{idx+1}] Accuracy: {accuracy_score(np.array(true_list), np.array(pred_list)):.4f}")

    print(f"Correct: {onnx_correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy_score(np.array(true_list), np.array(pred_list))}")
    return accuracy_score(np.array(true_list), np.array(pred_list))


if __name__ == "__main__":
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    if args.scale:
        x_test = np.round(x_test / args.scale)

    filename = args.save_path 
    accuracy = run_onnx_graph(ModelWrapper(filename), x_test, y_test, max=args.samples)
    print(f"ONNX Model: {accuracy}")
