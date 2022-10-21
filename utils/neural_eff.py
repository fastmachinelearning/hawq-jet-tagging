import logging
import numpy as np


def calc_neural_efficieny(model, data_loader, args):
    neural_eff = []
    for inputs, targets in data_loader:
        x1, x2, x3, x4 = get_activations(model, inputs.float(), args)

        e1 = np.count_nonzero(x1) / (64 * len(inputs))
        e2 = np.count_nonzero(x2) / (32 * len(inputs))
        e3 = np.count_nonzero(x3) / (32 * len(inputs))
        e4 = np.count_nonzero(x4) / (5 * len(inputs))

        neural_eff.append(np.sqrt(np.sqrt(e1 * e2 * e3 * e4)))
        # print(f"[Batch] Neural Efficiency: {neural_eff[-1]}")

    logging.info(f"[Model] Neural Efficiency: {np.mean(np.array(neural_eff))}")
    return np.mean(np.array(neural_eff))


def get_activations(model, data_in, args):
    x, act_scaling_factor = model.quant_input(data_in.float())

    x1 = model.dense_1(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn1(x1)
        x = model.act(x)
    else:
        x = model.act(x1)
    x, act_scaling_factor = model.quant_act_1(x, act_scaling_factor)

    x2 = model.dense_2(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn2(x2)
        x = model.act(x)
    else:
        x = model.act(x2)
    x, act_scaling_factor = model.quant_act_2(x, act_scaling_factor)

    x3 = model.dense_3(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn3(x3)
        x = model.act(x)
    else:
        x = model.act(x3)
    x, act_scaling_factor = model.quant_act_3(x, act_scaling_factor)

    x4 = model.dense_4(x, act_scaling_factor)

    return (
        x1.detach().numpy(),
        x2.detach().numpy(),
        x3.detach().numpy(),
        x4.detach().numpy(),
    )
