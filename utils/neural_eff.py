import logging
import numpy as np


def calc_neural_efficieny(model, data_loader, args):
    neural_eff_batch = []
    for inputs, _ in data_loader:
        a1, a2, a3, a4 = get_activations(model, inputs.float(), args)

        # layer neural_eff = entropy/num_nodes
        e1 = (np.count_nonzero(a1, axis=0) / len(inputs)).sum() / 64
        e2 = (np.count_nonzero(a2, axis=0) / len(inputs)).sum() / 32
        e3 = (np.count_nonzero(a3, axis=0) / len(inputs)).sum() / 32
        e4 = (np.count_nonzero(a4, axis=0) / len(inputs)).sum() / 5

        neural_eff_batch.append(np.sqrt(np.sqrt(e1 * e2 * e3 * e4)))

    neural_eff = np.mean(np.array(neural_eff_batch))
    logging.info(f"[Model] Neural Efficiency: {neural_eff}")

    return neural_eff


def get_activations(model, data_in, args):
    x, act_scaling_factor = model.quant_input(data_in.float())

    x = model.dense_1(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn1(x)
    a1 = model.act(x)
    x, act_scaling_factor = model.quant_act_1(a1, act_scaling_factor)

    x = model.dense_2(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn2(x)
    a2 = model.act(x)
    x, act_scaling_factor = model.quant_act_2(a2, act_scaling_factor)

    x = model.dense_3(x, act_scaling_factor)
    if args.batch_norm:
        x = model.bn3(x)
    a3 = model.act(x)
    x, act_scaling_factor = model.quant_act_3(a3, act_scaling_factor)

    x = model.dense_4(x, act_scaling_factor)
    a4 = model.act(x)

    return (
        a1.detach().numpy(),
        a2.detach().numpy(),
        a3.detach().numpy(),
        a4.detach().numpy(),
    )
