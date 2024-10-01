import pickle

import torch
def test(params, generator, model, train_loader_g, val_loader_g):
    path_y_prediction_train = params['path_y_prediction_train']
    path_y_prediction_val = params['path_y_prediction_val']
    log_file_path = params['log_path']

    avg_loss_test_t, y_pred_t, y_true_t, x_pred_t, x_true_t = test_prompt_generator(model, generator, train_loader_g, params)
    avg_loss_test_v, y_pred_v, y_true_v, x_pred_v, x_true_v = test_prompt_generator(model, generator, val_loader_g, params)

    with open(path_y_prediction_train, "wb") as f:
        pickle.dump((y_pred_t, y_true_t, x_pred_t, x_true_t), f)

    with open(path_y_prediction_val, "wb") as f:
        pickle.dump((y_pred_v, y_true_v, x_pred_v, x_true_v), f)

    print("avg_loss_test_train", avg_loss_test_t)
    print("avg_loss_test_val", avg_loss_test_v)

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"avg_loss_test_train: {avg_loss_test_t} \n")
        filehandle.write(f"avg_loss_test_val: {avg_loss_test_v} \n")


def test_all(params, generator, model, dataloder_g):
    path_y_prediction_g = params['path_y_prediction_g']
    log_file_path = params['log_path']

    avg_loss_test_t, y_pred_t, y_true_t, x_pred_t, x_true_t, labels = test_prompt_generator_all(model, generator, dataloder_g, params)

    with open(path_y_prediction_g, "wb") as f:
        pickle.dump((y_pred_t, y_true_t, x_pred_t, x_true_t, labels), f)

    print("avg_loss_test_t", avg_loss_test_t)

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"avg_loss_test_t: {avg_loss_test_t} \n")


def test_prompt_generator_all(model, generator, test_loader, params):
    device = params['device']
    criterion = torch.nn.MSELoss(reduction="sum")

    generator.eval()
    model.eval()
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for x_g, y_shift_g, y_shift_m, y_m, label in test_loader:

            x_g, y_shift_g = x_g.to(device), y_shift_g.to(device)
            y_shift_m, y_m = y_shift_m.to(device), y_m.to(device)

            pred_gen = generator.infer(x_g, y_shift_g.shape[1])
            pred_mod = model.infer_m(pred_gen, y_shift_m.shape[1])

            loss = criterion(pred_mod, y_m)

            test_loss += loss.item()
            if count == 0:
                y_true = y_m.cpu()
                y_pred = pred_mod.detach().cpu()
                x_true = x_g.cpu()
                x_pred = pred_gen.detach().cpu()
                labels = label
            else:
                y_true = torch.cat((y_true, y_m.cpu()))
                y_pred = torch.cat((y_pred, pred_mod.detach().cpu()))
                x_true = torch.cat((x_true, x_g.cpu()))
                x_pred = torch.cat((x_pred, pred_gen.detach().cpu()))
                labels = torch.cat((labels, label))

            count += 1
        avg_loss_test = test_loss/count

    return avg_loss_test, y_pred, y_true, x_pred, x_true, labels

def test_prompt_generator(model, generator, test_loader, params):
    device = params['device']
    criterion = torch.nn.MSELoss(reduction="sum")

    generator.eval()
    model.eval()
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for x_g, y_shift_g, y_shift_m, y_m in test_loader:

            x_g, y_shift_g = x_g.to(device), y_shift_g.to(device)
            y_shift_m, y_m = y_shift_m.to(device), y_m.to(device)

            pred_gen = generator.infer(x_g, y_shift_g.shape[1])
            pred_mod = model.infer_m(pred_gen, y_shift_m.shape[1])

            loss = criterion(pred_mod, y_m)

            test_loss += loss.item()
            if count == 0:
                y_true = y_m.cpu()
                y_pred = pred_mod.detach().cpu()
                x_true = x_g.cpu()
                x_pred = pred_gen.detach().cpu()
            else:
                y_true = torch.cat((y_true, y_m.cpu()))
                y_pred = torch.cat((y_pred, pred_mod.detach().cpu()))
                x_true = torch.cat((x_true, x_g.cpu()))
                x_pred = torch.cat((x_pred, pred_gen.detach().cpu()))

            count += 1
        avg_loss_test = test_loss/count

    return avg_loss_test, y_pred, y_true, x_pred, x_true






