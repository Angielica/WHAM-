import pickle
def test(params, trainer, generator, model, train_loader_g, val_loader_g):
    path_y_prediction_train = params['path_y_prediction_train']
    path_y_prediction_val = params['path_y_prediction_val']
    log_file_path = params['log_path']

    avg_loss_test_t, y_pred_t, y_true_t, x_pred_t, x_true_t = trainer.test(model, generator, train_loader_g)
    avg_loss_test_v, y_pred_v, y_true_v, x_pred_v, x_true_v = trainer.test(model, generator, val_loader_g)

    with open(path_y_prediction_train, "wb") as f:
        pickle.dump((y_pred_t, y_true_t, x_pred_t, x_true_t), f)

    with open(path_y_prediction_val, "wb") as f:
        pickle.dump((y_pred_v, y_true_v, x_pred_v, x_true_v), f)

    print("avg_loss_test_train", avg_loss_test_t)
    print("avg_loss_test_val", avg_loss_test_v)

    with open(log_file_path, 'a') as filehandle:
        filehandle.write(f"avg_loss_test_train: {avg_loss_test_t} \n")
        filehandle.write(f"avg_loss_test_val: {avg_loss_test_v} \n")






