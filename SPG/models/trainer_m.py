import numpy as np
import torch
from utility.plotter import plot_loss, plot_loss_log

from time import time

class TrainerM:
    def __init__(self, model_m, params):
        self.model = model_m
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr_model"])
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.params = params
        self.device = self.params["device"]

        self.log_file_path = self.params['log_path']

    def train_step(self, x_m, y_shift_m, y_m):
        self.optimizer.zero_grad()

        pred_mod = self.model(x_m, y_shift_m)
        loss = self.criterion(pred_mod, y_m)

        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, val_loader):
        self.model.eval()
        eval_loss = 0.0
        count = 0
        with torch.no_grad():
            for x_m, y_shift_m, y_m in val_loader:

                x_m, y_shift_m, y_m = x_m.to(self.device), y_shift_m.to(self.device), y_m.to(self.device)

                pred_mod = self.model(x_m, y_shift_m)

                loss = self.criterion(pred_mod, y_m)
                count += 1
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / count
        return avg_eval_loss

    def train(self, train_loader, val_loader):
        n_epochs = self.params["n_epochs_m"]
        loss_values = []
        val_loss_values = []
        best_loss = np.Inf

        start_train = time()

        with open(self.log_file_path, 'a') as filehandle:
            filehandle.write(f'Training model M. Starting time: {start_train}')


        for epoch in range(n_epochs):
            start_epoch = time()
            self.model.train()
            epoch_loss = 0.0
            count = 0
            for x_m, y_shift_m, y_m in train_loader:

                x_m, y_shift_m, y_m = x_m.to(self.device), y_shift_m.to(self.device), y_m.to(self.device)

                loss = self.train_step(x_m, y_shift_m, y_m)

                count += 1
                epoch_loss += loss.item()

            loss_values.append(epoch_loss / count)

            tmp_train_print = f"[TRAIN] Epoch [{epoch + 1}/{n_epochs}], epoch_loss/len(train_loader): "f"{(epoch_loss / len(train_loader)):.8f} \n"
            print(tmp_train_print)

            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(tmp_train_print)

            avg_eval_loss = self.evaluate(val_loader)
            val_loss_values.append(avg_eval_loss)

            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                torch.save(self.model.state_dict(), self.params["BEST_PATH_MOD"])

            tmp_val_print = f"[VAL] Epoch [{epoch + 1}/{n_epochs}], Eval Loss on val set: {avg_eval_loss:.8f} \n"

            print(tmp_val_print)

            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(tmp_val_print)

            end_epoch = time()
            print(f'End epoch: {epoch + 1}, elapsed time: {end_epoch - start_epoch}')

            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(f'End epoch: {epoch + 1}, elapsed time: {end_epoch - start_epoch} \n')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - start_train}')

        with open(self.log_file_path, 'a') as filehandle:
            filehandle.write(f'End training phase. Elapsed time: {end_train - start_train} \n')

        torch.save(self.model.state_dict(), self.params["LAST_PATH_MOD"])

        plot_loss(loss_values, val_loss_values, self.params["plot_path_loss_m"])
        plot_loss_log(loss_values, val_loss_values, self.params["plot_path_log_loss_m"])

    def test(self, model, generator, test_loader):
        generator.eval()
        model.eval()
        test_loss = 0.0
        count = 0
        with torch.no_grad():
            for x_m, y_shift_m, y_m in test_loader:

                x_m, y_shift_m, y_m = x_m.to(self.device), y_shift_m.to(self.device), y_m.to(self.device)

                pred_mod = model.infer_m(x_m, y_shift_m.shape[1])

                loss = self.criterion(pred_mod, y_m)

                test_loss += loss.item()
                if count == 0:
                    y_true = y_m.cpu()
                    y_pred = pred_mod.detach().cpu()
                else:
                    y_true = torch.cat((y_true, y_m.cpu()))
                    y_pred = torch.cat((y_pred, pred_mod.detach().cpu()))

                count += 1
            avg_loss_test = test_loss/count

        return avg_loss_test, y_pred, y_true
