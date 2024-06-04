import numpy as np
import torch

from time import time

from utility.plotter import plot_loss_training, plot_loss_training_log
from utility.utils import EarlyStopper


class Trainer:
    def __init__(self, model_g, model_m, params):
        self.generator = model_g
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params["lr_generator"])
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.model = model_m
        self.params = params
        self.device = self.params["device"]

        self.log_file_path = self.params['log_path']

    def train_step(self, x_g, y_shift_g, y_shift_m, y_m):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()

        pred_gen = self.generator(x_g, y_shift_g)
        pred_mod = self.model.infer_m(pred_gen, y_shift_m.shape[1])

        loss = self.criterion(pred_mod, y_m)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.params["clip_grad"])
        self.optimizer.step()

        return loss

    def evaluate(self, val_loader):
        self.generator.eval()
        eval_loss = 0.0
        count = 0
        with torch.no_grad():
            for x_g, y_shift_g, y_shift_m, y_m, _ in val_loader:

                x_g, y_shift_g = x_g.to(self.device), y_shift_g.to(self.device)
                y_shift_m, y_m = y_shift_m.to(self.device), y_m.to(self.device)

                pred_gen = self.generator.infer(x_g, y_shift_g.shape[1])
                pred_mod = self.model.infer_m(pred_gen, y_shift_m.shape[1])

                loss = self.criterion(pred_mod, y_m)
                count += 1
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / count
        return avg_eval_loss

    def test(self, model, generator, test_loader):
        generator.eval()
        model.eval()
        test_loss = 0.0
        count = 0
        with torch.no_grad():
            for x_g, y_shift_g, y_shift_m, y_m, _ in test_loader:

                x_g, y_shift_g = x_g.to(self.device), y_shift_g.to(self.device)
                y_shift_m, y_m = y_shift_m.to(self.device), y_m.to(self.device)

                pred_gen = generator.infer(x_g, y_shift_g.shape[1])
                pred_mod = model.infer_m(pred_gen, y_shift_m.shape[1])

                loss = self.criterion(pred_mod, y_m)

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

    def train(self, train_loader, early_stop=True):
        self.model.eval()
        n_epochs = self.params["n_epochs_g"]

        early_stopped = False
        early_stopper = EarlyStopper(patience=30, min_delta=0.0001)

        loss_values = []
        best_loss = np.Inf

        start_train = time()

        print(f'Training model G. Starting time: {start_train}\n')

        with open(self.log_file_path, 'a') as filehandle:
            filehandle.write(f'Training model G. Starting time: {start_train}\n')

        for epoch in range(n_epochs):
            start_epoch = time()
            self.generator.train()
            epoch_loss = 0.0
            count = 0
            for x_g, y_shift_g, y_shift_m, y_m, _ in train_loader:

                x_g, y_shift_g = x_g.to(self.device), y_shift_g.to(self.device)
                y_shift_m, y_m = y_shift_m.to(self.device), y_m.to(self.device)

                loss = self.train_step(x_g, y_shift_g, y_shift_m, y_m)

                count += 1
                epoch_loss += loss.item()

            loss_values.append(epoch_loss / count)

            print(f"Epoch [{epoch + 1}/{n_epochs}], epoch_loss/len(train_loader): "f"{(epoch_loss / len(train_loader)):.8f} \n")

            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(f"[TRAIN] Epoch [{epoch + 1}/{n_epochs}], epoch_loss/len(train_loader): "f"{(epoch_loss / len(train_loader)):.8f} \n")

            current_loss = loss_values[-1]

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(self.generator.state_dict(), self.params["BEST_PATH_GEN"])

            end_epoch = time()
            print(f'End epoch: {epoch+1}, elapsed time: {end_epoch - start_epoch} \n')
            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(f'End epoch: {epoch+1}, elapsed time: {end_epoch - start_epoch} \n')

            if early_stop and early_stopper.early_stop(current_loss):
                early_stopped = True
                break

        if early_stopped:
            print('Early stopping \n')

            with open(self.log_file_path, 'a') as filehandle:
                filehandle.write(f'Early stopping \n')

        end_train = time()
        print(f'End training phase. Elapsed time: {end_train - start_train}')
        with open(self.log_file_path, 'a') as filehandle:
            filehandle.write(f'End training phase. Elapsed time: {end_train - start_train} \n')

        torch.save(self.generator.state_dict(), self.params["LAST_PATH_GEN"])

        plot_loss_training(loss_values, self.params["plot_path_loss_g"])
        plot_loss_training_log(loss_values, self.params["plot_path_log_loss_g"])
