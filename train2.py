import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.constants import troy_pound
from torch.utils.tensorboard import SummaryWriter
import psutil
import subprocess
class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "val_total": [],
            "val_forecast": [],
        }
        self.epoch_times = []
        print("use_cuda", use_cuda)
        print("torch.cuda.is_available()", torch.cuda.is_available())
        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[1]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[1]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")

        train_start = time.time()
        for epoch in range(self.n_epochs):
            # 记录 epoch 开始时的 GPU 和 CPU 使用情况
            gpu_start = get_gpu_usage()
            cpu_start = get_cpu_usage()
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
        #
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
        #
                preds = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
        #
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)
        #
                if y.size(0) != preds.size(0):
                    min_len = min(y.size(0), preds.size(0))
                    y = y[:min_len]
                    preds = preds[:min_len]

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                loss = forecast_loss
        #
                loss.backward()
                self.optimizer.step()
        #
                forecast_b_losses.append(forecast_loss.item())
        #
            forecast_b_losses = np.array(forecast_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss
        #
            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)
        #
            # Evaluate on validation set
            forecast_val_loss, total_val_loss = "NA", "NA"
            if val_loader is not None:
                forecast_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
                self.losses["val_total"].append(total_val_loss)
        #
                if total_val_loss <= self.losses["val_total"][-1]:
                    self.save(f"model.pt")
        #
            if self.log_tensorboard:
                self.write_loss(epoch)
        #
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # 记录 epoch 结束时的 GPU 和 CPU 使用情况
            gpu_end = get_gpu_usage()
            cpu_end = get_cpu_usage()
            # 计算资源使用变化2
            gpu_usage = {
                "GPU Utilization (%)": gpu_end["GPU Utilization (%)"],
                "GPU Memory Used (MB)": gpu_end["GPU Memory Used (MB)"],
            }
            cpu_usage = {
                "CPU Utilization (%)": cpu_end["CPU Utilization (%)"],
                "Memory Used (GB)": cpu_end["Memory Used (GB)"],
            }


            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
        #
                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )
        #
                s += f" [{epoch_time:.1f}s]"
                print(s)
                #if epoch % 3 == 0:
                 #   print("GPU使用:", gpu_usage)
                 #   print("CPU使用:", cpu_usage)
        #
        if val_loader is None:
            self.save(f"model.pt")
        #
        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        # 打印 GPU 使用情况
        # get_gpu_memory()
        # get_cpu_memory()
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """

        self.model.eval()

        forecast_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                # print("preds.shape:", preds.ndim)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)


                if y.size(0) != preds.size(0):
                    min_len = min(y.size(0), preds.size(0))
                    y = y[:min_len]
                    preds = preds[:min_len]

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                # recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                forecast_losses.append(forecast_loss.item())

        forecast_losses = np.array(forecast_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())

        total_loss = forecast_loss

        return forecast_loss, total_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)



def get_gpu_usage():
    """
    获取 GPU 的使用量和使用率。
    Returns:
        dict: GPU 使用信息，包括显存使用量和利用率。
    """
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**2  # 单位: MB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # 单位: MB
        gpu_utilization = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        ).decode('utf-8').strip()
        return {
            "GPU Utilization (%)": int(gpu_utilization),
            "GPU Memory Used (MB)": gpu_memory_used,
            "GPU Memory Total (MB)": gpu_memory_total,
        }
    else:
        return {"GPU": "No GPU available"}

def get_cpu_usage():
    """
    获取 CPU 的使用量和使用率。
    Returns:
        dict: CPU 使用信息，包括内存和 CPU 使用率。
    """
    cpu_utilization = psutil.cpu_percent(interval=1)  # CPU 使用率
    memory_info = psutil.virtual_memory()
    return {
        "CPU Utilization (%)": cpu_utilization,
        "Memory Total (GB)": memory_info.total / 1024**3,
        "Memory Used (GB)": psutil.virtual_memory().used / 1024 ** 3,  # 转换为 GB
    }