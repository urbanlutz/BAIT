from torch.utils.tensorboard import SummaryWriter
import torch

DEFAULT_RUNS_LOCATION = "./runs"

class Monitor:
    writer = None

    def __del__(self):
        if self.writer:
            self.writer.flush()

    def track(self, name: str, value:float, x_axis):
        self.writer.add_scalar(name, value, x_axis)

    def track_params(self, params, metrics, name):
        params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v for k, v in params.items()}
        self.writer.add_hparams(params, metrics, run_name=name)

    def track_hist(self, name:str, tensor:torch.Tensor, x_axis):
        self.writer.add_histogram(name, torch.clone(tensor), x_axis)

    def start(self, experiment:str = None):
        path = f"{DEFAULT_RUNS_LOCATION}/{experiment}"
        if self.writer:
            self.writer.flush()
        self.writer = SummaryWriter(path)

MONITOR = Monitor()