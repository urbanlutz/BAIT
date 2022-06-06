import torch

class Environment:
    device: torch.device
    verbose: bool
    seed: int
    log = True
    experiment_name: str

    def __init__(self, verbose = False, seed=None):
        self.verbose = verbose
        self.device = device(0)

        if seed:
            self.seed = seed
            torch.manual_seed(seed)

def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

ENV = Environment()