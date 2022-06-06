from src.experiment import run_experiments
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
# from pympler.tracker import SummaryTracker



# tracker = SummaryTracker()

if __name__ == "__main__":
    run_experiments()
    # tracker.print_diff()