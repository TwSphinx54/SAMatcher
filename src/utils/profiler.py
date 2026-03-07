import torch
import time
from contextlib import contextmanager
from collections import defaultdict


class Profiler:
    """Base class for profilers."""
    def __init__(self, accelerator=None):
        self.accelerator = accelerator

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    @contextmanager
    def profile(self, action_name: str) -> None:
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def summary(self) -> str:
        return ""


class PassThroughProfiler(Profiler):
    """A profiler that does nothing."""
    pass


class InferenceProfiler(Profiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time.
    """

    def __init__(self, accelerator=None):
        super().__init__(accelerator)
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)

    def start(self, action_name: str) -> None:
        if self.accelerator is None or self.accelerator.is_main_process:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.current_actions[action_name] = time.perf_counter()

    def stop(self, action_name: str) -> None:
        if self.accelerator is None or self.accelerator.is_main_process:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if action_name in self.current_actions:
                duration = time.perf_counter() - self.current_actions.pop(action_name)
                self.recorded_durations[action_name].append(duration)

    def summary(self) -> str:
        if self.accelerator is None or self.accelerator.is_main_process:
            output = "\nProfiler Summary:\n"
            output += "----------------------------------------------------------------------\n"
            output += "  Action                         Avg duration (ms)    Total count\n"
            output += "----------------------------------------------------------------------\n"
            for action, durations in self.recorded_durations.items():
                if durations:
                    avg_duration = sum(durations) * 1000.0 / len(durations)
                    total_count = len(durations)
                    output += f"  {action:<30} {avg_duration:>20.4f} {total_count:>15}\n"
            output += "----------------------------------------------------------------------\n"
            if self.accelerator:
                self.accelerator.print(output)
            else:
                print(output)
            return output
        return ""


def build_profiler(name, accelerator=None, output_dir=None):
    if name == 'inference':
        return InferenceProfiler(accelerator=accelerator)
    # Add other profilers here if needed, e.g., PyTorch profiler
    # elif name == 'pytorch':
    #     from torch.profiler import profile, record_function, ProfilerActivity
    #     # Configure PyTorch profiler, output_dir might be useful here
    #     return ... 
    else:
        return PassThroughProfiler(accelerator=accelerator)
