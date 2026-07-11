"""SB3 training callbacks for logging and timing."""

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def _get_tb_writer(logger):
    """Get SB3's own SummaryWriter from its logger."""
    from stable_baselines3.common.logger import TensorBoardOutputFormat

    for fmt in logger.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            return fmt.writer
    return None


def _log_scalar_info(writer, info, prefix, step):
    """Write all scalar values from an info dict to TensorBoard."""
    for key, value in info.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


class TimingCallback(BaseCallback):
    """Callback to save environment timing stats at the end of training."""

    def __init__(self, log_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir

    def _on_training_end(self) -> None:
        """Save timing stats when training ends."""
        try:
            env = self.training_env.envs[0]  # type: ignore[attr-defined]
            while hasattr(env, "env"):
                env = env.env

            stats = env.get_timing_stats()
            if stats and stats.n_steps > 0:
                summary = stats.summary()
                print(f"\n{'='*60}")
                print("TRAINING TIMING STATS")
                print(f"{'='*60}")
                print(summary)

                benchmark_path = self.log_dir / "timing_stats.txt"
                with open(benchmark_path, "w") as f:
                    f.write("Training Timing Statistics\n")
                    f.write(f"{'='*40}\n\n")
                    f.write(summary)
                    f.write("\n")
                print(f"\nTiming stats saved to: {benchmark_path}")
                print(f"{'='*60}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not save timing stats: {e}")

    def _on_step(self) -> bool:
        return True


class LoggingEvalCallback(EvalCallback):
    """EvalCallback that also logs per-step eval info metrics to TensorBoard."""

    def __init__(self, *args, eval_on_start: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_step_count = 0
        self._tb_writer = None
        self._eval_on_start = eval_on_start

    def _on_training_start(self):
        """Run an evaluation at the very start (fresh model only)."""
        super()._on_training_start()
        if self._eval_on_start:
            self._on_step()

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        try:
            if self._tb_writer is None:
                self._tb_writer = _get_tb_writer(self.logger)
            if self._tb_writer is None:
                return
            info = locals_.get("info")
            if info:
                _log_scalar_info(self._tb_writer, info, "eval", self._eval_step_count)
            self._eval_step_count += 1
        except Exception as e:
            print(f"Error in LoggingEvalCallback: {e}")


class LoggingTrainCallback(BaseCallback):
    """Logs per-step env info scalars to TensorBoard during training.

    Not added to the callback list by default — enable manually for debugging.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._tb_writer = None

    def _on_step(self) -> bool:
        try:
            if self._tb_writer is None:
                self._tb_writer = _get_tb_writer(self.logger)
            if self._tb_writer is None:
                return True

            infos = self.locals.get("infos")
            if infos:
                for i, info in enumerate(infos):
                    _log_scalar_info(
                        self._tb_writer, info, f"env_{i}", self.num_timesteps
                    )
        except Exception as e:
            print(f"Error in LoggingTrainCallback: {e}")

        return True
