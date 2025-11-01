import logging
import time
from functools import wraps
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


class VerbosityLevel:
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class PhlagLogger:
    def __init__(self, name: str = "phlag", verbosity: int = VerbosityLevel.NORMAL):
        self.logger = logging.getLogger(name)
        self.verbosity = verbosity
        self._setup_logger()
        self._method_timings = {}

    def _setup_logger(self):
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def set_verbosity(self, level: int):
        self.verbosity = level

    def quiet(self, msg: str, *args, **kwargs):
        if self.verbosity >= VerbosityLevel.QUIET:
            self.logger.info(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        if self.verbosity >= VerbosityLevel.NORMAL:
            self.logger.info(msg, *args, **kwargs)

    def verbose(self, msg: str, *args, **kwargs):
        if self.verbosity >= VerbosityLevel.VERBOSE:
            self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        if self.verbosity >= VerbosityLevel.DEBUG:
            self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def section(self, title: str):
        if self.verbosity >= VerbosityLevel.NORMAL:
            separator = "=" * 60
            self.logger.info(f"\n{separator}")
            self.logger.info(f"  {title}")
            self.logger.info(separator)

    def subsection(self, title: str):
        if self.verbosity >= VerbosityLevel.VERBOSE:
            separator = "-" * 60
            self.logger.info(f"\n{separator}")
            self.logger.info(f"  {title}")
            self.logger.info(separator)

    def progress(self, current: int, total: int, prefix: str = "Progress"):
        if self.verbosity >= VerbosityLevel.NORMAL:
            percentage = (current / total) * 100
            self.logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")

    def array_stats(self, arr, name: str):
        if self.verbosity >= VerbosityLevel.DEBUG:
            if isinstance(arr, (jnp.ndarray, np.ndarray)):
                self.logger.debug(
                    f"{name} - shape: {arr.shape}, "
                    f"min: {float(jnp.nanmin(arr)):.6f}, "
                    f"max: {float(jnp.nanmax(arr)):.6f}, "
                    f"mean: {float(jnp.nanmean(arr)):.6f}, "
                    f"nan_count: {int(jnp.isnan(arr).sum())}"
                )

    def dict_summary(self, d: dict, name: str):
        if self.verbosity >= VerbosityLevel.VERBOSE:
            self.logger.info(f"{name}:")
            for key, value in d.items():
                if isinstance(value, (jnp.ndarray, np.ndarray)):
                    self.logger.info(f"  {key}: array {value.shape}")
                else:
                    self.logger.info(f"  {key}: {value}")

    def params_summary(self, params_dict: dict):
        if self.verbosity >= VerbosityLevel.NORMAL:
            self.logger.info("Parameters:")
            for key, value in params_dict.items():
                self.logger.info(f"  {key}: {value}")

    def timing_summary(self):
        if self.verbosity >= VerbosityLevel.NORMAL and self._method_timings:
            self.section("Timing Summary")
            sorted_timings = sorted(
                self._method_timings.items(),
                key=lambda x: x[1]["total_time"],
                reverse=True,
            )
            for method_name, stats in sorted_timings:
                avg_time = stats["total_time"] / stats["count"]
                self.logger.info(
                    f"{method_name}: "
                    f"{stats['total_time']:.4f}s total, "
                    f"{avg_time:.4f}s avg "
                    f"({stats['count']} calls)"
                )

    def _record_timing(self, method_name: str, elapsed_time: float):
        if method_name not in self._method_timings:
            self._method_timings[method_name] = {"total_time": 0.0, "count": 0}
        self._method_timings[method_name]["total_time"] += elapsed_time
        self._method_timings[method_name]["count"] += 1


def log_method_call(logger: PhlagLogger, log_args: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__

            if log_args and logger.verbosity >= VerbosityLevel.DEBUG:
                args_repr = [repr(a) for a in args[1:]][:3]
                kwargs_repr = [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]]
                signature = ", ".join(args_repr + kwargs_repr)
                if len(args) > 4 or len(kwargs) > 3:
                    signature += ", ..."
                logger.debug(f"→ Calling {method_name}({signature})")
            elif logger.verbosity >= VerbosityLevel.VERBOSE:
                logger.verbose(f"→ Calling {method_name}()")

            result = func(*args, **kwargs)

            if logger.verbosity >= VerbosityLevel.VERBOSE:
                logger.verbose(f"✓ Completed {method_name}()")

            return result

        return wrapper

    return decorator


def log_timing(logger: PhlagLogger, message: Optional[str] = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            display_msg = message or f"Executing {method_name}"

            start_time = time.perf_counter()

            if logger.verbosity >= VerbosityLevel.VERBOSE:
                logger.verbose(f"⏱  {display_msg}...")

            result = func(*args, **kwargs)

            elapsed_time = time.perf_counter() - start_time
            logger._record_timing(method_name, elapsed_time)

            if logger.verbosity >= VerbosityLevel.VERBOSE:
                logger.verbose(f"✓ {display_msg} completed in {elapsed_time:.4f}s")
            elif logger.verbosity >= VerbosityLevel.NORMAL:
                logger.info(f"✓ {display_msg} completed")

            return result

        return wrapper

    return decorator


def log_stage(logger: PhlagLogger, stage_name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger.verbosity >= VerbosityLevel.NORMAL:
                logger.section(stage_name)

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time

            if logger.verbosity >= VerbosityLevel.NORMAL:
                logger.info(f"Stage completed in {elapsed_time:.2f}s")

            return result

        return wrapper

    return decorator


def log_iteration(logger: PhlagLogger, process_name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, iteration: int, *args, **kwargs):
            if logger.verbosity >= VerbosityLevel.VERBOSE:
                logger.subsection(f"{process_name} - Iteration {iteration + 1}")
            elif logger.verbosity >= VerbosityLevel.NORMAL:
                logger.info(f"{process_name} iteration {iteration + 1}")

            result = func(self, iteration, *args, **kwargs)

            return result

        return wrapper

    return decorator
