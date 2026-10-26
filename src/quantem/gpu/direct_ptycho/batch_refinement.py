"""Independent Nelder-Mead searches sharing batched objective evaluations."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from threading import Event

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def _minimize_batch(
    objective: Callable[[np.ndarray], np.ndarray],
    starts: list[np.ndarray],
    *,
    bounds: list[tuple[float | None, float | None]],
    options: dict[str, float | int],
) -> list[OptimizeResult]:
    """Batch one evaluation per active solver without changing its decisions.

    Worker threads run SciPy's independent search drivers. The calling thread
    alone evaluates the objective, so device arrays and execution contexts stay
    on their owning thread. Each batch waits for one request from every active
    solver; completed solvers leave the batch. Bounds and termination remain
    SciPy's responsibility.
    """
    requests: Queue[tuple[int, np.ndarray | None, Future[float] | None]] = Queue()
    stopped = Event()

    def solve(index: int, start: np.ndarray) -> OptimizeResult:
        def evaluate(point: np.ndarray) -> float:
            if stopped.is_set():
                raise RuntimeError("The batched objective stopped.")
            response: Future[float] = Future()
            requests.put((index, point.copy(), response))
            return response.result()

        try:
            return minimize(
                evaluate, start, method="Nelder-Mead", bounds=bounds,
                options=options,
            )
        finally:
            requests.put((index, None, None))

    with ThreadPoolExecutor(max_workers=len(starts)) as pool:
        tasks = [pool.submit(solve, index, start)
                 for index, start in enumerate(starts)]
        active = len(tasks)
        pending: dict[int, tuple[np.ndarray, Future[float]]] = {}
        try:
            while active:
                while len(pending) < active:
                    index, point, response = requests.get()
                    if response is None:
                        active -= 1
                    else:
                        pending[index] = (point, response)
                if pending:
                    ordered = [pending[index] for index in sorted(pending)]
                    values = objective(np.asarray([point for point, _ in ordered]))
                    for (_, response), value in zip(ordered, values, strict=True):
                        response.set_result(float(value))
                    pending.clear()
        except BaseException as error:
            # Release every waiting solver before the executor joins its threads.
            stopped.set()
            for _, response in pending.values():
                if not response.done():
                    response.set_exception(error)
            while active:
                _, _, response = requests.get()
                if response is None:
                    active -= 1
                elif not response.done():
                    response.set_exception(error)
            raise
        return [task.result() for task in tasks]
