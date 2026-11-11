"""One-deep OGA RT pipeline with one in-flight decision and held final keys."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter

def run_cycle(env, agent, observation, task, schema, previous_action):
    ready = Event()
    future = None
    latency = 0.0
    def decide():
        nonlocal latency
        start = perf_counter()
        try:
            return agent.act(observation, task, schema)
        finally:
            latency = perf_counter() - start
            ready.set()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='oga-rt-decision') as pool:
        def start_decision():
            nonlocal future
            assert future is None, 'A cycle must start exactly one model request'
            future = pool.submit(decide)
        result = env.step(previous_action, pause_before_observe=True,
                          rt_decision_start=start_decision, rt_decision_ready=ready)
        # A terminal state detected before action execution needs no new request.
        action = future.result() if future is not None else previous_action
    return action, latency, result
