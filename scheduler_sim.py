from __future__ import annotations
import argparse
import heapq
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

@dataclass
class Task:
    id: int
    arrival: int
    exec_time: int
    base_priority: int
    priority: int
    wait: int = 0
    start: int | None = None
    finish: int | None = None

    def __lt__(self, other: "Task") -> bool:
        return (-self.priority, self.arrival, self.id) < (-other.priority, other.arrival, other.id)


@dataclass
class Processor:
    id: int
    speed: float
    busy_until: int = 0


def generate_workload(
    *,
    rng: random.Random,
    num_ticks: int,
    arrival_prob: float,
    exec_time_min: int,
    exec_time_max: int,
    priority_min: int,
    priority_max: int,
) -> list[Task]:
    tasks: list[Task] = []
    task_id = 0
    for t in range(num_ticks):
        if rng.random() < arrival_prob:
            exec_time = rng.randint(exec_time_min, exec_time_max)
            priority = rng.randint(priority_min, priority_max)
            tasks.append(Task(task_id, t, exec_time, priority, priority))
            task_id += 1
    return tasks


def clone_tasks(tasks: Iterable[Task]) -> list[Task]:
    cloned: list[Task] = []
    for t in tasks:
        cloned.append(Task(t.id, t.arrival, t.exec_time, t.base_priority, t.base_priority))
    return cloned


def simulate(
    *,
    tasks: list[Task],
    processor_speeds: list[float],
    num_ticks: int,
    aging_enabled: bool,
    aging_threshold: int,
    alpha_prom: float,
    max_priority: int,
) -> dict[str, Any]:
    arrivals: dict[int, list[Task]] = {}
    for task in tasks:
        arrivals.setdefault(task.arrival, []).append(task)

    processors = [Processor(i, speed) for i, speed in enumerate(processor_speeds)]
    task_queue: list[Task] = []

    def apply_aging() -> None:
        if not aging_enabled or not task_queue:
            return

        for t in task_queue:
            t.wait += 1

        if len(task_queue) < 2 * len(processors):
            return

        avg_wait = statistics.mean(t.wait for t in task_queue)
        promote_after = max(aging_threshold, int(math.ceil(avg_wait * alpha_prom)))

        candidates = [t for t in task_queue if t.priority < max_priority and t.wait > promote_after]
        if not candidates:
            return

        to_promote = max(candidates, key=lambda t: (t.wait, -t.priority, -t.arrival))
        to_promote.priority += 1
        to_promote.wait = 0
        heapq.heapify(task_queue)

    def schedule_at_time(current_time: int) -> None:
        for p in processors:
            if p.busy_until <= current_time and task_queue:
                task = heapq.heappop(task_queue)
                duration = int(math.ceil(task.exec_time / p.speed))
                task.start = current_time
                task.finish = current_time + max(duration, 1)
                p.busy_until = task.finish

    current_time = 0
    for current_time in range(num_ticks):
        for task in arrivals.get(current_time, []):
            heapq.heappush(task_queue, task)
        apply_aging()
        schedule_at_time(current_time)

    safety_limit = num_ticks + 200_000
    current_time = num_ticks
    while task_queue or any(p.busy_until > current_time for p in processors):
        if current_time > safety_limit:
            break
        apply_aging()
        schedule_at_time(current_time)
        current_time += 1

    completed = [t for t in tasks if t.finish is not None and t.start is not None]
    if not completed:
        return {
            "tasks_total": len(tasks),
            "tasks_completed": 0,
            "makespan": 0,
            "avg_flow": 0,
            "avg_wait": 0,
            "p95_wait": 0,
            "max_wait": 0,
            "wait_gt_1000": 0,
            "avg_wait_by_priority": {},
        }

    makespan = max(t.finish for t in completed)
    flow_times = [t.finish - t.arrival for t in completed]
    wait_times = [t.start - t.arrival for t in completed]
    wait_times_sorted = sorted(wait_times)
    p95_idx = max(0, math.ceil(0.95 * len(wait_times_sorted)) - 1)
    p95_wait = wait_times_sorted[p95_idx]

    avg_wait_by_prio: dict[int, float] = {}
    for prio in sorted({t.base_priority for t in completed}):
        wt = [t.start - t.arrival for t in completed if t.base_priority == prio]
        if wt:
            avg_wait_by_prio[prio] = statistics.mean(wt)

    return {
        "tasks_total": len(tasks),
        "tasks_completed": len(completed),
        "makespan": makespan,
        "avg_flow": statistics.mean(flow_times),
        "avg_wait": statistics.mean(wait_times),
        "p95_wait": p95_wait,
        "max_wait": max(wait_times),
        "wait_gt_1000": sum(1 for w in wait_times if w > 1000),
        "avg_wait_by_priority": avg_wait_by_prio,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows to aggregate")

    def mean_stdev(values: list[float]) -> dict[str, float]:
        if len(values) == 1:
            return {"mean": values[0], "stdev": 0.0}
        return {"mean": statistics.mean(values), "stdev": statistics.stdev(values)}

    out: dict[str, Any] = {
        "n": len(rows),
        "makespan": mean_stdev([float(r["makespan"]) for r in rows]),
        "avg_flow": mean_stdev([float(r["avg_flow"]) for r in rows]),
        "avg_wait": mean_stdev([float(r["avg_wait"]) for r in rows]),
        "p95_wait": mean_stdev([float(r["p95_wait"]) for r in rows]),
        "max_wait": mean_stdev([float(r["max_wait"]) for r in rows]),
        "wait_gt_1000": mean_stdev([float(r["wait_gt_1000"]) for r in rows]),
        "tasks_total": mean_stdev([float(r["tasks_total"]) for r in rows]),
    }
    by_prio: dict[int, dict[str, float]] = {}
    priorities = sorted({p for r in rows for p in r.get("avg_wait_by_priority", {}).keys()})
    for prio in priorities:
        values: list[float] = []
        for r in rows:
            v = r.get("avg_wait_by_priority", {}).get(prio)
            if v is not None:
                values.append(float(v))
        if values:
            by_prio[prio] = mean_stdev(values)
    out["avg_wait_by_priority"] = by_prio
    return out


def markdown_summary(*, results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("| Load (arrival prob) | Aging | Avg makespan (ticks) | Avg flow (ticks) | Avg wait (ticks) | P95 wait (ticks) | Max wait (ticks) |")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|")
    for load in results["loads"]:
        arrival_prob = load["arrival_prob"]
        for aging_label, key in [("No", "no_aging"), ("Yes", "aging")]:
            agg = load[key]
            lines.append(
                "| "
                + f"{arrival_prob:.2f}"
                + " | "
                + aging_label
                + " | "
                + f"{agg['makespan']['mean']:.2f} ± {agg['makespan']['stdev']:.2f}"
                + " | "
                + f"{agg['avg_flow']['mean']:.2f} ± {agg['avg_flow']['stdev']:.2f}"
                + " | "
                + f"{agg['avg_wait']['mean']:.2f} ± {agg['avg_wait']['stdev']:.2f}"
                + " | "
                + f"{agg['p95_wait']['mean']:.2f} ± {agg['p95_wait']['stdev']:.2f}"
                + " | "
                + f"{agg['max_wait']['mean']:.2f} ± {agg['max_wait']['stdev']:.2f}"
                + " |"
            )
    return "\n".join(lines) + "\n"


def run_suite(
    *,
    seeds: list[int],
    loads: list[float],
    num_ticks: int,
    num_processors: int,
    speed_min: float,
    speed_max: float,
    exec_time_min: int,
    exec_time_max: int,
    priority_min: int,
    priority_max: int,
    aging_threshold: int,
    alpha_prom: float,
    max_priority: int,
) -> dict[str, Any]:
    suite: dict[str, Any] = {"params": {}, "loads": []}
    suite["params"] = {
        "seeds": seeds,
        "num_ticks": num_ticks,
        "num_processors": num_processors,
        "speed_range": [speed_min, speed_max],
        "exec_time_range": [exec_time_min, exec_time_max],
        "priority_range": [priority_min, priority_max],
        "aging_threshold": aging_threshold,
        "alpha_prom": alpha_prom,
        "max_priority": max_priority,
    }

    for arrival_prob in loads:
        no_aging_rows: list[dict[str, Any]] = []
        aging_rows: list[dict[str, Any]] = []
        for seed in seeds:
            rng = random.Random(seed)
            processor_speeds = [rng.uniform(speed_min, speed_max) for _ in range(num_processors)]
            workload = generate_workload(
                rng=rng,
                num_ticks=num_ticks,
                arrival_prob=arrival_prob,
                exec_time_min=exec_time_min,
                exec_time_max=exec_time_max,
                priority_min=priority_min,
                priority_max=priority_max,
            )

            no_aging_rows.append(
                simulate(
                    tasks=clone_tasks(workload),
                    processor_speeds=processor_speeds,
                    num_ticks=num_ticks,
                    aging_enabled=False,
                    aging_threshold=aging_threshold,
                    alpha_prom=alpha_prom,
                    max_priority=max_priority,
                )
            )
            aging_rows.append(
                simulate(
                    tasks=clone_tasks(workload),
                    processor_speeds=processor_speeds,
                    num_ticks=num_ticks,
                    aging_enabled=True,
                    aging_threshold=aging_threshold,
                    alpha_prom=alpha_prom,
                    max_priority=max_priority,
                )
            )

        suite["loads"].append(
            {
                "arrival_prob": arrival_prob,
                "no_aging": aggregate(no_aging_rows),
                "aging": aggregate(aging_rows),
            }
        )

    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Priority scheduling + aging simulation (project experiments).")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results")
    parser.add_argument("--markdown", action="store_true", help="Print a markdown table summary")
    args = parser.parse_args()

    results = run_suite(
        seeds=list(range(10)),
        loads=[0.40, 0.70, 0.90],
        num_ticks=2500,
        num_processors=4,
        speed_min=0.7,
        speed_max=1.6,
        exec_time_min=5,
        exec_time_max=25,
        priority_min=0,
        priority_max=3,
        aging_threshold=20,
        alpha_prom=1.7,
        max_priority=5,
    )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0

    if args.markdown:
        print(markdown_summary(results=results))
        return 0

    print("Run with --markdown for a table or --json for raw output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
