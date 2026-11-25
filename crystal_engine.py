
import os

for name in [

    "OMP_NUM_THREADS",

    "OPENBLAS_NUM_THREADS",

    "MKL_NUM_THREADS",

    "NUMEXPR_NUM_THREADS",

    "BLIS_NUM_THREADS"

]:

    os.environ[name] = "1"

import sys
import time
import json
import math
import ctypes
import hashlib
import traceback
import itertools
import multiprocessing as mp

from pathlib import Path
from collections import deque
from ctypes import wintypes
from multiprocessing import shared_memory
from queue import Empty

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])

CFG = json.loads(
    (OUT / "config.json").read_text(
        encoding="utf-8"
    )
)

CTX = mp.get_context("spawn")

SIGMA = 5.670374419e-8

WAITING = -4
RUNNING = -2
COMPLETE = -3

# ============================================================
# 3. WINDOWS HARDWARE DISCOVERY
# ============================================================

K = ctypes.WinDLL(
    "kernel32",
    use_last_error=True
)

class GROUP_AFFINITY(ctypes.Structure):

    _fields_ = [

        ("Mask", ctypes.c_size_t),

        ("Group", wintypes.WORD),

        ("Reserved", wintypes.WORD * 3)

    ]


class PROCESSOR_NUMBER(ctypes.Structure):

    _fields_ = [

        ("Group", wintypes.WORD),

        ("Number", ctypes.c_ubyte),

        ("Reserved", ctypes.c_ubyte)

    ]


K.GetCurrentThread.restype = wintypes.HANDLE

K.GetActiveProcessorGroupCount.restype = (
    wintypes.WORD
)

K.GetActiveProcessorCount.argtypes = [
    wintypes.WORD
]

K.GetActiveProcessorCount.restype = (
    wintypes.DWORD
)

K.SetThreadGroupAffinity.argtypes = [

    wintypes.HANDLE,

    ctypes.POINTER(GROUP_AFFINITY),

    ctypes.POINTER(GROUP_AFFINITY)

]

K.SetThreadGroupAffinity.restype = wintypes.BOOL

K.GetCurrentProcessorNumberEx.argtypes = [

    ctypes.POINTER(PROCESSOR_NUMBER)

]

K.GetCurrentProcessorNumberEx.restype = None

K.GetNumaProcessorNodeEx.argtypes = [

    ctypes.POINTER(PROCESSOR_NUMBER),

    ctypes.POINTER(wintypes.USHORT)

]

K.GetNumaProcessorNodeEx.restype = wintypes.BOOL


def current_cpu():

    p = PROCESSOR_NUMBER()

    K.GetCurrentProcessorNumberEx(
        ctypes.byref(p)
    )

    return int(p.Group), int(p.Number)


def bind_cpu(info):

    affinity = GROUP_AFFINITY()

    affinity.Group = info["group"]

    affinity.Mask = 1 << info["cpu"]

    previous = GROUP_AFFINITY()

    ok = K.SetThreadGroupAffinity(

        K.GetCurrentThread(),

        ctypes.byref(affinity),

        ctypes.byref(previous)

    )

    if not ok:

        raise ctypes.WinError(
            ctypes.get_last_error()
        )

    if current_cpu() != (

        info["group"],
        info["cpu"]

    ):

        raise RuntimeError(
            "CPU affinity mismatch"
        )


def topology():

    groups = {}

    for group in range(
        K.GetActiveProcessorGroupCount()
    ):

        groups[group] = []

        for cpu in range(
            K.GetActiveProcessorCount(group)
        ):

            p = PROCESSOR_NUMBER(
                group, cpu, 0
            )

            node = wintypes.USHORT()

            ok = K.GetNumaProcessorNodeEx(

                ctypes.byref(p),

                ctypes.byref(node)

            )

            groups[group].append({

                "group": group,

                "cpu": cpu,

                "numa": (
                    int(node.value)
                    if ok else -1
                )

            })

    result = []

    for i in range(
        max(map(len, groups.values()))
    ):

        for group in sorted(groups):

            if i < len(groups[group]):

                result.append(
                    groups[group][i]
                )

    return result

# ============================================================
# 4. IDENTICAL THERMAL COMPUTATIONAL MODEL
# ============================================================

def make_inputs():

    rng = np.random.default_rng(
        CFG["seed"]
    )

    shape = (

        max(CFG["job_rows"]),

        CFG["width"]

    )

    T0 = np.full(
        shape, 22.0,
        dtype=np.float64
    )

    cap = rng.uniform(
        80, 120, shape
    )

    ranges = [

        (0.3, 0.9),

        (0.1, 0.5),

        (0.2, 0.6),

        (0.1, 0.4),

        (0.1, 0.3),

        (0.05, 0.2),

        (0.7, 0.95),

        (0.5, 1.5)

    ]

    C = np.stack([

        rng.uniform(a, b, shape)

        for a, b in ranges

    ])

    t = np.arange(CFG["steps"])

    outside = (
        -10 + 8*np.sin(2*np.pi*t/150)
    )

    ground = np.full(
        CFG["steps"], 8.0
    )

    wind = (
        1 + 0.5*np.sin(2*np.pi*t/73)
    )

    target = np.full(
        CFG["steps"], 22.0
    )

    outside[CFG["steps"]//3:] -= 5

    wind[CFG["steps"]//2:] += 1

    target[3*CFG["steps"]//4:] += 2

    external = np.stack(

        [outside, ground, wind, target],

        axis=1

    )

    return T0, cap, C, external


def solve(T, cap, C, external):

    T = T.copy()

    cap = cap.copy()

    C = C.copy()

    common = C[0]+C[1]+C[2]+C[4]

    for outside, ground, wind, target in external:

        F = -(common+C[5]*wind)*(T-outside)

        F -= C[3]*(T-ground)

        TK = T+273.15

        OK = outside+273.15

        T2 = TK*TK

        O2 = OK*OK

        F -= (

            0.1*C[6]*SIGMA

            *(T2*T2-O2*O2)

        )

        F += (

            C[7]

            *np.maximum(target-T, 0)

            *10

        )

        T += CFG["dt"]*F/cap

    return T

# ============================================================
# 5. REPRODUCIBLE EXTERNAL TASK STREAM
#
# Identical arrival pattern in every mode.
# The arrival clock belongs to the external
# environment, not the crystal protocol.
# ============================================================

def make_jobs():

    rng = np.random.default_rng(
        CFG["seed"]+330
    )

    jobs = []

    t = 0.0

    offset = 0

    tile_offset = 0

    q = CFG["compute_quantum"]

    while t < CFG["duration_seconds"]:

        rows = int(rng.choice(

            CFG["job_rows"],

            p=CFG["job_probabilities"]

        ))

        tiles = (

            rows+q-1

        )//q

        jobs.append({

            "job": len(jobs),

            "arrival": float(t),

            "rows": rows,

            "output_offset": offset,

            "tile_start": tile_offset,

            "tile_end": tile_offset+tiles,

            "tiles": tiles,

            "input_offset": int(rng.integers(

                0,

                max(CFG["job_rows"])-rows+1

            ))

        })

        offset += rows

        tile_offset += tiles

        if rng.random() < CFG["burst_probability"]:

            interval = rng.uniform(

                CFG["burst_interval_min"],

                CFG["burst_interval_max"]

            )

        else:

            # Correct the non-burst interval
            # so the overall mean is approximately
            # the configured arrival interval.

            burst_mean = (

                CFG["burst_interval_min"]

                + CFG["burst_interval_max"]

            )/2

            normal_mean = (

                CFG["mean_arrival_seconds"]

                - CFG["burst_probability"]*burst_mean

            )/(1-CFG["burst_probability"])

            interval = rng.exponential(
                normal_mean
            )

        t += float(interval)

    return jobs

# ============================================================
# 6. SHARED NUMERICAL MEMORY
# ============================================================

class SharedBlock:

    def __init__(
        self, shape, initial=None
    ):

        self.shape = tuple(shape)

        self.shm = shared_memory.SharedMemory(

            create=True,

            size=int(np.prod(shape))*8

        )

        self.array = np.ndarray(

            self.shape,

            dtype=np.float64,

            buffer=self.shm.buf

        )

        if initial is None:

            self.array.fill(np.nan)

        else:

            self.array[:] = initial

    def descriptor(self):

        return {

            "name": self.shm.name,

            "shape": self.shape

        }

    def close(self):

        self.array = None

        self.shm.close()

        self.shm.unlink()


def attach(desc):

    shm = shared_memory.SharedMemory(

        name=desc["name"]

    )

    array = np.ndarray(

        tuple(desc["shape"]),

        dtype=np.float64,

        buffer=shm.buf

    )

    return shm, array

# ============================================================
# 7. PERMANENT AUTONOMOUS CRYSTAL
#
# State:
# IDLE=1 means the crystal is available.
#
# No periodic work search.
# No central work assignment.
# No artificial limit on collective size.
# Personal wakeup event for each worker.
# ============================================================

def crystal_worker(

    wid,
    cpu_info,
    cpus,
    jobs,
    memory,
    tile_info,
    external,
    raw,
    lock,
    wakeups,
    commands,
    results,
    ready

):

    handles = []

    arrays = {}

    try:

        bind_cpu(cpu_info)

        count = len(cpus)

        nj = len(jobs)

        nt = len(tile_info)

        for name, desc in memory.items():

            shm, a = attach(desc)

            handles.append(shm)

            arrays[name] = a

        del a

        owner = np.frombuffer(

            raw["owner"],

            dtype=np.int32

        )

        released = np.frombuffer(

            raw["released"],

            dtype=np.int32

        )

        remaining = np.frombuffer(

            raw["remaining"],

            dtype=np.int32

        )

        idle = np.frombuffer(

            raw["idle"],

            dtype=np.int32

        )

        generations = np.frombuffer(

            raw["generation"],

            dtype=np.int32

        )

        assignments = np.frombuffer(

            raw["assignment"],

            dtype=np.int32

        )

        sequence = np.frombuffer(

            raw["sequence"],

            dtype=np.int64

        )

        rates = np.frombuffer(

            raw["rates"],

            dtype=np.float64

        )

        pressure_time = np.frombuffer(

            raw["pressure"],

            dtype=np.float64

        )

        job_start = np.frombuffer(

            raw["job_start"],

            dtype=np.float64

        )

        job_end = np.frombuffer(

            raw["job_end"],

            dtype=np.float64

        )

        job_release = np.frombuffer(

            raw["job_release"],

            dtype=np.float64

        )

        assigned_at = np.frombuffer(

            raw["assigned_at"],

            dtype=np.float64

        )

        assigned_by = np.frombuffer(

            raw["assigned_by"],

            dtype=np.int32

        )

        stop = raw["stop"]

        alpha = CFG["speed_alpha"]

        domains = [

            (

                "NUMA", int(c["numa"])

            )

            if c["numa"] >= 0

            else (

                "GROUP", int(c["group"])

            )

            for c in cpus

        ]

        members = {}

        for i, domain in enumerate(domains):

            members.setdefault(
                domain, []
            ).append(i)

        ready.put({

            "wid": wid,

            "pid": os.getpid(),

            "cpu": cpu_info,

            "affinity_ok": True

        })

        # Permanent process loop.
        # Each experimental mode receives
        # the same clean initial state.
        #
        # Experience persists between tasks,
        # not between independent benchmarks.

        while True:

            cmd = commands.get()

            if cmd["type"] == "STOP":

                break

            mode = cmd["mode"]

            repeat = cmd["repeat"]

            origin = cmd["origin"]

            local_rate = 0.0

            handoff_cost = 0.0

            wakeup_cost = 0.0

            current_job = -1

            local_queue = deque()

            seen = {}

            feedback = []

            events = []

            intervals = []

            cpu_start = time.process_time()

            journal_path = (

                OUT / "LIVE_JOURNALS"

                / (

                    f"{mode}_R{repeat}"

                    f"_C{wid:03d}.jsonl"

                )

            )

            journal = open(

                journal_path,

                "a",

                encoding="utf-8",

                buffering=1

            )

            idle_seconds = 0.0

            idle_cpu_seconds = 0.0

            counters = {

                "transfers": 0,

                "prediction_rejects": 0,

                "overload_rejects": 0,

                "overload_pulses": 0,

                "tasks_claimed": 0,

                "feedback_observations": 0,

                "feedback_censored": 0

            }

            def emit(kind, **values):

                events.append({

                    "time": (
                        time.perf_counter()-origin
                    ),

                    "wid": wid,

                    "mode": mode,

                    "repeat": repeat,

                    "event": kind,

                    **values

                })

                journal.write(

                    json.dumps(

                        events[-1],

                        ensure_ascii=False,

                        default=str

                    ) + "\n"

                )


            def pressure_locked(domain, now):

                return sum(

                    math.exp(

                        -max(

                            0.0,

                            now-pressure_time[i]

                        )/CFG["overload_tau"]

                    )

                    for i in members[domain]

                    if pressure_time[i] > 0

                )

            # =================================================
            # First wave: readiness
            # =================================================

            def declare_idle_locked():

                if idle[wid] == 1:

                    return

                idle[wid] = 1

                generations[wid] += 1

                assignments[wid] = -1

                rates[wid] = local_rate

                local_queue.clear()

                seen.clear()

                emit(

                    "IDLE",

                    generation=int(
                        generations[wid]
                    ),

                    remembered_speed=local_rate

                )

            def collect_signals_locked():

                for peer in range(count):

                    if peer == wid:

                        continue

                    if idle[peer] != 1:

                        continue

                    gen = int(
                        generations[peer]
                    )

                    if seen.get(peer) == gen:

                        continue

                    seen[peer] = gen

                    local_queue.append(
                        (peer, gen)
                    )

            # =================================================
            # Initial task absorption
            #
            # A free crystal claims an external task
            # atomically. The source does not choose
            # which crystal receives it.
            # =================================================

            def claim_external_locked():

                available = np.flatnonzero(

                    released == 1

                )

                if not len(available):

                    return -1

                j = int(available[0])

                job = jobs[j]

                a = job["tile_start"]

                b = job["tile_end"]

                if not np.all(

                    owner[a:b] == WAITING

                ):

                    raise RuntimeError(
                        "External task ownership error"
                    )

                released[j] = 2

                owner[a:b] = wid

                idle[wid] = 0

                assignments[wid] = j

                assigned_at[wid] = (
                    time.perf_counter()
                )

                assigned_by[wid] = -1

                counters["tasks_claimed"] += 1

                emit(

                    "TASK_CLAIM",

                    job=j,

                    rows=job["rows"]

                )

                return j

            # =================================================
            # Second wave: half-work donation
            #
            # Personal wakeup event.
            # Atomic receiver reservation.
            # =================================================

            def share_work(j):

                nonlocal handoff_cost

                started = time.perf_counter()

                with lock:

                    job = jobs[j]

                    a = job["tile_start"]

                    b = job["tile_end"]

                    pending = (

                        np.flatnonzero(

                            owner[a:b] == wid

                        )+a

                    )

                    W = len(pending)

                    if W < 2:

                        return

                    if not np.any(idle == 1):

                        return

                    collect_signals_locked()

                    checks = len(local_queue)

                    while checks > 0:

                        checks -= 1

                        peer, gen = (
                            local_queue.popleft()
                        )

                        if (

                            idle[peer] != 1

                            or generations[peer] != gen

                        ):

                            continue

                        # Third wave: resource admission.

                        if mode == "WAVE_OVERLOAD":

                            p = pressure_locked(

                                domains[peer],

                                time.perf_counter()

                            )

                            if p >= CFG["overload_threshold"]:

                                counters[
                                    "overload_rejects"
                                ] += 1

                                local_queue.append(
                                    (peer, gen)
                                )

                                emit(

                                    "OVERLOAD_BLOCK",

                                    job=j,

                                    peer=peer,

                                    pressure=p,

                                    domain=str(
                                        domains[peer]
                                    )

                                )

                                continue

                        n = W//2

                        donated = pending[-n:]

                        kept = pending[:-n]

                        donated_rows = sum(

                            tile_info[i][2]

                            for i in donated

                        )

                        kept_rows = sum(

                            tile_info[i][2]

                            for i in kept

                        )

                        total_rows = (
                            donated_rows+kept_rows
                        )

                        peer_rate = float(
                            rates[peer]
                        )

                        if peer_rate <= 0:

                            peer_rate = local_rate

                        alone = None

                        together = None

                        # Prediction has optimistic
                        # cold start, no compulsory wait.

                        if (

                            mode == "WAVE_PREDICT"

                            and local_rate > 0

                            and peer_rate > 0

                        ):

                            alone = (
                                total_rows/local_rate
                            )

                            cost = (

                                handoff_cost

                                + wakeup_cost

                                + time.perf_counter()

                                - started

                            )

                            together = (

                                cost+max(

                                    kept_rows/local_rate,

                                    donated_rows/peer_rate

                                )

                            )

                            if together >= alone:

                                counters[
                                    "prediction_rejects"
                                ] += 1

                                local_queue.append(
                                    (peer, gen)
                                )

                                emit(

                                    "PREDICTION_REJECT",

                                    job=j,

                                    peer=peer,

                                    alone=alone,

                                    together=together

                                )

                                continue

                        # Atomically reserve receiver.

                        idle[peer] = 0

                        assignments[peer] = j

                        assigned_at[peer] = (
                            time.perf_counter()
                        )

                        assigned_by[peer] = wid

                        owner[donated] = peer

                        if (

                            mode == "WAVE_OVERLOAD"

                            and local_rate > 0

                        ):

                            feedback.append({

                                "job": j,

                                "peer": peer,

                                "generation": gen,

                                "peer_sequence": int(
                                    sequence[peer]
                                ),

                                "self_sequence": int(
                                    sequence[wid]
                                ),

                                "before": local_rate,

                                "kept_rows": kept_rows,

                                "donated_rows": donated_rows,

                                "cost": (
                                    time.perf_counter()-started
                                )

                            })

                        counters["transfers"] += 1

                        # Only the selected receiver wakes.

                        wakeups[peer].set()

                        emit(

                            "TRANSFER",

                            job=j,

                            peer=peer,

                            generation=gen,

                            pending_tiles=W,

                            donated_tiles=n,

                            donated_rows=donated_rows,

                            retained_rows=kept_rows,

                            predicted_alone=alone,

                            predicted_together=together

                        )

                        elapsed = (
                            time.perf_counter()-started
                        )

                        handoff_cost = (

                            elapsed

                            if handoff_cost <= 0

                            else (

                                (1-alpha)*handoff_cost

                                +alpha*elapsed

                            )

                        )

                        break

            # =================================================
            # Third wave: local overload observations
            #
            # No periodic maintenance.
            # One bounded contribution per crystal.
            # =================================================

            def evaluate_feedback():

                if mode != "WAVE_OVERLOAD":

                    return

                if not feedback:

                    return

                resolved = []

                with lock:

                    for item in feedback:

                        peer = item["peer"]

                        if (

                            generations[peer]

                            != item["generation"]

                            or assignments[peer]

                            != item["job"]

                            or assignments[wid]

                            != item["job"]

                        ):

                            counters[
                                "feedback_censored"
                            ] += 1

                            resolved.append(item)

                            continue

                        if (

                            sequence[peer]

                            <= item["peer_sequence"]

                            or sequence[wid]

                            <= item["self_sequence"]

                        ):

                            continue

                        peer_rate = float(
                            rates[peer]
                        )

                        if (

                            peer_rate <= 0

                            or local_rate <= 0

                        ):

                            continue

                        slowdown = (

                            1-local_rate/item["before"]

                        )

                        estimated_alone = (

                            (

                                item["kept_rows"]

                                + item["donated_rows"]

                            )/item["before"]

                        )

                        estimated_together = (

                            item["cost"]

                            +max(

                                item["kept_rows"]/local_rate,

                                item["donated_rows"]/peer_rate

                            )

                        )

                        negative = (

                            estimated_together

                            >= estimated_alone

                        )

                        counters[
                            "feedback_observations"
                        ] += 1

                        if (

                            negative

                            or slowdown

                            > CFG["slowdown_threshold"]

                        ):

                            pressure_time[wid] = (
                                time.perf_counter()
                            )

                            counters[
                                "overload_pulses"
                            ] += 1

                            emit(

                                "OVERLOAD_PULSE",

                                job=item["job"],

                                peer=peer,

                                slowdown=slowdown,

                                negative=bool(negative),

                                domain=str(
                                    domains[wid]
                                ),

                                pressure=pressure_locked(

                                    domains[wid],

                                    time.perf_counter()

                                )

                            )

                        resolved.append(item)

                for item in resolved:

                    feedback.remove(item)

            # =================================================
            # Autonomous crystal lifecycle
            # =================================================

            while True:

                with lock:

                    if stop.value:

                        break

                    assigned = int(
                        assignments[wid]
                    )

                    if assigned >= 0:

                        if assigned != current_job:

                            current_job = assigned

                            if assigned_by[wid] >= 0:

                                latency = max(

                                    0.0,

                                    time.perf_counter()

                                    -assigned_at[wid]

                                )

                                wakeup_cost = (

                                    latency

                                    if wakeup_cost <= 0

                                    else (

                                        (1-alpha)*wakeup_cost

                                        +alpha*latency

                                    )

                                )

                                emit(

                                    "ABSORB",

                                    job=current_job,

                                    donor=int(
                                        assigned_by[wid]
                                    ),

                                    wakeup_latency=latency

                                )

                    else:

                        current_job = (
                            claim_external_locked()
                        )

                    if current_job < 0:

                        declare_idle_locked()

                        # Clear while holding lock:
                        # prevents lost wakeup races.

                        wakeups[wid].clear()

                        sleep_required = True

                    else:

                        sleep_required = False

                if sleep_required:

                    t0 = time.perf_counter()

                    c0 = time.process_time()

                    notified = wakeups[wid].wait()

                    idle_seconds += (
                        time.perf_counter()-t0
                    )

                    idle_cpu_seconds += (
                        time.process_time()-c0
                    )

                    if not notified:

                        raise TimeoutError(
                            f"Crystal {wid} idle timeout"
                        )

                    continue

                j = current_job

                evaluate_feedback()

                share_work(j)

                with lock:

                    job = jobs[j]

                    a = job["tile_start"]

                    b = job["tile_end"]

                    pending = (

                        np.flatnonzero(

                            owner[a:b] == wid

                        )+a

                    )

                    if not len(pending):

                        assignments[wid] = -1

                        current_job = -1

                        counters[
                            "feedback_censored"
                        ] += len(feedback)

                        feedback.clear()

                        continue

                    tile = int(pending[0])

                    owner[tile] = RUNNING

                # =============================================
                # Useful computation
                # =============================================

                job_id, row, nrows = tile_info[tile]

                if job_id != j:

                    raise RuntimeError(
                        "Invalid job assignment"
                    )

                input_a = (

                    jobs[j]["input_offset"]+row

                )

                input_b = input_a+nrows

                output_a = (

                    jobs[j]["output_offset"]+row

                )

                output_b = output_a+nrows

                t0 = time.perf_counter()

                block = solve(

                    arrays["T0"][input_a:input_b],

                    arrays["cap"][input_a:input_b],

                    arrays["C"][:, input_a:input_b],

                    external

                )

                arrays["output"][

                    output_a:output_b

                ] = block

                t1 = time.perf_counter()

                duration = max(
                    t1-t0, 1e-12
                )

                instantaneous_rate = (

                    nrows/duration

                )

                local_rate = (

                    instantaneous_rate

                    if local_rate <= 0

                    else (

                        (1-alpha)*local_rate

                        +alpha*instantaneous_rate

                    )

                )

                intervals.append({

                    "mode": mode,

                    "repeat": repeat,

                    "wid": wid,

                    "job": j,

                    "tile": tile,

                    "rows": nrows,

                    "start": t0-origin,

                    "end": t1-origin,

                    "speed": instantaneous_rate

                })

                with lock:

                    if owner[tile] != RUNNING:

                        raise RuntimeError(
                            "Duplicate computation"
                        )

                    owner[tile] = COMPLETE

                    rates[wid] = local_rate

                    sequence[wid] += 1

                    if (

                        job_start[j] < 0

                        or t0 < job_start[j]

                    ):

                        job_start[j] = t0

                    job_end[j] = max(
                        job_end[j], t1
                    )

                    remaining[j] -= 1

                    if remaining[j] == 0:

                        emit(

                            "TASK_COMPLETE",

                            job=j

                        )

                evaluate_feedback()

            journal.flush()

            journal.close()

            results.put({

                "wid": wid,

                "mode": mode,

                "repeat": repeat,

                "cpu_seconds": (

                    time.process_time()-cpu_start

                ),

                "idle_seconds": idle_seconds,

                "idle_cpu_seconds": idle_cpu_seconds,

                "remembered_speed": local_rate,

                "handoff_cost": handoff_cost,

                "wakeup_cost": wakeup_cost,

                "affinity_ok": (

                    current_cpu() == (

                        cpu_info["group"],

                        cpu_info["cpu"]

                    )

                ),

                **counters,

                "events": events,

                "intervals": intervals

            })

    except BaseException:

        error_text = traceback.format_exc()

        error_path = (

            OUT / "DIAGNOSTICS"

            / (

                f"{os.getpid()}"

                f"_C{wid:03d}_error.txt"

            )

        )

        error_path.write_text(

            error_text,

            encoding="utf-8"

        )

        results.put({

            "wid": wid,

            "error": traceback.format_exc()

        })

    finally:

        if "journal" in locals():

            if not journal.closed:

                journal.close()

        arrays.clear()

        for shm in handles:

            shm.close()

# ============================================================
# 8. PERMANENT COMPUTATIONAL ENVIRONMENT
# ============================================================

class Environment:

    def __init__(
        self, cpus, jobs, inputs
    ):

        self.cpus = cpus

        self.jobs = jobs

        self.count = len(cpus)

        T0, cap, C, external = inputs

        q = CFG["compute_quantum"]

        self.tile_info = []

        for job in jobs:

            for row in range(

                0, job["rows"], q

            ):

                self.tile_info.append((

                    job["job"],

                    row,

                    min(q, job["rows"]-row)

                ))

        nj = len(jobs)

        nt = len(self.tile_info)

        n = self.count

        total_rows = sum(

            job["rows"]

            for job in jobs

        )

        self.blocks = {

            "T0": SharedBlock(T0.shape, T0),

            "cap": SharedBlock(cap.shape, cap),

            "C": SharedBlock(C.shape, C),

            "output": SharedBlock((

                total_rows,

                CFG["width"]

            ))

        }

        I = ctypes.c_int32

        L = ctypes.c_int64

        D = ctypes.c_double

        self.raw = {

            "owner": CTX.RawArray(I, nt),

            "released": CTX.RawArray(I, nj),

            "remaining": CTX.RawArray(I, nj),

            "idle": CTX.RawArray(I, n),

            "generation": CTX.RawArray(I, n),

            "assignment": CTX.RawArray(I, n),

            "sequence": CTX.RawArray(L, n),

            "rates": CTX.RawArray(D, n),

            "pressure": CTX.RawArray(D, n),

            "job_start": CTX.RawArray(D, nj),

            "job_end": CTX.RawArray(D, nj),

            "job_release": CTX.RawArray(D, nj),

            "assigned_at": CTX.RawArray(D, n),

            "assigned_by": CTX.RawArray(I, n),

            "stop": CTX.RawValue(I, 0)

        }

        self.lock = CTX.Lock()

        self.wakeups = [

            CTX.Event()

            for _ in cpus

        ]

        self.commands = [

            CTX.Queue()

            for _ in cpus

        ]

        self.results = CTX.Queue()

        self.ready = CTX.Queue()

        self.processes = []

        memory = {

            k: b.descriptor()

            for k, b in self.blocks.items()

        }

        t0 = time.perf_counter()

        try:

            for wid, info in enumerate(cpus):

                p = CTX.Process(

                    target=crystal_worker,

                    args=(

                        wid,

                        info,

                        cpus,

                        jobs,

                        memory,

                        self.tile_info,

                        external,

                        self.raw,

                        self.lock,

                        self.wakeups,

                        self.commands[wid],

                        self.results,

                        self.ready

                    )

                )

                p.start()

                self.processes.append(p)

            ready = [

                self.ready.get(timeout=180)

                for _ in cpus

            ]

        except BaseException:

            self.close()

            raise

        self.startup_seconds = (

            time.perf_counter()-t0

        )

        pd.DataFrame(ready).to_json(

            OUT/"persistent_crystals.json",

            orient="records",

            indent=4

        )

        print(

            f"PERMANENT CRYSTALS: {n}",

            flush=True

        )

        print(

            f"STARTUP: {self.startup_seconds:.3f}s",

            flush=True

        )

    def view(
        self, key, dtype=np.int32
    ):

        return np.frombuffer(

            self.raw[key],

            dtype=dtype

        )

    def reset(self):

        with self.lock:

            for key in [

                "owner",

                "released",

                "remaining",

                "idle",

                "generation",

                "assignment",

                "assigned_by"

            ]:

                self.view(key).fill(0)

            self.view("owner").fill(WAITING)

            self.view("assignment").fill(-1)

            self.view("assigned_by").fill(-1)

            self.view(

                "sequence",

                np.int64

            ).fill(0)

            for key in [

                "rates",

                "pressure",

                "assigned_at"

            ]:

                self.view(

                    key,

                    np.float64

                ).fill(0)

            for key in [

                "job_start",

                "job_end",

                "job_release"

            ]:

                self.view(

                    key,

                    np.float64

                ).fill(-1)

            self.view("remaining")[:] = [

                j["tiles"]

                for j in self.jobs

            ]

            self.raw["stop"].value = 0

            for event in self.wakeups:

                event.clear()

        self.blocks["output"].array.fill(

            np.nan

        )

    def assert_alive(self):

        for p in self.processes:

            if p.exitcode is not None:

                try:

                    item = self.results.get_nowait()

                    if "error" in item:

                        raise RuntimeError(
                            item["error"]
                        )

                except Empty:

                    pass

                raise RuntimeError(

                    f"Worker {p.pid} exited: {p.exitcode}"

                )

    def run(self, mode, repeat):

        self.reset()

        # Synchronize the stream start without
        # starting a new set of processes.

        origin = time.perf_counter()+2.0

        for cmd in self.commands:

            cmd.put({

                "type": "RUN",

                "mode": mode,

                "repeat": repeat,

                "origin": origin

            })

        # Wait until the persistent environment
        # reaches initial readiness.
        # This is benchmark setup only.

        deadline = time.perf_counter()+60

        while True:

            with self.lock:

                ready_count = int(

                    self.view("idle").sum()

                )

            if ready_count == self.count:

                break

            self.assert_alive()

            if time.perf_counter() > deadline:

                raise TimeoutError(
                    "Initial readiness timeout"
                )

            time.sleep(0.005)

        if time.perf_counter() >= origin:

            raise RuntimeError(

                "Workers missed synchronized start"

            )

        # =====================================================
        # External task source
        #
        # It publishes tasks but does not assign
        # them to particular crystals.
        # =====================================================

        released = self.view("released")

        job_release = self.view(

            "job_release",

            np.float64

        )

        next_job = 0

        while next_job < len(self.jobs):

            target = (

                origin

                + self.jobs[next_job]["arrival"]

            )

            delay = (

                target-time.perf_counter()

            )

            if delay > 0:

                time.sleep(delay)

            with self.lock:

                now = time.perf_counter()

                while (

                    next_job < len(self.jobs)

                    and origin

                    +self.jobs[next_job]["arrival"]

                    <= now

                ):

                    released[next_job] = 1

                    job_release[next_job] = now

                    next_job += 1

                # Arrival announcement only.
                # Free workers compete to absorb
                # independent external tasks.

                free = np.flatnonzero(

                    self.view("idle") == 1

                )

                for peer in free:

                    self.wakeups[int(peer)].set()

            self.assert_alive()

        # =====================================================
        # Drain the environment.
        #
        # Parent observes completion only;
        # it never distributes computational tiles.
        # =====================================================

        deadline = (

            time.perf_counter()+CFG["timeout"]

        )

        while True:

            with self.lock:

                done = bool(np.all(

                    self.view("remaining") == 0

                ))

            if done:

                break

            self.assert_alive()

            if time.perf_counter() > deadline:

                raise TimeoutError(
                    "Environment drain timeout"
                )

            time.sleep(0.01)

        # Ensure no worker remains inside an
        # unfinished useful computation.

        with self.lock:

            self.raw["stop"].value = 1

            for event in self.wakeups:

                event.set()

        deadline = (

            time.perf_counter()

            + CFG["shutdown_timeout"]

        )

        workers = []

        events = []

        intervals = []

        while len(workers) < self.count:

            try:

                item = self.results.get(

                    timeout=5

                )

            except Empty:

                self.assert_alive()

                if time.perf_counter() > deadline:

                    raise TimeoutError(
                        "Worker result timeout"
                    )

                continue

            if "error" in item:

                raise RuntimeError(item["error"])

            events.extend(item.pop("events"))

            intervals.extend(item.pop("intervals"))

            workers.append(item)

        with self.lock:

            starts = self.view(

                "job_start",

                np.float64

            ).copy()

            ends = self.view(

                "job_end",

                np.float64

            ).copy()

            actual_release = self.view(

                "job_release",

                np.float64

            ).copy()

            ownership_ok = bool(

                np.all(

                    self.view("owner") == COMPLETE

                )

            )

        return {

            "workers": workers,

            "events": events,

            "intervals": intervals,

            "starts": starts-origin,

            "ends": ends-origin,

            "releases": actual_release-origin,

            "ownership_ok": ownership_ok,

            "output": self.blocks["output"].array.copy()

        }

    def close(self):

        with self.lock:

            self.raw["stop"].value = 1

            for event in self.wakeups:

                event.set()

        for cmd in self.commands:

            cmd.put({"type": "STOP"})

        deadline = time.perf_counter()+5

        for p in self.processes:

            p.join(

                timeout=max(

                    0,

                    deadline-time.perf_counter()

                )

            )

        for p in self.processes:

            if p.is_alive():

                p.terminate()

            p.join()

        for block in self.blocks.values():

            block.close()

# ============================================================
# 9. NUMERICAL VALIDATION
#
# Independent serial calculation for every
# unique input segment.
#
# Results reused for all modes and repetitions.
# ============================================================

def make_reference(jobs, inputs):

    T0, cap, C, external = inputs

    cache = {}

    references = []

    started = time.perf_counter()

    for job in jobs:

        key = (

            job["input_offset"],

            job["rows"]

        )

        if key not in cache:

            a = key[0]

            b = a+key[1]

            cache[key] = solve(

                T0[a:b],

                cap[a:b],

                C[:, a:b],

                external

            )

        references.append(cache[key])

    output = np.concatenate(

        references,

        axis=0

    )

    seconds = (

        time.perf_counter()-started

    )

    return output, seconds, len(cache)

# ============================================================
# 10. COLLECTIVE DYNAMICS
#
# Count active computational processes and
# simultaneously active independent tasks.
# ============================================================

def dynamics(intervals):

    changes = []

    for row in intervals:

        changes.append((

            row["start"],

            1,

            row["job"]

        ))

        changes.append((

            row["end"],

            -1,

            row["job"]

        ))

    changes.sort(

        key=lambda x: (x[0], x[1])

    )

    active = 0

    per_job = {}

    records = []

    for t, delta, j in changes:

        active += delta

        per_job[j] = (

            per_job.get(j, 0)+delta

        )

        if per_job[j] == 0:

            del per_job[j]

        records.append({

            "time": t,

            "active": active,

            "simultaneous_collectives": (

                len(per_job)

            )

        })

    return pd.DataFrame(records)

# ============================================================
# 11. MAIN EXPERIMENT
# ============================================================






class CrystalMailbox:

    def __init__(self):

        self.reader, self.writer = CTX.Pipe(
            duplex=False
        )

        # Several senders may address this crystal.
        # Serialize writes to this mailbox only.

        self.send_lock = CTX.Lock()

    def put(self, message):

        with self.send_lock:

            self.writer.send(message)

    def get(self):

        return self.reader.recv()

    def get_nowait(self):

        if not self.reader.poll(0):

            raise Empty

        return self.reader.recv()

    def close(self):

        self.reader.close()

        self.writer.close()


def local_crystal_worker(
    wid, cpu_info, cpus, jobs, memory,
    tile_info, external, state,
    peer_locks, job_locks, source_lock,
    inboxes, results, ready
):
    handles = []
    arrays = {}

    try:
        bind_cpu(cpu_info)

        count = len(cpus)

        for name, desc in memory.items():
            shm, array = attach(desc)
            handles.append(shm)
            arrays[name] = array

        del array

        idle = np.frombuffer(
            state["idle"], dtype=np.int32
        )

        generation = np.frombuffer(
            state["generation"], dtype=np.int32
        )

        rates = np.frombuffer(
            state["rates"], dtype=np.float64
        )

        released = np.frombuffer(
            state["released"], dtype=np.int32
        )

        remaining = np.frombuffer(
            state["remaining"], dtype=np.int32
        )

        starts = np.frombuffer(
            state["starts"], dtype=np.float64
        )

        ends = np.frombuffer(
            state["ends"], dtype=np.float64
        )

        releases = np.frombuffer(
            state["releases"], dtype=np.float64
        )

        ready.put(wid)

        while True:

            cmd = inboxes[wid].get()

            if cmd["type"] == "STOP_PROCESS":
                break

            if cmd["type"] != "RUN":
                raise RuntimeError(
                    "Unexpected command outside trial"
                )

            mode = cmd["mode"]
            repeat = cmd["repeat"]
            origin = cmd["origin"]

            predictive = mode.endswith("PREDICT")

            # Individual crystal memory.
            work = deque()

            pending_ack = {}

            acknowledgements = 0

            received_packets = 0

            received_roots = 0

            signal_queue = deque()
            seen = {}

            local_rate = 0.0
            communication_cost = 0.0
            wakeup_cost = 0.0

            events = []
            intervals = []

            transfers = 0
            prediction_rejects = 0
            tasks_claimed = 0

            idle_seconds = 0.0
            idle_cpu_seconds = 0.0

            cpu_start = time.process_time()

            journal_path = (

                OUT / "LIVE_JOURNALS"

                / (

                    f"{mode}_R{repeat}"

                    f"_C{wid:03d}.jsonl"

                )

            )

            journal = open(

                journal_path,

                "a",

                encoding="utf-8",

                buffering=1

            )

            alpha = CFG["speed_alpha"]

            def emit(kind, **kwargs):

                events.append({
                    "mode": mode,
                    "repeat": repeat,
                    "wid": wid,
                    "time": time.perf_counter()-origin,
                    "event": kind,
                    **kwargs
                })

                journal.write(

                    json.dumps(

                        events[-1],

                        ensure_ascii=False,

                        default=str

                    ) + "\n"

                )


            def handle_ack(message):

                nonlocal acknowledgements

                packet_id = message["packet_id"]

                expected = pending_ack.pop(
                    packet_id, None
                )

                actual = (

                    int(message["peer"]),

                    int(message["generation"])

                )

                if expected is None:

                    raise RuntimeError(
                        f"Unexpected ACK: {packet_id}"
                    )

                if expected != actual:

                    raise RuntimeError(
                        f"ACK identity mismatch: {packet_id}"
                    )

                acknowledgements += 1

                emit(

                    "ACK_RECEIVED",

                    packet_id=packet_id,

                    peer=message["peer"],

                    job=message["job"],

                    generation=message["generation"]

                )


            def drain_control_messages():

                while True:

                    try:

                        message = inboxes[wid].get_nowait()

                    except Empty:

                        return

                    if message["type"] == "ACK":

                        handle_ack(message)

                    elif message["type"] == "STOP_PROCESS":

                        raise SystemExit

                    else:

                        raise RuntimeError(

                            "Unexpected message while computing: "

                            + str(message["type"])

                        )


            def accept_work(message):

                nonlocal wakeup_cost

                nonlocal received_packets

                nonlocal received_roots

                nonlocal tasks_claimed

                kind = message["type"]

                if kind not in ("WORK", "NEW_TASK"):

                    raise RuntimeError(
                        "Invalid computational message"
                    )

                if work:

                    raise RuntimeError(
                        "Work received by a busy crystal"
                    )

                with peer_locks[wid]:

                    if idle[wid] != 0:

                        raise RuntimeError(
                            "Receiver was not reserved"
                        )

                    if (

                        message["generation"]

                        != generation[wid]

                    ):

                        raise RuntimeError(
                            "Receiver generation mismatch"
                        )

                tiles = message["tiles"]

                j = int(message["job"])

                if not tiles:

                    raise RuntimeError(
                        "Empty computational packet"
                    )

                if len(tiles) != len(set(tiles)):

                    raise RuntimeError(
                        "Duplicate tiles inside packet"
                    )

                if any(
                    tile_info[t][0] != j for t in tiles
                ):

                    raise RuntimeError(
                        "Mixed job identifiers"
                    )

                # Computational ownership becomes local.

                work.extend(tiles)

                if kind == "NEW_TASK":

                    tasks_claimed += 1

                    received_roots += 1

                    emit(

                        "TASK_CLAIM",

                        job=j,

                        rows=jobs[j]["rows"],

                        admission="MAILBOX"

                    )

                    return

                received_packets += 1

                latency = max(

                    0.0,

                    time.perf_counter()-message["clock"]

                )

                wakeup_cost = (

                    latency

                    if wakeup_cost <= 0

                    else (

                        (1-alpha)*wakeup_cost

                        + alpha*latency

                    )

                )

                emit(

                    "ABSORB",

                    job=j,

                    donor=message["donor"],

                    packet_id=message["packet_id"],

                    generation=message["generation"],

                    wakeup_latency=latency

                )

                # Send explicit acknowledgement through
                # the donor's existing personal mailbox.
                #
                # The donor does not wait for ACK.
                # It continues computing its own tiles.

                ack = {

                    "type": "ACK",

                    "packet_id": message["packet_id"],

                    "peer": wid,

                    "generation": int(generation[wid]),

                    "job": j

                }

                inboxes[message["donor"]].put(ack)

                emit(

                    "ACK_SENT",

                    job=j,

                    packet_id=message["packet_id"],

                    donor=message["donor"]

                )

            def claim_external():

                nonlocal tasks_claimed

                # Protect only admission and this crystal's
                # readiness. No tile ownership lock.

                with peer_locks[wid]:

                    # A donor has already reserved us.
                    if idle[wid] != 1:
                        return False

                    with source_lock:

                        available = np.flatnonzero(
                            released == 1
                        )

                        if not available.size:
                            return False

                        j = int(available[0])

                        released[j] = 2
                        idle[wid] = 0

                        job = jobs[j]

                        work.extend(
                            range(
                                job["tile_start"],
                                job["tile_end"]
                            )
                        )

                tasks_claimed += 1

                emit(
                    "TASK_CLAIM",
                    job=j,
                    rows=job["rows"]
                )

                return True

            def collect_signals():

                # Advisory observations of readiness.
                # Authoritative verification happens
                # under the selected receiver's lock.

                for peer in range(count):

                    if peer == wid or idle[peer] != 1:
                        continue

                    gen = int(generation[peer])

                    if seen.get(peer) == gen:
                        continue

                    seen[peer] = gen

                    signal_queue.append((peer, gen))

            def share_work():

                nonlocal transfers
                nonlocal prediction_rejects
                nonlocal communication_cost

                W = len(work)

                if W < 2:
                    return

                if not np.any(idle == 1):
                    return

                collect_signals()

                checked = len(signal_queue)

                while checked:

                    checked -= 1

                    peer, gen = signal_queue.popleft()

                    if idle[peer] != 1:
                        continue

                    n = W//2

                    # All tiles currently owned by this
                    # crystal belong to the same task.

                    j = tile_info[work[0]][0]

                    donor_tiles = list(work)

                    donated = donor_tiles[-n:]
                    kept = donor_tiles[:-n]

                    donated_rows = sum(
                        tile_info[t][2]
                        for t in donated
                    )

                    kept_rows = sum(
                        tile_info[t][2]
                        for t in kept
                    )

                    started = time.perf_counter()

                    with peer_locks[peer]:

                        # Atomic readiness verification.
                        if (
                            idle[peer] != 1
                            or generation[peer] != gen
                        ):
                            continue

                        peer_rate = float(rates[peer])

                        if peer_rate <= 0:
                            peer_rate = local_rate

                        alone = None
                        together = None

                        if (
                            predictive
                            and local_rate > 0
                            and peer_rate > 0
                        ):

                            alone = (
                                kept_rows+donated_rows
                            )/local_rate

                            cost = (
                                communication_cost
                                + wakeup_cost
                                + time.perf_counter()
                                - started
                            )

                            together = cost + max(
                                kept_rows/local_rate,
                                donated_rows/peer_rate
                            )

                            if together >= alone:

                                prediction_rejects += 1

                                signal_queue.append(
                                    (peer, gen)
                                )

                                emit(
                                    "PREDICTION_REJECT",
                                    job=j,
                                    peer=peer,
                                    alone=alone,
                                    together=together
                                )

                                continue

                        # Reserve receiver locally.
                        idle[peer] = 0

                        packet_id = (
                            f"{wid}:{transfers}:{gen}"
                        )

                        packet = {
                            "type": "WORK",
                            "packet_id": packet_id,
                            "job": j,
                            "donor": wid,
                            "generation": gen,
                            "tiles": donated,
                            "clock": time.perf_counter()
                        }

                        # Transfer actual local work,
                        # not global ownership rights.

                        for _ in range(n):
                            work.pop()

                        pending_ack[packet_id] = (
                            peer, gen
                        )

                        try:

                            inboxes[peer].put(packet)

                        except BaseException:

                            pending_ack.pop(
                                packet_id, None
                            )

                            # A send failure is fatal for
                            # this trial: delivery may be
                            # ambiguous after a pipe error.

                            raise

                    transfers += 1

                    spent = (
                        time.perf_counter()-started
                    )

                    communication_cost = (
                        spent
                        if communication_cost <= 0
                        else (
                            (1-alpha)*communication_cost
                            +alpha*spent
                        )
                    )

                    emit(
                        "TRANSFER",
                        job=j,
                        peer=peer,
                        generation=gen,
                        packet_id=packet_id,
                        pending_tiles=W,
                        donated_tiles=n,
                        donated_rows=donated_rows,
                        retained_rows=kept_rows,
                        predicted_alone=alone,
                        predicted_together=together
                    )

                    break

            # =============================================
            # AUTONOMOUS LIFECYCLE
            # =============================================

            while True:

                if not work:

                    # Preserve the original readiness
                    # generation and individual experience.

                    with peer_locks[wid]:

                        if idle[wid] == 2:

                            generation[wid] += 1

                            rates[wid] = local_rate

                            idle[wid] = 1

                            signal_queue.clear()

                            seen.clear()

                            emit(

                                "IDLE",

                                generation=int(
                                    generation[wid]
                                ),

                                remembered_speed=local_rate

                            )

                    # A crystal may autonomously absorb
                    # a task left waiting in the environment.

                    if claim_external():

                        continue

                    t0 = time.perf_counter()

                    c0 = time.process_time()

                    # No timeout here.
                    # Absence of work is not a failure.

                    message = inboxes[wid].get()

                    idle_seconds += (

                        time.perf_counter()-t0

                    )

                    idle_cpu_seconds += (

                        time.process_time()-c0

                    )

                    kind = message["type"]

                    if kind == "STOP_PROCESS":

                        return

                    if kind == "STOP_TRIAL":

                        if pending_ack:

                            raise RuntimeError(

                                "Trial stopped with pending ACKs: "

                                + str(list(pending_ack))

                            )

                        emit(

                            "STOP_TRIAL_RECEIVED",

                            acknowledgements=acknowledgements,

                            received_packets=received_packets

                        )

                        break

                    if kind == "ACK":

                        handle_ack(message)

                        continue

                    if kind in ("WORK", "NEW_TASK"):

                        accept_work(message)

                        continue

                    raise RuntimeError(

                        "Unknown message: " + str(kind)

                    )

                # Consume ACK messages between computations.
                # No idle polling.

                drain_control_messages()

                # Local work sharing.
                share_work()

                # The owner takes a tile from its own memory.
                tile = work.popleft()

                j, row, nrows = tile_info[tile]

                job = jobs[j]

                a = job["input_offset"]+row
                b = a+nrows

                oa = job["output_offset"]+row
                ob = oa+nrows

                t0 = time.perf_counter()

                block = solve(
                    arrays["T0"][a:b],
                    arrays["cap"][a:b],
                    arrays["C"][:, a:b],
                    external
                )

                arrays["output"][oa:ob] = block

                t1 = time.perf_counter()

                duration = max(t1-t0, 1e-12)

                instant_rate = nrows/duration

                local_rate = (
                    instant_rate
                    if local_rate <= 0
                    else (
                        (1-alpha)*local_rate
                        +alpha*instant_rate
                    )
                )

                intervals.append({
                    "mode": mode,
                    "repeat": repeat,
                    "wid": wid,
                    "job": j,
                    "tile": tile,
                    "rows": nrows,
                    "start": t0-origin,
                    "end": t1-origin,
                    "speed": instant_rate
                })

                # Independent synchronization per task.
                # Unrelated collectives do not compete
                # for one completion lock.

                with job_locks[j]:

                    if starts[j] < 0 or t0 < starts[j]:
                        starts[j] = t0

                    ends[j] = max(ends[j], t1)

                    remaining[j] -= 1

                    if remaining[j] < 0:
                        raise RuntimeError(
                            "Duplicate tile completion"
                        )

                    if remaining[j] == 0:

                        emit(
                            "TASK_COMPLETE",
                            job=j
                        )

                with peer_locks[wid]:

                    rates[wid] = local_rate

                    if not work:
                        idle[wid] = 2

            journal.flush()

            journal.close()

            results.put({
                "wid": wid,
                "mode": mode,
                "repeat": repeat,
                "cpu_seconds":
                    time.process_time()-cpu_start,
                "idle_seconds": idle_seconds,
                "idle_cpu_seconds": idle_cpu_seconds,
                "remembered_speed": local_rate,
                "handoff_cost": communication_cost,
                "wakeup_cost": wakeup_cost,
                "affinity_ok": (
                    current_cpu() == (
                        cpu_info["group"],
                        cpu_info["cpu"]
                    )
                ),
                "transfers": transfers,

                "acknowledgements": acknowledgements,

                "received_packets": received_packets,

                "received_roots": received_roots,

                "pending_acknowledgements":
                    len(pending_ack),
                "prediction_rejects": prediction_rejects,
                "overload_rejects": 0,
                "overload_pulses": 0,
                "tasks_claimed": tasks_claimed,
                "feedback_observations": 0,
                "feedback_censored": 0,
                "events": events,
                "intervals": intervals
            })

    except BaseException:

        error_text = traceback.format_exc()

        error_path = (

            OUT / "DIAGNOSTICS"

            / (

                f"{os.getpid()}"

                f"_C{wid:03d}_error.txt"

            )

        )

        error_path.write_text(

            error_text,

            encoding="utf-8"

        )

        results.put({
            "wid": wid,
            "error": traceback.format_exc()
        })

    finally:

        if "journal" in locals():

            if not journal.closed:

                journal.close()

        arrays.clear()

        for shm in handles:
            shm.close()


# ============================================================
# LOCAL COMPUTATIONAL ENVIRONMENT
# ============================================================


class LocalEnvironment:

    def __init__(self, cpus, jobs, inputs):

        self.cpus = cpus
        self.jobs = jobs
        self.count = len(cpus)

        T0, cap, C, self.external = inputs

        q = CFG["compute_quantum"]

        self.tile_info = [
            (
                job["job"],
                row,
                min(q, job["rows"]-row)
            )
            for job in jobs
            for row in range(0, job["rows"], q)
        ]

        n = self.count
        nj = len(jobs)

        self.blocks = {
            "T0": SharedBlock(T0.shape, T0),
            "cap": SharedBlock(cap.shape, cap),
            "C": SharedBlock(C.shape, C),
            "output": SharedBlock((
                sum(j["rows"] for j in jobs),
                CFG["width"]
            ))
        }

        I = ctypes.c_int32
        D = ctypes.c_double

        self.state = {
            "idle": CTX.RawArray(I, n),
            "generation": CTX.RawArray(I, n),
            "rates": CTX.RawArray(D, n),
            "released": CTX.RawArray(I, nj),
            "remaining": CTX.RawArray(I, nj),
            "starts": CTX.RawArray(D, nj),
            "ends": CTX.RawArray(D, nj),
            "releases": CTX.RawArray(D, nj),
            "stop": CTX.RawValue(I, 0)
        }

        self.peer_locks = [
            CTX.Lock() for _ in cpus
        ]

        self.job_locks = [
            CTX.Lock() for _ in jobs
        ]

        self.source_lock = CTX.Lock()

        self.inboxes = [

            CrystalMailbox()

            for _ in cpus

        ]

        self.trial_active = False

        self.current_trial = None

        self.results = CTX.Queue()
        self.ready = CTX.Queue()

        self.processes = []

        memory = {
            k: v.descriptor()
            for k, v in self.blocks.items()
        }

        try:

            for wid, cpu in enumerate(cpus):

                p = CTX.Process(
                    target=local_crystal_worker,
                    args=(
                        wid, cpu, cpus, jobs,
                        memory, self.tile_info,
                        self.external, self.state,
                        self.peer_locks,
                        self.job_locks,
                        self.source_lock,
                        self.inboxes,
                        self.results,
                        self.ready
                    )
                )

                p.start()
                self.processes.append(p)

            for _ in cpus:
                self.ready.get(timeout=180)

        except BaseException:
            self.close()
            raise

        print(
            f"LOCAL CRYSTALS: {n}",
            flush=True
        )

    def view(self, key, dtype=np.int32):

        return np.frombuffer(
            self.state[key],
            dtype=dtype
        )

    def assert_alive(self):

        for p in self.processes:

            if p.exitcode is not None:
                raise RuntimeError(
                    f"Local crystal {p.pid} exited: "
                    f"{p.exitcode}"
                )

    def reset(self):

        self.view("idle").fill(2)
        self.view("generation").fill(0)
        self.view("released").fill(0)

        self.view(
            "rates", np.float64
        ).fill(0)

        for name in [
            "starts", "ends", "releases"
        ]:
            self.view(
                name, np.float64
            ).fill(-1)

        self.view("remaining")[:] = [
            j["tiles"] for j in self.jobs
        ]

        self.state["stop"].value = 0

        self.blocks["output"].array.fill(np.nan)

    def diagnose(self, reason):

        snapshot = {

            "reason": str(reason),

            "trial": self.current_trial,

            "remaining_total": int(

                self.view("remaining").sum()

            ),

            "remaining_by_job":

                self.view("remaining").tolist(),

            "released":

                self.view("released").tolist(),

            "workers": [

                {

                    "wid": i,

                    "pid": p.pid,

                    "exitcode": p.exitcode,

                    "alive": p.is_alive(),

                    "state": int(

                        self.view("idle")[i]

                    ),

                    "generation": int(

                        self.view("generation")[i]

                    ),

                    "remembered_speed": float(

                        self.view(
                            "rates", np.float64
                        )[i]

                    )

                }

                for i, p in enumerate(self.processes)

            ]

        }

        path = (

            OUT / "DIAGNOSTICS"

            / (

                "failure_"

                + str(time.time_ns())

                + ".json"

            )

        )

        path.write_text(

            json.dumps(snapshot, indent=4),

            encoding="utf-8"

        )

        print(

            "DIAGNOSTICS:", path,

            flush=True

        )


    def run(self, mode, repeat):

        self.trial_active = True

        self.current_trial = {

            "mode": mode,

            "repeat": repeat

        }

        try:

            result = self._run(mode, repeat)

        except BaseException as error:

            self.diagnose(error)

            raise

        self.trial_active = False

        return result

    def _run(self, mode, repeat):

        self.reset()

        origin = time.perf_counter()+2.0

        for q in self.inboxes:

            q.put({
                "type": "RUN",
                "mode": mode,
                "repeat": repeat,
                "origin": origin
            })

        deadline = time.perf_counter()+60

        while not np.all(
            self.view("idle") == 1
        ):

            self.assert_alive()

            if time.perf_counter() > deadline:
                raise TimeoutError(
                    "Local initialization timeout"
                )

            time.sleep(0.005)

        if time.perf_counter() >= origin:
            raise RuntimeError(
                "Local crystals missed start"
            )

        released = self.view("released")

        release_times = self.view(
            "releases", np.float64
        )

        next_job = 0

        # External workload generator.
        # Identical scheduled arrivals in every mode.

        while next_job < len(self.jobs):

            target = (
                origin
                +self.jobs[next_job]["arrival"]
            )

            delay = target-time.perf_counter()

            if delay > 0:
                time.sleep(delay)

            with self.source_lock:

                now = time.perf_counter()

                while (
                    next_job < len(self.jobs)
                    and origin
                    +self.jobs[next_job]["arrival"]
                    <= now
                ):

                    released[next_job] = 1

                    release_times[next_job] = now

                    next_job += 1

            # Targeted admission of external tasks.

            for peer in range(self.count):

                with self.peer_locks[peer]:

                    if self.view("idle")[peer] != 1:

                        continue

                    with self.source_lock:

                        available = np.flatnonzero(
                            released == 1
                        )

                        if not available.size:

                            break

                        j = int(available[0])

                        job = self.jobs[j]

                        released[j] = 2

                        self.view("idle")[peer] = 0

                        gen = int(

                            self.view("generation")[peer]

                        )

                    message = {

                        "type": "NEW_TASK",

                        "job": j,

                        "generation": gen,

                        "clock": time.perf_counter(),

                        "tiles": list(

                            range(

                                job["tile_start"],

                                job["tile_end"]

                            )

                        )

                    }

                    self.inboxes[peer].put(message)

            self.assert_alive()

        deadline = (
            time.perf_counter()+CFG["timeout"]
        )

        while not np.all(
            self.view("remaining") == 0
        ):

            self.assert_alive()

            if time.perf_counter() > deadline:
                raise TimeoutError(
                    "Local environment drain timeout"
                )

            time.sleep(0.01)

        # All computational tiles are complete.
        # Every WORK acknowledgement has already
        # been sent before its receiver computed.

        for mailbox in self.inboxes:

            mailbox.put({
                "type": "STOP_TRIAL"
            })

        deadline = (

            time.perf_counter()

            + CFG["shutdown_timeout"]

        )

        workers = []
        events = []
        intervals = []

        while len(workers) < self.count:

            try:

                item = self.results.get(timeout=1)

            except Empty:

                self.assert_alive()

                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        "Local result timeout"
                    )

                continue

            if "error" in item:
                raise RuntimeError(item["error"])

            events.extend(item.pop("events"))
            intervals.extend(item.pop("intervals"))
            workers.append(item)

        # Post-run verification, not a runtime
        # ownership table.

        actual = sorted(
            x["tile"] for x in intervals
        )

        expected = list(
            range(len(self.tile_info))
        )

        sent = [
            e["packet_id"]
            for e in events
            if e["event"] == "TRANSFER"
        ]

        received = [
            e["packet_id"]
            for e in events
            if e["event"] == "ABSORB"
        ]

        acknowledged = [

            e["packet_id"]

            for e in events

            if e["event"] == "ACK_RECEIVED"

        ]

        acknowledgements_sent = [

            e["packet_id"]

            for e in events

            if e["event"] == "ACK_SENT"

        ]

        stopped_workers = [

            e["wid"]

            for e in events

            if e["event"] == "STOP_TRIAL_RECEIVED"

        ]

        ack_ok = bool(

            sorted(sent) == sorted(acknowledged)

            and sorted(sent)
                == sorted(acknowledgements_sent)

            and sum(
                w["pending_acknowledgements"]
                for w in workers
            ) == 0

        )

        stop_ok = (

            sorted(stopped_workers)

            == list(range(self.count))

        )

        ownership_ok = bool(

            ack_ok

            and stop_ok

            and actual == expected
            and sorted(sent) == sorted(received)
            and len(sent) == len(set(sent))
            and np.all(
                self.view("remaining") == 0
            )
        )

        return {
            "workers": workers,
            "events": events,
            "intervals": intervals,
            "starts": self.view(
                "starts", np.float64
            ).copy()-origin,
            "ends": self.view(
                "ends", np.float64
            ).copy()-origin,
            "releases": release_times.copy()-origin,
            "ownership_ok": ownership_ok,
            "output":
                self.blocks["output"].array.copy()
        }

    def close(self):

        # On a successful trial, processes are
        # waiting for a RUN or STOP_PROCESS message.
        #
        # On failure, do not attempt to obtain
        # locks potentially held by a failed worker.

        if not self.trial_active:

            for i, p in enumerate(self.processes):

                if p.is_alive():

                    try:

                        self.inboxes[i].put({

                            "type": "STOP_PROCESS"

                        })

                    except (OSError, EOFError):

                        pass

        else:

            for p in self.processes:

                if p.is_alive():

                    p.terminate()

        deadline = time.perf_counter() + 5

        for p in self.processes:

            p.join(

                timeout=max(

                    0,

                    deadline-time.perf_counter()

                )

            )

        for p in self.processes:

            if p.is_alive():

                p.terminate()

                p.join(timeout=5)

            if p.is_alive():

                p.kill()

                p.join(timeout=5)

        for mailbox in self.inboxes:

            mailbox.close()

        for q in [self.results, self.ready]:

            q.cancel_join_thread()

            q.close()

        for block in self.blocks.values():

            block.close()




class ComparativeEnvironment:

    def __init__(self, cpus, jobs, inputs):

        self.global_env = None
        self.local_env = None

        try:

            self.global_env = Environment(
                cpus, jobs, inputs
            )

            self.local_env = LocalEnvironment(
                cpus, jobs, inputs
            )

        except BaseException:

            self.close()
            raise

        self.tile_info = self.global_env.tile_info

    def run(self, mode, repeat):

        if mode.startswith("GLOBAL_"):

            old_mode = mode.replace(
                "GLOBAL_", "WAVE_", 1
            )

            res = self.global_env.run(
                old_mode, repeat
            )

            # Normalize labels for common analysis.

            for key in [
                "workers", "events", "intervals"
            ]:

                for item in res[key]:
                    item["mode"] = mode

            return res

        return self.local_env.run(
            mode, repeat
        )

    def close(self):

        if self.local_env is not None:
            self.local_env.close()

        if self.global_env is not None:
            self.global_env.close()


def main():

    cpus = topology()

    pd.DataFrame(cpus).to_csv(

        OUT/"cpu_topology.csv",

        index=False

    )

    print(

        pd.DataFrame(cpus).groupby(

            ["group", "numa"]

        ).size(),

        flush=True

    )

    jobs = make_jobs()

    pd.DataFrame(jobs).to_csv(

        OUT/"external_task_stream.csv",

        index=False

    )

    print(

        f"EXTERNAL JOBS: {len(jobs)}",

        flush=True

    )

    print(

        f"TOTAL ROWS: {sum(j['rows'] for j in jobs)}",

        flush=True

    )

    inputs = make_inputs()

    reference, ref_seconds, ref_calculations = (

        make_reference(jobs, inputs)

    )

    reference_hash = hashlib.sha256(

        reference.tobytes()

    ).hexdigest()

    print(

        f"REFERENCE: {ref_seconds:.3f}s",

        flush=True

    )

    env = ComparativeEnvironment(cpus, jobs, inputs)

    all_jobs = []

    all_workers = []

    all_intervals = []

    all_events = []

    all_dynamics = []

    trial_records = []

    rng = np.random.default_rng(

        CFG["seed"]+331

    )

    expected = (

        CFG["repeats"]

        *len(CFG["modes"])

    )

    completed = 0

    try:

        for repeat in range(

            CFG["repeats"]

        ):

            order = rng.permutation(

                CFG["modes"]

            )

            for mode in order:

                mode = str(mode)

                print(

                    f"\nRUN {completed+1}/{expected}: "

                    f"{mode}, repeat={repeat}",

                    flush=True

                )

                res = env.run(

                    mode, repeat

                )

                output = res["output"]

                error = float(

                    np.max(

                        np.abs(output-reference)

                    )

                )

                checksum = hashlib.sha256(

                    output.tobytes()

                ).hexdigest()

                expected_tiles = list(

                    range(len(env.tile_info))

                )

                actual_tiles = sorted(

                    x["tile"]

                    for x in res["intervals"]

                )

                unique_ok = (

                    expected_tiles == actual_tiles

                )

                affinity_ok = all(

                    w["affinity_ok"]

                    for w in res["workers"]

                )

                transfers = [

                    e for e in res["events"]

                    if e["event"] == "TRANSFER"

                ]

                absorptions = [

                    e for e in res["events"]

                    if e["event"] == "ABSORB"

                ]

                reservations = [

                    (

                        e["peer"],

                        e["generation"]

                    )

                    for e in transfers

                ]

                reservations_ok = (

                    len(reservations)

                    == len(set(reservations))

                )

                half_ok = all(

                    e["donated_tiles"]

                    == e["pending_tiles"]//2

                    for e in transfers

                )

                valid = bool(

                    res["ownership_ok"]

                    and unique_ok

                    and affinity_ok

                    and reservations_ok

                    and half_ok

                    and len(transfers) == len(absorptions)

                    and np.isfinite(output).all()

                    and error <= 1e-9

                    and checksum == reference_hash

                    and np.all(res["starts"] >= res["releases"])

                    and np.all(res["ends"] >= res["starts"])

                )

                intervals = pd.DataFrame(

                    res["intervals"]

                )

                trial_jobs = []

                for job in jobs:

                    j = job["job"]

                    sub = intervals[

                        intervals["job"] == j

                    ]

                    jd = dynamics(

                        sub.to_dict("records")

                    )

                    row = {

                        "mode": mode,

                        "repeat": repeat,

                        "job": j,

                        "rows": job["rows"],

                        "scheduled_arrival":

                            job["arrival"],

                        "actual_arrival":

                            res["releases"][j],

                        "arrival_lateness":

                            res["releases"][j]

                            -job["arrival"],

                        "start": res["starts"][j],

                        "finish": res["ends"][j],

                        "response_seconds":

                            res["starts"][j]

                            -res["releases"][j],

                        "completion_seconds":

                            res["ends"][j]

                            -res["releases"][j],

                        "compute_lifetime":

                            res["ends"][j]

                            -res["starts"][j],

                        "participating_crystals":

                            sub["wid"].nunique(),

                        "peak_active":

                            int(jd["active"].max()),

                        "compute_wall_sum":

                            float(

                                (

                                    sub["end"]

                                    -sub["start"]

                                ).sum()

                            )

                    }

                    trial_jobs.append(row)

                job_df = pd.DataFrame(

                    trial_jobs

                )

                dyn = dynamics(

                    res["intervals"]

                )

                dyn["mode"] = mode

                dyn["repeat"] = repeat

                horizon = max(

                    float(res["ends"].max()),

                    CFG["duration_seconds"]

                )

                # Time-weighted activity over
                # the entire observation horizon.

                t = np.r_[

                    0.0,

                    dyn["time"].to_numpy(),

                    horizon

                ]

                a = np.r_[

                    0,

                    dyn["active"].to_numpy(),

                    0

                ]

                mean_active = float(

                    np.sum(

                        np.diff(t)*a[:-1]

                    )/horizon

                )

                workers = pd.DataFrame(

                    res["workers"]

                )

                row = {

                    "mode": mode,

                    "repeat": repeat,

                    "jobs": len(jobs),

                    "horizon_seconds": horizon,

                    "median_response_ms":

                        1000*job_df[

                            "response_seconds"

                        ].median(),

                    "p95_response_ms":

                        1000*job_df[

                            "response_seconds"

                        ].quantile(0.95),

                    "median_completion_ms":

                        1000*job_df[

                            "completion_seconds"

                        ].median(),

                    "p95_completion_ms":

                        1000*job_df[

                            "completion_seconds"

                        ].quantile(0.95),

                    "max_completion_ms":

                        1000*job_df[

                            "completion_seconds"

                        ].max(),

                    "cpu_seconds":

                        workers["cpu_seconds"].sum(),

                    "cpu_seconds_per_job":

                        workers["cpu_seconds"].sum()

                        /len(jobs),

                    "idle_cpu_seconds":

                        workers["idle_cpu_seconds"].sum(),

                    "mean_active": mean_active,

                    "peak_active":

                        int(dyn["active"].max()),

                    "peak_simultaneous_collectives":

                        int(dyn[

                            "simultaneous_collectives"

                        ].max()),

                    "mean_job_peak":

                        job_df["peak_active"].mean(),

                    "jobs_reaching_full_collective":

                        int((

                            job_df["peak_active"]

                            == len(cpus)

                        ).sum()),

                    "transfers":

                        workers["transfers"].sum(),

                    "prediction_rejects":

                        workers["prediction_rejects"].sum(),

                    "overload_rejects":

                        workers["overload_rejects"].sum(),

                    "overload_pulses":

                        workers["overload_pulses"].sum(),

                    "feedback_observations":

                        workers["feedback_observations"].sum(),

                    "feedback_censored":

                        workers["feedback_censored"].sum(),

                    "max_arrival_lateness_ms":

                        1000*job_df[

                            "arrival_lateness"

                        ].max(),

                    "ack_sent": sum(

                        e["event"] == "ACK_SENT"

                        for e in res["events"]

                    ),

                    "ack_received": sum(

                        e["event"] == "ACK_RECEIVED"

                        for e in res["events"]

                    ),

                    "mailbox_task_roots": sum(

                        e["event"] == "TASK_CLAIM"

                        and e.get("admission")
                            == "MAILBOX"

                        for e in res["events"]

                    ),

                    "stop_messages_received": sum(

                        e["event"]
                            == "STOP_TRIAL_RECEIVED"

                        for e in res["events"]

                    ),

                    "max_error": error,

                    "checksum": checksum,

                    "ownership_ok": res["ownership_ok"],

                    "unique_tiles_ok": unique_ok,

                    "reservations_ok": reservations_ok,

                    "half_split_ok": half_ok,

                    "affinity_ok": affinity_ok,

                    "valid": valid

                }

                all_jobs.extend(trial_jobs)

                all_workers.extend(res["workers"])

                all_events.extend(res["events"])

                all_intervals.extend(res["intervals"])

                all_dynamics.append(dyn)

                trial_records.append(row)

                completed += 1

                pd.DataFrame(trial_records).to_csv(

                    OUT/"benchmark_raw.csv",

                    index=False

                )

                pd.DataFrame(all_jobs).to_csv(

                    OUT/"JOB_RESULTS.csv",

                    index=False

                )

                pd.DataFrame(all_workers).to_csv(

                    OUT/"CRYSTAL_EXPERIENCE.csv",

                    index=False

                )

                print(

                    f"jobs={len(jobs)} "

                    f"response={row['median_response_ms']:.3f}ms "

                    f"completion={row['median_completion_ms']:.3f}ms "

                    f"CPU={row['cpu_seconds']:.3f}s "

                    f"peak={row['peak_active']} "

                    f"collectives={row['peak_simultaneous_collectives']} "

                    f"pulses={row['overload_pulses']} "

                    f"valid={valid}",

                    flush=True

                )

                if not valid:

                    raise RuntimeError(

                        "Numerical or protocol validation failed"

                    )

    finally:

        env.close()

        pd.DataFrame(all_events).to_csv(

            OUT/"ALL_EVENTS.csv",

            index=False

        )

        pd.DataFrame(all_intervals).to_csv(

            OUT/"COMPUTATION_INTERVALS.csv",

            index=False

        )

        if all_dynamics:

            pd.concat(

                all_dynamics,

                ignore_index=True

            ).to_csv(

                OUT/"COLLECTIVE_DYNAMICS.csv",

                index=False

            )

    # ========================================================
    # 12. STATISTICAL ANALYSIS
    # ========================================================

    df = pd.DataFrame(trial_records)

    job_df = pd.DataFrame(all_jobs)

    dyn_df = pd.concat(

        all_dynamics,

        ignore_index=True

    )

    summary = df.groupby("mode").agg(

        median_response_ms=(

            "median_response_ms", "median"

        ),

        p95_response_ms=(

            "p95_response_ms", "median"

        ),

        median_completion_ms=(

            "median_completion_ms", "median"

        ),

        p95_completion_ms=(

            "p95_completion_ms", "median"

        ),

        cpu_seconds=(

            "cpu_seconds", "median"

        ),

        cpu_seconds_per_job=(

            "cpu_seconds_per_job", "median"

        ),

        idle_cpu_seconds=(

            "idle_cpu_seconds", "median"

        ),

        mean_active=(

            "mean_active", "median"

        ),

        peak_active=(

            "peak_active", "max"

        ),

        mean_job_peak=(

            "mean_job_peak", "median"

        ),

        peak_collectives=(

            "peak_simultaneous_collectives", "max"

        ),

        transfers=(

            "transfers", "sum"

        ),

        prediction_rejects=(

            "prediction_rejects", "sum"

        ),

        overload_rejects=(

            "overload_rejects", "sum"

        ),

        overload_pulses=(

            "overload_pulses", "sum"

        ),

        valid=(

            "valid", "all"

        )

    ).reset_index()

    summary.to_csv(

        OUT/"benchmark_summary.csv",

        index=False

    )

    # Performance by task size.

    by_size = job_df.groupby(

        ["mode", "rows"]

    ).agg(

        jobs=("job", "count"),

        median_completion=(

            "completion_seconds", "median"

        ),

        p95_completion=(

            "completion_seconds",

            lambda x: x.quantile(0.95)

        ),

        median_response=(

            "response_seconds", "median"

        ),

        peak_collective=(

            "peak_active", "median"

        ),

        participating=(

            "participating_crystals", "median"

        )

    ).reset_index()

    by_size.to_csv(

        OUT/"PERFORMANCE_BY_JOB_SIZE.csv",

        index=False

    )

    # ========================================================
    # 13. PERFORMANCE FIGURES
    # ========================================================

    for metric, label in [

        (

            "median_completion_ms",

            "Median task completion, ms"

        ),

        (

            "p95_completion_ms",

            "95th percentile completion, ms"

        ),

        (

            "median_response_ms",

            "Task response delay, ms"

        ),

        (

            "cpu_seconds_per_job",

            "CPU seconds per task"

        ),

        (

            "idle_cpu_seconds",

            "Idle CPU seconds"

        ),

        (

            "mean_active",

            "Mean active crystals"

        )

    ]:

        fig, ax = plt.subplots(

            figsize=(10, 6)

        )

        data = summary.sort_values(metric)

        ax.barh(

            data["mode"],

            data[metric]

        )

        ax.invert_yaxis()

        ax.set_xlabel(label)

        ax.set_title(

            "KH-V-03.3b.2 — "+label

        )

        ax.grid(axis="x", alpha=0.25)

        fig.tight_layout()

        fig.savefig(

            OUT/f"PERFORMANCE_{metric}.png",

            dpi=150

        )

        plt.close(fig)

    # ========================================================
    # 14. LIFECYCLE OF THE COMPUTATIONAL FIELD
    # ========================================================

    for (mode, repeat), group in dyn_df.groupby(

        ["mode", "repeat"]

    ):

        fig, ax = plt.subplots(

            figsize=(16, 6)

        )

        group = group.sort_values(

            "time",

            kind="stable"

        )

        ax.step(

            group["time"],

            group["active"],

            where="post",

            label="Active crystals"

        )

        for job in jobs:

            ax.axvline(

                job["arrival"],

                alpha=0.09,

                linewidth=0.8

            )

        ax.set_ylim(

            0, len(cpus)+1

        )

        ax.set_xlabel("Seconds")

        ax.set_ylabel("Active crystals")

        ax.set_title(

            f"{mode} — continuous environment R{repeat}"

        )

        ax.grid(alpha=0.20)

        ax.legend()

        fig.tight_layout()

        fig.savefig(

            OUT/f"ENVIRONMENT_{mode}_R{repeat}.png",

            dpi=160

        )

        plt.close(fig)

        # Number of simultaneously computing jobs.

        fig, ax = plt.subplots(

            figsize=(16, 5)

        )

        ax.step(

            group["time"],

            group["simultaneous_collectives"],

            where="post"

        )

        ax.set_xlabel("Seconds")

        ax.set_ylabel("Simultaneous collectives")

        ax.set_title(

            f"{mode} — independent collectives R{repeat}"

        )

        ax.grid(alpha=0.25)

        fig.tight_layout()

        fig.savefig(

            OUT/f"COLLECTIVES_{mode}_R{repeat}.png",

            dpi=150

        )

        plt.close(fig)

    # ========================================================
    # 15. TASK SIZE VS COLLECTIVE SIZE
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(11, 6)

    )

    for mode, group in by_size.groupby("mode"):

        ax.plot(

            group["rows"],

            group["peak_collective"],

            marker="o",

            label=mode

        )

    ax.set_xlabel("Task size, rows")

    ax.set_ylabel("Median peak collective")

    ax.set_title(

        "Emergent collective size"

    )

    ax.grid(alpha=0.25)

    ax.legend()

    fig.tight_layout()

    fig.savefig(

        OUT/"EMERGENT_COLLECTIVE_SIZE.png",

        dpi=160

    )

    plt.close(fig)

    # ========================================================
    # 16. TASK COMPLETION DISTRIBUTION
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(11, 6)

    )

    for mode, group in job_df.groupby("mode"):

        x = np.sort(

            group["completion_seconds"].to_numpy()

            *1000

        )

        y = (

            np.arange(1, len(x)+1)

            /len(x)

        )

        ax.step(

            x, y,

            where="post",

            label=mode

        )

    ax.set_xlabel(

        "Task completion time, ms"

    )

    ax.set_ylabel(

        "Fraction of completed tasks"

    )

    ax.set_title(

        "Task latency distribution"

    )

    ax.grid(alpha=0.25)

    ax.legend()

    fig.tight_layout()

    fig.savefig(

        OUT/"TASK_LATENCY_CDF.png",

        dpi=160

    )

    plt.close(fig)

    # ========================================================
    # 17. FINAL REPORT
    # ========================================================

    report = {

        "experiment": CFG["experiment"],

        "expected_trials": expected,

        "completed_trials": completed,

        "all_valid": bool(df["valid"].all()),

        "unique_checksums":

            int(df["checksum"].nunique()),

        "reference_seconds": ref_seconds,

        "unique_reference_segments":

            ref_calculations,

        "reference_sha256":

            reference_hash,

        "automatically_discovered_crystals":

            len(cpus),

        "persistent_environment_starts": 2,

        "external_jobs_per_trial": len(jobs),

        "total_completed_jobs": len(all_jobs),

        "persistent_experience_between_jobs": True,

        "experience_reset_between_trials": True,

        "external_arrival_schedule_identical": True,

        "central_tile_scheduler": False,

        "global_control_wakeup_events": True,

        "local_wakeup_events": False,

        "local_single_mailbox": True,

        "explicit_acknowledgements": True,

        "targeted_external_admission": True,

        "idle_timeout_disabled": True,

        "live_journals": True,

        "fault_recovery_tested": False,

        "shutdown_timeout_seconds":
            CFG["shutdown_timeout"],

        "periodic_worker_polling": False,

        "fixed_collective_size": False,

        "shared_synchronization": True,
        "local_work_ownership_tested": True,
        "global_owner_array_in_local_mode": False,
        "individual_receiver_locks": True,
        "independent_task_completion_locks": True,

        "composite_crystals_tested": False,

        "hardware_cache_counters_measured": False,

        "summary":

            json.loads(

                summary.to_json(orient="records")

            )

    }

    (OUT/"report.json").write_text(

        json.dumps(report, indent=4),

        encoding="utf-8"

    )

    print("\nFINAL SUMMARY", flush=True)

    print(

        summary.to_string(index=False),

        flush=True

    )

    print(

        "\nALL VALID:",

        report["all_valid"],

        flush=True

    )


if __name__ == "__main__":

    mp.freeze_support()

    main()
