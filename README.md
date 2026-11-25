## What we are building

We are exploring a way to organize computation in which many separate worker processes jointly execute tasks. Each process is treated as a computational element: it receives work, executes its share, and interacts with other participants. Within the project, these elements are called “crystals”: the term refers to a software model, not physical crystals.

The long-term idea is to study how such elements can form groups and distribute work using their accumulated experience. A future experiment will investigate hierarchical organization, in which a group acts as a single composite element to the outside while distributing work internally. This is a direction for future research, not a capability established by the current version.

## What the prototype does

Version KH-V-03.3b.2 compares four task distribution modes on one computer:

| Interaction | Simple splitting | Prediction-based |
| --- | --- | --- |
| Shared structure | GLOBAL HALF | GLOBAL PREDICT |
| Personal message channels | MAIL HALF | MAIL PREDICT |

In MAIL modes, processes exchange messages to transfer work and acknowledge receipt. The prototype records events, validates results, and measures task start latency, task completion time, and CPU time. Prediction had little effect on distribution in this experiment; its value for other workloads remains to be tested.

## Tested configurations

On Windows Server 2019, the experiment used two processor groups of 20 logical processors each, with five repetitions of each mode. On a separate Windows PC with one group and 16 logical processors, a shorter test ran one repetition per mode. Linux and macOS are not supported because the code uses the Windows API. Other configurations have not yet been tested.

## Running the experiment

Windows, Python 3.12, and the libraries listed in `requirements.txt` are required. Open PowerShell in the project directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiment.py --quick
```

Without `--quick`, the original full experiment configuration is used; it takes substantially longer and loads the available processors. Each run creates a new directory under `runs` without overwriting the original data. A successful run reports `ALL VALID: True` and exits with code 0. Also inspect `benchmark_raw.csv` and the diagnostic files in the run directory.

## Original results

The `results/` directory contains the server run tables. The summary reports median task start latency of 9.552 ms for GLOBAL HALF and 1.014 ms for MAIL HALF; median task completion times were 450.58 ms and 444.09 ms, respectively. These figures apply to one test workload and do not demonstrate a universal speedup. Several implementation details changed, so the difference cannot be attributed to a single mechanism. Absolute values may differ on other machines.

All 20 server trials passed the built-in checks. The prototype uses shared memory and locks; network communication and fault tolerance were not investigated. The results and checks need independent reproduction.

## Next questions

Future work includes independent reproduction across different workloads and configurations and comparisons with other suitable task distribution methods. Network communication, recovery from failures, and composite computational elements remain to be investigated. Compatibility with all hardware and system restrictions has not been verified.

## Project origins

The project grew out of the author’s KH research and uses some of its imagery as inspiration for a computational architecture. This experiment studies a software system and does not itself test the physical hypotheses of the KH concept.

## Author and AI contribution

The project idea, research questions, and direction belong to the author. The author is not a programmer and does not consider himself a vibe coder. The code was developed with the help of ChatGPT and Gemini, which also assisted with experiments and analysis. The project welcomes independent verification and feedback.

## Status

This is a preliminary research prototype; the API and result formats may change. Licensing and terms of use have not yet been determined by the author.


## Planned development

We plan to investigate coordinated task allocation across CPUs, GPUs and AI accelerators, including the cost of moving data and differences in device capabilities. Another research direction is a shared computing environment spanning multiple networked servers. A proposed performance memory would retain device- and workload-specific profiles and update them as conditions change. Potential AI-server workloads are a future area for evaluation.

These are research goals, not current capabilities. The published prototype is CPU-only and runs on one Windows machine; real accelerators, cross-server operation, fault recovery and performance-profile memory have not yet been implemented or validated. See the [research journal](RESEARCH_JOURNAL.md) for the current results, limitations and proposed tests.
