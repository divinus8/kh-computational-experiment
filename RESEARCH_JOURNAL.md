# Research journal

## 2026-11-26 — CPU task distribution baseline

### Question and setup

Can personal message channels and predictive work sharing improve the handling of a stream of tasks on one computer? KH-V-03.3b.2 compares GLOBAL HALF, GLOBAL PREDICT, MAIL HALF and MAIL PREDICT. The modes differ in communication and work-sharing policy. The prototype uses Windows processor groups, shared memory and locks; its workers are software elements called crystals.

### Observations

On Windows Server 2019, with two processor groups of 20 logical processors, each of four modes was run five times. All 20 trials passed the built-in checks. In the published summary, median task-start latency was 9.552 ms for GLOBAL HALF and 1.014 ms for MAIL HALF; median task completion time was 450.58 ms and 444.09 ms, respectively. Prediction changed work distribution little in this workload. The tables are available in [results](results/).

A short test of the four modes also passed on a separate Windows PC with 16 logical processors in one group. This does not establish compatibility with all hardware or restricted environments. Linux and macOS are unsupported by this version.

### Interpretation and limitations

The lower start latency is an observation for this workload, not a ninefold reduction in total execution time or a general advantage over established schedulers. Several implementation details differ between modes, so the difference cannot be assigned solely to message channels. No independent reproduction, networked-server test, real accelerator test or fault-tolerance test is reported here.

### Research directions — proposed, not implemented

- Measure and remember device- and workload-specific performance profiles, compare cold starts with profile-assisted starts and adapt when conditions change.
- Study allocation among CPUs, GPUs and AI accelerators, including data transfer, memory limits, contention and numerical correctness.
- Investigate multiple networked servers working as a coordinated environment, including communication cost, failures and recovery.
- Compare against suitable existing schedulers using response latency, throughput, resource use and correctness over varied workloads.
- Test composite computational elements separately and measure the cost of hierarchy.

The project originates in the author’s KH research; these experiments concern software behavior and do not test the physical hypotheses of KH.
