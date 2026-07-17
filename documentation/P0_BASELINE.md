# P0 legacy baseline

P0 freezes the legacy simulator behavior before EP-MoE runtime changes are
introduced. The automated golden suite covers three independent paths:

| Mode | Total cycles API | Layer cycles | Stall cycles | Bank allocation |
| --- | ---: | ---: | ---: | --- |
| Static Bank, no prefetch | 33 | 33 | 6 | `2:2:2` |
| Dynamic Bank, no prefetch | 27 | 27 | 0 | `1:1:4` |
| Original non-Bank memory | 35 | 27 | 0 | N/A |

The non-Bank API total is 35 because the original double-buffered memory model's
overall completion time includes memory scheduling beyond the compute model's 27
cycles. This difference is intentional and is part of the frozen baseline.

Integer cycle, stall and allocation fields are exact. Floating-point utilization
is compared to nine decimal places. All fixtures explicitly disable legacy
prefetch and EP-MoE. SRAM and DRAM read/write counts are also frozen for every
baseline.

Run the regression suite with:

```sh
python3 -m unittest discover -s tests -v
```

Generated reports belong under `outputs/` or a temporary directory and are not
source artifacts. Experiment scripts now resolve paths relative to the repository
and the Python batch runner writes a separate configuration snapshot for every
run instead of modifying `configs/MoE/baseline.cfg`.

The unused `read_buffer_old.py` and `write_buffer_old.py` copies were retained
during baseline work and removed after the EP runtime and legacy golden suites
passed final acceptance.
