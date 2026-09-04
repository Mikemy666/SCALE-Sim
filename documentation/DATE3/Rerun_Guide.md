# DATE3 rerun guide after the paper-control expansion

Exp1--Exp3 are unchanged. Exp4--Exp6 must be rerun because their public
comparison rows now distinguish Static-555, Static-Opt, mapping-only, fixed-PF,
and full PIVOT. Exp7 has not been run and can be deferred.

```bash
.venv/bin/python run_date3_experiments.py --exp exp4 --skip-details
.venv/bin/python run_date3_experiments.py --exp exp5 --skip-details
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details
```

All commands are hash-resumable. Reissuing the same command skips completed,
current variants. `--force` is not required after this change.

Exp6 may be split by variable:

```bash
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable expert_count
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable token_count
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable top_k
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable expert_parallel
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable routing_severity
.venv/bin/python run_date3_experiments.py --exp exp6 --skip-details --exp6-variable routing_seed
```

Run Exp7 later with:

```bash
.venv/bin/python run_date3_experiments.py --exp exp7 --skip-details
```

After a completed experiment, rebuild/execute the notebook and figures with:

```bash
.venv/bin/python scripts/DATE3/update_exp4_notebook.py
.venv/bin/python scripts/DATE3/execute_notebook_cells.py exp4

.venv/bin/python scripts/DATE3/update_exp5_notebook.py
.venv/bin/python scripts/DATE3/execute_notebook_cells.py exp5

.venv/bin/python scripts/DATE3/update_exp6_notebook.py
.venv/bin/python scripts/DATE3/execute_notebook_cells.py exp6

.venv/bin/python scripts/DATE3/update_exp7_notebook.py
.venv/bin/python scripts/DATE3/execute_notebook_cells.py exp7
```
