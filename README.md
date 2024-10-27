# Data Generation
Code to generate the data included in [FILE PATH].

```
from sc_causal.causal.util import create_in_distribution_h5ad, create_out_of_distribution_h5ad

fname_iid = 'out_iid_dataset' # Name of output directory
create_in_distribution_h5ad(fname_iid)

fname_ood = 'out_ood_dataset'
create_out_of_distribution_h5ad(fname_ood)
```

# Running Baselines

To run baselines, cd to the `sc_causal/GEARS/demo` directory and run

```
python run_baselines.py --id ood_split_id
```

where `ood_split_id` is a number, either [0, 1, 2, 3, 4, -1]. Setting `ood_split_id = -1` runs the in-distribution task.

# Training

To train SCCVAE:
```
cd sc_causal
python causal/run.py -x 'out_filename' -d 0 --ood --split-num 0
```

Flags:
- `x` Name of output directory.
- `-d` Which CUDA device to run the code.
- `--ood` If this flag is included, runs the OOD split given by `--split-num`. Otherwise, runs the IID task.
- `-m` Changes the graph in the SCM. Can be `full`, `causal`, `conditional`, or `random`.
- `--random-graph-seed` If training a `random` graph, keeps the generated graph consistent across all OOD splits.

# Evaluation
SCCVAE evaluation consists of two steps: Shift selection (for OOD tasks) and inference.

## Shift Select (OOD only)

In `sc_causal/causal/select_model_shift_values.py`, edit the list `SAVEDIRS` to be the list of all models to evaluate shift selection for, specifying the OOD split number, the model directory, and the sparse graph. Then, while in the `sc_causal` directory, run

```
python causal/select_model_shift_values.py
```

## Inference
```
python causal/inference.py -m 'out_filename' -s 'descrption' -n 'load_model_name' -d 0 --data-split 'test'  --ood-split 4

```

Flags
- `m` Model output directory.
- `s` Descriptive file name for saving output adata.
- `n` Model name to load from. E.g., `best_val_mmd_shiftselect_hard`
- `d` CUDA device.
- `--data-split` Evaluating on train, test, or val datasets.
- `--ood-split` If OOD, which OOD split is being evaluated.

# Figure replication
To reproduce the figures from the paper. Starting in the root directory,
```
cd sc_causal/visualize_paper_tables_and_figures
python tables.py
python fig2a.py
python fig2b.py
python fig3_fig7.py
python fig4_fig8.py
python fig5a.py
python fig5b.py
python fig6.py
python fig9.py
```