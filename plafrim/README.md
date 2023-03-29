# MEG analysis on Plafrim

In 5 steps:

## Step 1 - change data and report paths in Makefile

Edit the `Makefile` and change the variables `DATA_ROOT` and `REPORT_ROOT`. 

`DATA_ROOT` should point to a folder with the following organisation:

- `DATA_ROOT/meg`: contains all preprocessed MEG data (npy format).
- `DATA_ROOT/vectors`: contains all features (BERT, syntactic trees...)
  
`REPORT_ROOT` will be the output directory, where all results for all subjects will
be stored.

In addition to that, `DATA_ROOT/raw` may contain all raw MEG data (from MASC-MEG dataset):

```
raw/-
    |_ sub-01
    |_ sub-02
    |...
    |_ sub-27
```

I recommend you running preprocessing from raw data so we work on exactly the same data.

You sould use storage in `/beegfs/<YOUR-USER-ID>` for all data, so that processes can access it at runtime. You have a quota of 1To.

## Step 2 - Make sure that file names are correct

Here are the conventions:
- all features files in  `vectors/` should begin with the name of the feature, for instance `bert-residual_context_5_2` or `incontrege_set_0`.
- if features have a descriptive suffix (for instance `rh_5` for BERT right context 5) it should be separated from feature name by a dash `-`, like in `bert-residual_context_5_2` or `bert-base-lw-5` or `bert-lw-base-rh_5`. Everything inside the suffix must be separated with underscores `_`, like in `bert-lw-base-rh_5`, `bert-residual_context_5_2`,  `bert-base-lw-lag_5_2`.
- all concatenated features files should be named as `concat-<feat1_filename>+<feat2_filename>`, as in `concat-bert-base-lw-5+contrege_comp_set_0`.

In summary, all files should be of the form `feature-suffix_description.npy` or  `concat-feature1-suffix_desc+feature2-suffix_desc.npy`.

## Step 3 - Make sure that data and features are correctly formatted

MEG data should be of shape (4, n_words, 208, 81).

Feature data should be of shape:
- (4, n_words, n_layers, n_dim) for any features containing BERT embeddings (including concatenated features)
- (4, n_words, n_dim) for other features.

BERT features are identified by the `bert` keyword in their filename.

Default number of BERT layer is 12. It can be tuned in the `src/generate_job` script (`BERT_LAYERS` constant in introduction).

## Step 4 - Prepare your environment

Run the following command:

```bash
guix install python-numpy python-scikit-learn python-scipy python-pandas python-tqdm python-matplotlib
python3 --version  # this should be around 3.9
python3 -m pip install mne mne-bids
```

This will setup your environment.

## Step 5 - Run the jobs

If you want to filter the number of subjects/features, simply remove the MEG and features files you want to exclude from the `DATA_ROOT/meg` and `DATA_ROOT/vectors` directories. The script will run every combination of jobs possible from the files in that directories.

Use the following command to run jobs:


### Preprocessing

From `DATA_ROOT/raw` will preprocess and create npy files in `DATA_ROOT/meg`:

```bash
make preprocess
```

### Regression and significance test (same time)

From `DATA_ROOT/meg` and `DATA_ROOT/vectors` will run regression and permutation tests and output results in `REPORT_ROOT`:

```bash
make regress-and-test
```

### Regression and significance test (separately)

> :warning: if you do this on all subjects, features and layers, it will completely fill the memory


From `DATA_ROOT/meg` and `DATA_ROOT/vectors` will run regression or permutation tests and output results in `REPORT_ROOT`:

```bash
make regression
make sigtest
```

## Check the jobs

You can see that jobs are running from the command `squeue`. You can cancel everything with `scancel -u <YOUR-USER-ID>`. You can also look and outputs and errors from the files in the directories  `stdout/` and  `stderr/` (that you might need to create at project's root).

## Clean

If you canceled the jobs, want to run new ones or changed the data, first call the `make clean` command to cleanup `stdout/`,  `stderr/`, and `jobs/` before submitting new jobs.
