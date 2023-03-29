import argparse
import itertools

parser = argparse.ArgumentParser()

parser.add_argument("job", type=str)
parser.add_argument("--root", "-r", type=str)
parser.add_argument("--output", "-o", type=str)
parser.add_argument("--meg_data", type=str)
parser.add_argument("--feat_data", type=str)


BERT_LAYERS = 12
MAX_JOBS = 80
MAX_TIME = "40:00:00"

HEADER = "#!/usr/bin/env bash"

PYENV = """
sleep $SLURM_ARRAY_TASK_ID
set -e
"""

# module load language/python/3.9
# guix install python-numpy python-scipy python-scikit-learn python-tqdm

INTRO = """
\necho "====={0} information ===="
echo "Node List: " $SLURM_NODELIST
echo "jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "submit directory:" $SLURM_SUBMIT_DIR
echo "submit host:" $SLURM_SUBMIT_HOST
echo "In the directory:" $PWD
echo "As the user:" $USER

echo "Running task {0}-$SLURM_ARRAY_TASK_ID on $(/bin/hostname)"
"""


class Commands:
    commands: str
    header: str

    def __init__(self, script_output):
        self.script_output = script_output
        self.commands = ""
        self.header = ""

    def add_command(self, cmd):
        self.commands += "\n" + cmd
        return self.commands

    def finalize(self):
        return self.header + self.commands + "\n"

    def write(self):
        with open(self.script_output, "w+") as fp:
            fp.write(self.finalize())


class SBatchLauncherCommands(Commands):
    def __init__(self, script_output, *sbatch_commands):
        super().__init__(script_output)

        self.sbatch_commands = list(sbatch_commands)

    def add_sbatch(self, sbatch):
        self.sbatch_commands.append(sbatch)

    def finalize(self):
        delay = 1
        for i, cmd in enumerate(self.sbatch_commands):
            if i == 0:
                self.add_command("job=$(sbatch --parsable " + cmd.script_output + ")")
            else:
                self.add_command(
                    f"job=$(sbatch --parsable "
                    + cmd.script_output
                    + " --dependency=afterany:$job)"
                )
            self.add_command("echo Launching job: $job")
            
            delay += 1

        return self.header + self.commands + "\n"


class SBatchCommand(Commands):
    defaults = {
        "name": "meg",
        "node": "1",
        "proc": "1",
        "cpus": "1",
        "time": MAX_TIME,
        "array": "0-1",
        "stdout": "stdout/%A_%a.stdout",
        "stderr": "stderr/%A_%a.stderr",
    }

    sbatch_cmds = {
        "name": "#SBATCH -J {}",
        "proc": "#SBATCH -n {}",
        "node": "#SBATCH -N {}",
        "cpus": "#SBATCH -c {}",
        "time": "#SBATCH --time {}",
        "array": "#SBATCH --array {}",
        "stdout": "#SBATCH -o {}",
        "stderr": "#SBATCH -e {}",
    }

    def apply_header(self, **kwargs):
        header = ""

        if "name" in kwargs:
            job_name = kwargs["name"]
        else:
            job_name = self.defaults["name"]

        for name, c in self.sbatch_cmds.items():
            arg = kwargs.get(name, self.defaults[name])
            string = c.format(arg)
            header += f"\n{string}"

        self.header = HEADER + header + PYENV + INTRO.format(job_name)

        return self.header


def gen_preprocess(root, output):
    job_name = "preprocess-meg"

    sbatch = SBatchCommand("jobs/preprocess.sl")

    subjects = []
    with open("jobs/config/subject_list.txt", "r") as fp:
        for line in fp.readlines():
            subjects.append(line)

    n_subjects = len(subjects)

    sbatch.apply_header(
        name=job_name,
        cpus="8",
        stdout=f"stdout/{job_name}-%A_%a.stdout",
        stderr=f"stderr/{job_name}-%A_%a.stderr",
        array=f"0-{n_subjects - 1}",
    )

    find_sub_id = f"$(expr $SLURM_ARRAY_TASK_ID % {n_subjects} + 1)"

    sbatch.add_command(
        f'srun -l python3 src/preprocess.py "{root}" "{output}" -s {find_sub_id}'
    )
    sbatch.add_command(f'srun -l echo "Done !"')

    sbatch.write()

    return


def get_layers(features):
    all_features = []
    all_layers = []
    for feat in features:
        if "bert" in feat:
            all_features += [feat] * BERT_LAYERS
            all_layers += [str(i) for i in range(BERT_LAYERS)]
        else:
            all_features.append(feat)
            all_layers.append("-1")

    return all_features, all_layers


def batch_jobs(jobs):
    if len(jobs) <= MAX_JOBS:
        return [jobs]
    else:
        batchs = [jobs[i : i + MAX_JOBS] for i in range(0, len(jobs), MAX_JOBS)]
        return batchs


def gen_regression(output, meg_data, feat_data):
    subjects = []
    with open("jobs/config/subject_list.txt", "r") as fp:
        for line in fp.readlines():
            subjects.append(line.split("-")[-1].split("\n")[0])

    n_subjects = len(subjects)

    features = []
    with open("jobs/config/feature_list.txt", "r") as fp:
        for line in fp.readlines():
            features.append(line)

    features, layers = get_layers(features)
    n_features = len(features)

    jobs = list(itertools.product(range(n_subjects), range(n_features)))
    batchs = batch_jobs(jobs)

    main_script = SBatchLauncherCommands("jobs/regression-main.sh")

    for i, batch in enumerate(batchs):
        batch_subjects = [subjects[s] for s, _ in batch]
        batch_features = [features[f] for _, f in batch]
        batch_layers = [layers[f] for _, f in batch]

        job_name = f"regression-meg"

        sbatch = SBatchCommand(f"jobs/regression-batch{i}.sl")

        sbatch.apply_header(
            name=job_name,
            cpus="8",
            stdout=f"stdout/{job_name}-%A_%a.stdout",
            stderr=f"stderr/{job_name}-%A_%a.stderr",
            array=f"0-{len(batch) - 1}",
            time="01:00:00",
        )

        sbatch.add_command(f"subjects=({' '.join(batch_subjects)})")
        sbatch.add_command(f"features=({' '.join(batch_features)})")
        sbatch.add_command(f"layers=({' '.join(batch_layers)})")

        find_sub = "${subjects[$SLURM_ARRAY_TASK_ID]}"
        find_feat = "${features[$SLURM_ARRAY_TASK_ID]}"
        find_layer = "${layers[$SLURM_ARRAY_TASK_ID]}"

        sbatch.add_command(f"echo Sub: {find_sub} Feat: {find_feat}, Layer: {find_layer}")
        
        sbatch.add_command(
            f'python3 src/meg_encoding_ridge.py "{meg_data}" '
            + f'"{feat_data}" "{output}" -s {find_sub} ' 
            + f'-f {find_feat} -l {find_layer}'
        )
        sbatch.add_command(f'echo "Done !"')

        sbatch.write()

        main_script.add_sbatch(sbatch)

    main_script.write()

    return


def gen_sigtest():
    reports = []
    with open("jobs/config/reg_reports.txt", "r") as fp:
        for line in fp.readlines():
            reports.append(line)

    jobs = list(range(len(reports)))
    batchs = batch_jobs(jobs)

    main_script = SBatchLauncherCommands("jobs/sigtest-main.sh")

    for i, batch in enumerate(batchs):
        batch_reports = [reports[b] for b in batch]

        job_name = f"sigtest-meg"

        sbatch = SBatchCommand(f"jobs/sigtest-batch{i}.sl")

        sbatch.apply_header(
            name=job_name,
            cpus="8",
            stdout=f"stdout/{job_name}-%A_%a.stdout",
            stderr=f"stderr/{job_name}-%A_%a.stderr",
            array=f"0-{len(batch) - 1}",
        )

        sbatch.add_command(f"reports=({' '.join(batch_reports)})")

        find_report = "${reports[$SLURM_ARRAY_TASK_ID]}"

        sbatch.add_command(
            f"srun -l python3 src/r2_significance_test.py -r {find_report}"
        )
        sbatch.add_command(f'echo "Done !"')

        sbatch.write()

        main_script.add_sbatch(sbatch)

    main_script.write()

    return


def gen_regression_and_sigtest(output, meg_data, feat_data):
    subjects = []
    with open("jobs/config/subject_list.txt", "r") as fp:
        for line in fp.readlines():
            subjects.append(line.split("-")[-1].split("\n")[0])

    n_subjects = len(subjects)

    features = []
    with open("jobs/config/feature_list.txt", "r") as fp:
        for line in fp.readlines():
            features.append(line)

    features, layers = get_layers(features)
    n_features = len(features)

    jobs = list(itertools.product(range(n_subjects), range(n_features)))
    batchs = batch_jobs(jobs)

    main_script = SBatchLauncherCommands("jobs/regress-test-main.sh")

    for i, batch in enumerate(batchs):
        batch_subjects = [subjects[s] for s, _ in batch]
        batch_features = [features[f] for _, f in batch]
        batch_layers = [layers[f] for _, f in batch]

        job_name = f"regress-test-meg"

        sbatch = SBatchCommand(f"jobs/regress-test-batch{i}.sl")

        sbatch.apply_header(
            name=job_name,
            cpus="8",
            stdout=f"stdout/{job_name}-%A_%a.stdout",
            stderr=f"stderr/{job_name}-%A_%a.stderr",
            array=f"0-{len(batch) - 1}",
            time="30:00:00",
        )

        sbatch.add_command(f"subjects=({' '.join(batch_subjects)})")
        sbatch.add_command(f"features=({' '.join(batch_features)})")
        sbatch.add_command(f"layers=({' '.join(batch_layers)})")

        find_sub = "${subjects[$SLURM_ARRAY_TASK_ID]}"
        find_feat = "${features[$SLURM_ARRAY_TASK_ID]}"
        find_layer = "${layers[$SLURM_ARRAY_TASK_ID]}"

        sbatch.add_command(f"echo Sub: {find_sub} Feat: {find_feat}, Layer: {find_layer}")
        
        sbatch.add_command(
            f'out=$(python3 src/meg_encoding_ridge.py "{meg_data}" '
            + f'"{feat_data}" "{output}" -s {find_sub} ' 
            + f'-f {find_feat} -l {find_layer})'
        )
        sbatch.add_command(f'echo Regression: done!')
        
        sbatch.add_command(f"report=$(echo $out | tail -n 1); echo Saved at $report;")

        sbatch.add_command(
            f"srun -l python3 src/r2_significance_test.py -r $report"
        )

        sbatch.add_command(f"echo Significance: done!")
        
        sbatch.add_command(f"rm $report/y_pred*; rm $report/y_test*;")
        
        sbatch.write()

        main_script.add_sbatch(sbatch)

    main_script.write()

    return


if __name__ == "__main__":
    args = parser.parse_args()

    if args.job == "preprocess":
        gen_preprocess(args.root, args.output)
    elif args.job == "regression":
        gen_regression(args.output, args.meg_data, args.feat_data)
    elif args.job == "sigtest":
        gen_sigtest()
    elif args.job == "regress-test":
        gen_regression_and_sigtest(args.output, args.meg_data, args.feat_data)
