# DVC Guide for this project

DVC versions **data, models and pipelines** on top of git.
Git stores small text files (`dvc.yaml`, `dvc.lock`, `params.yaml`, `*.dvc`). DVC stores the big files (CSV, model) in a cache and in a remote (our S3 container).

```
params.yaml ──► create_dataset ──► train.csv, test.csv
                add_data ────────► extra.csv
                merge_data ──────► train_full.csv
                validate_data ───► (checks the data contract)
                train ───────────► model.joblib
                test ────────────► predictions.csv
                evaluate ────────► metrics.json
```

---

## 1. One-time setup

```bash
pip install -r requirements.txt     # includes dvc and dvc-s3

git init
dvc init                            # creates .dvc/ (commit it)

# Start the S3 container first (see DOCKERFILE)
docker build -f DOCKERFILE -t local-s3 .
docker run -d --name s3 -p 9090:9090 local-s3

# Add the remote
dvc remote add -d s3store s3://mybucket
dvc remote modify s3store endpointurl http://localhost:9090
dvc remote modify s3store region us-east-1
dvc remote modify --local s3store access_key_id test        # --local = not committed
dvc remote modify --local s3store secret_access_key test

dvc remote list                     # check
```

## 2. Daily workflow

```bash
dvc repro                 # run only the stages whose inputs changed
dvc push                  # upload data/model to the S3 remote
git add -A
git commit -m "describe the change"
git tag data-v3           # name the version
```

Rule of thumb: **`dvc repro` → `dvc push` → `git commit` → `git tag`**.

## 3. Running the pipeline

| Command | What it does |
|---|---|
| `dvc repro` | Run all stages that are out of date. Unchanged stages are skipped. |
| `dvc repro train` | Run `train` and the stages it depends on. |
| `dvc repro -s train` | Run only the `train` stage (single stage). |
| `dvc repro -f` | Force re-run of everything. |
| `dvc repro --dry` | Show what would run, without running it. |
| `dvc status` | Is the workspace in sync with `dvc.lock`? Shows what changed. |
| `dvc dag` | Print the stage graph. |
| `dvc dag --md` | Same graph as a Mermaid diagram. |
| `dvc exp run` | Run as an experiment (does not touch your git history). |

## 4. Changing the dataset

Edit [params.yaml](params.yaml), then `dvc repro`:

```yaml
dataset:
  n_samples: 1000     # rows in the base dataset
  n_features: 3
  noise: 1.0
  test_size: 0.2
  seed: 42

extra_data:
  n_samples: 500      # extra rows added to the training set
  seed: 7
```

You can also override params without editing the file, using experiments:

```bash
dvc exp run -S extra_data.n_samples=2000
dvc exp run -S dataset.noise=2.0
```

## 5. Seeing changes between versions

Versions are git tags (`data-v1`, `data-v2`, …). Any git revision works: tag, branch, commit hash, `HEAD~1`.

```bash
# Metrics
dvc metrics show                       # current numbers
dvc metrics diff data-v1 data-v2       # table: old, new, change
dvc metrics diff data-v1               # that version vs your workspace

# Params (settings that changed)
dvc params diff data-v1 data-v2

# Data / output files
dvc diff data-v1 data-v2               # added / modified / deleted files
dvc diff data-v1 data-v2 --targets data/train_full.csv
dvc diff data-v1 data-v2 --show-hash

# Git side
git log --oneline --decorate           # versions with their tags
git tag                                # list versions
git diff data-v1 data-v2 -- dvc.lock   # hashes of every data file per version
```

## 6. Going back to an old version

```bash
git checkout data-v1        # switch git files (dvc.yaml, dvc.lock, params.yaml)
dvc checkout                # restore the matching data files and model
# if files are missing locally:
dvc pull

# come back to the latest
git checkout master         # (or main)
dvc checkout
```

> Careful: after `git checkout data-v1` you are in "detached HEAD".
> Do not make new work there. Go back to `master` first.
> If `git checkout` complains about local changes, commit them or run `git stash` first.

To get just one file from an old version without switching:

```bash
git show data-v1:dvc.lock                          # see its hash
dvc get . data/train.csv --rev data-v1 -o old_train.csv
```

## 7. Remote storage (S3)

| Command | What it does |
|---|---|
| `dvc push` | Upload cached data/models to the remote. |
| `dvc pull` | Download whatever the current commit needs. |
| `dvc fetch` | Download to the cache only (no workspace change). |
| `dvc status -c` | Compare local cache with the remote. |
| `dvc gc -w` | Delete cache files not used by the current workspace. |
| `dvc gc -w -c` | Same, also on the remote. **Destructive, be careful.** |

Check the bucket directly:

```python
import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:9090",
                  aws_access_key_id="a", aws_secret_access_key="b", region_name="us-east-1")
print(s3.list_objects_v2(Bucket="mybucket")["KeyCount"])
```

> The S3 container (S3Mock) is **not persistent**. If you run `docker rm -f s3`,
> pushed data is gone and you must run `dvc push` again (the local cache `.dvc/cache` still has it).

## 8. Tracking a file by hand (not in the pipeline)

For a file you add yourself, such as a real CSV you downloaded:

```bash
dvc add data/my_real_data.csv      # creates data/my_real_data.csv.dvc and git-ignores the csv
git add data/my_real_data.csv.dvc data/.gitignore
git commit -m "track my_real_data"
dvc push
```

After editing the file, run `dvc add` again, then commit the updated `.dvc` file.

## 9. Experiments (try ideas without commits)

```bash
dvc exp run -S extra_data.n_samples=3000   # run with a changed param
dvc exp show                               # table of all experiments
dvc exp diff                               # compare to the last commit
dvc exp apply <exp-name>                   # keep an experiment in the workspace
dvc exp branch <exp-name> my-branch        # turn it into a git branch
dvc exp remove <exp-name>
```

## 10. Reading `dvc.yaml`

```yaml
stages:
  train:
    cmd: python -m pipeline.train          # command to run
    deps:                         # re-run when any of these change
      - pipeline/train.py
      - data/train_full.csv
    params:                       # re-run when these params change
      - dataset
    outs:                         # files DVC tracks and caches
      - model.joblib
    metrics:                      # small json shown by `dvc metrics`
      - metrics.json:
          cache: false            # keep it in git, not the DVC cache
```

- `dvc.lock` is generated. It stores the hash of every dep and out. **Commit it, never edit it.**
- Stage order comes from outputs feeding other stages' deps.
- To add a stage: edit `dvc.yaml`, or `dvc stage add -n name -d dep -o out python script.py`.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `you are not inside of a DVC repository` | Run `dvc init` (after `git init`) or `cd` into the project folder. |
| `no remote specified` | `dvc remote add -d s3store s3://mybucket` (and install `dvc-s3`). |
| `Unable to locate credentials` | Run the two `dvc remote modify --local …` commands from section 1. |
| Connection refused on `dvc push` | The S3 container is not running: `docker ps`, then `docker start s3`. |
| `data/xxx.csv` missing after `git checkout` | Run `dvc checkout` (or `dvc pull` if the cache is empty). |
| `failed to reproduce stage` | Read the stage output above the error; run the script directly to debug. |
| Pipeline re-runs everything | Something in a dep changed; use `dvc status` to see what. |
| Docker: `container name "/s3" is already in use` | It is already running, or `docker rm -f s3` then run again. |
| Detached HEAD warning in git | `git checkout master`, and don't commit while detached. |

## 12. Cheat sheet

```bash
dvc repro                          # run pipeline
dvc push / dvc pull                # sync with S3
dvc status                         # what is out of date
dvc dag                            # show stages
dvc metrics show                   # current metrics
dvc metrics diff data-v1 data-v2   # compare metrics
dvc params diff data-v1 data-v2    # compare settings
dvc diff data-v1 data-v2           # compare data files
git checkout data-v1 && dvc checkout   # restore an old version
dvc exp run -S key=value           # try a change as an experiment
```
