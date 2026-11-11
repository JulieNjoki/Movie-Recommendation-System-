# Merge Guide: combining the branches into `main`

Branch names below (`cleaning`, `eda`, `modeling`, `evaluation`) are the ones from the group's plan. Adjust if yours
differ (`git branch -a` lists them). I have not seen your repository, so check the file locations in step 4 against
what you actually have.

## 1. Safety first

```bash
git fetch --all --prune
git checkout main
git pull origin main
git checkout -b integration          # work here; main stays untouched until you are happy
git status                           # should say "nothing to commit, working tree clean"
```

## 2. Merge the branches in pipeline order

```bash
git merge --no-ff origin/eda        -m "Merge EDA branch"
git merge --no-ff origin/cleaning   -m "Merge cleaning branch"
git merge --no-ff origin/modeling   -m "Merge modeling branch"
git merge --no-ff origin/evaluation -m "Merge evaluation branch"
```

`--no-ff` keeps a visible merge commit for each phase, so the history shows who contributed what.

## 3. If there is a conflict

Git stops and lists the files. The likely ones are `README.md`, `requirements.txt` and `.gitignore`, because every
branch may have created its own. Notebooks and CSVs rarely conflict because each branch has different files.

```bash
git status                           # "both added" / "both modified" files are the conflicts
# open each file, look for <<<<<<<  =======  >>>>>>> markers, keep what you need from both sides
git add <file>
git commit                           # completes the merge, then continue with the next branch
```

* `requirements.txt`: keep **all** lines from both sides, remove duplicates.
* `README.md` / `.gitignore`: take the version in this repository (they replace the branch versions anyway).
* Two copies of the same CSV with different contents: stop and ask who changed it. The cleaned files should be exactly
  the ones produced by the cleaning notebook.
* To abandon a bad merge and start that step again: `git merge --abort`.

## 4. Move files into the final layout

Check where each branch put its files (`git ls-files`), then use `git mv` so history follows the files. Example,
assuming everything is at the repo root; edit the left-hand paths to match yours:

```bash
mkdir -p data/raw data/processed notebooks docs src tests models

git mv ratings.csv movies.csv tags.csv links.csv data/raw/
git mv ratings_clean.csv movies_clean.csv tags_clean.csv links_clean.csv train.csv test.csv data/processed/

git mv 01_data_exploration.ipynb      notebooks/01_data_exploration.ipynb
git mv Data_Cleaning2.ipynb           notebooks/02_data_cleaning.ipynb
git mv phase3_model_development.ipynb notebooks/03_model_development.ipynb
git mv phase4_model_evaluation.ipynb  notebooks/04_model_evaluation.ipynb

git commit -m "Reorganise into data/, notebooks/ layout"
```

If a `processed/` folder already exists from the cleaning branch, `git mv processed/* data/processed/` instead.

## 5. Add the Phase 5 files

Copy in from the integrated package: `README.md`, `requirements.txt`, `.gitignore`, `app.py`, `cli.py`, `src/`, `tests/`,
`docs/`, `models/.gitkeep` and `notebooks/05_integration_and_demo.ipynb`. Then:

```bash
pip install -r requirements.txt
python -m pytest -q                  # expect: 12 passed
python cli.py --demo                 # expect: a top-5 list
git add -A
git commit -m "Add integration layer, interfaces, tests and documentation"
```

## 6. Push and merge to main

```bash
git push -u origin integration
```

Then open a pull request `integration` into `main` on GitHub/GitLab and have a teammate review it. Merging through the
pull request keeps a record of the review. If your team merges locally instead:

```bash
git checkout main
git merge --no-ff integration -m "Integrate all project phases"
git push origin main
```

## 7. Check after merging

```bash
git log --oneline --graph --decorate -15      # four phase merges visible
git ls-files | head -40                       # matches the layout in README.md
```

Large files: all CSVs together are about 15 MB, well inside normal Git limits. Do not commit `phase3_outputs/` or
`*.pkl` model files (they are in `.gitignore`).
