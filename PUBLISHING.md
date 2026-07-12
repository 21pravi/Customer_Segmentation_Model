# Publishing to GitHub

Everything below assumes the repository will live at
`https://github.com/21pravi/Customer_Segmentation_Model`. Change the name if you prefer
something else — but update `README.md` (the Pages link and the clone URL) to match.

---

## Before you push — read this

**`INTERVIEW_NOTES.md` must not be published.** It is already listed on the first line of
`.gitignore`, and this has been verified:

```bash
git check-ignore -v INTERVIEW_NOTES.md
# .gitignore:5:INTERVIEW_NOTES.md    INTERVIEW_NOTES.md
```

Keep the file locally. If you ever run `git add -f INTERVIEW_NOTES.md`, you defeat the
purpose of having written it.

**`data/`, `models/`, and the two full per-subscriber exports are also ignored.** They total
~140 MB and are fully regenerable with `make all`. Committing them would bloat the repository
history permanently — git keeps every version of every blob forever, and `git rm` does not
reclaim the space.

What *does* get committed: 36 files, ~511 KB. The aggregate CSVs, `metrics.json`, the two
figures and the rendered dashboard go in as evidence that the pipeline actually ran.

---

## 1. Create the repository

On GitHub, create a new **public** repository named `Customer_Segmentation_Model`.
Do **not** initialise it with a README, `.gitignore`, or licence — this repo already has all
three, and GitHub's versions would conflict on the first pull.

Suggested description:

> Telecom subscriber segmentation with K-Means and a Gaussian Mixture Model, a supervised
> churn classifier, and a randomised control-group design. Interactive dashboard included.

Suggested topics: `machine-learning` `clustering` `gaussian-mixture-model` `kmeans`
`churn-prediction` `customer-segmentation` `telecom` `scikit-learn` `data-science`

---

## 2. Push

```bash
cd telecom_segmentation

git init
git branch -M main
git add .
git status                     # confirm INTERVIEW_NOTES.md is absent
git commit -m "Telecom subscriber segmentation and churn pipeline"

git remote add origin https://github.com/21pravi/Customer_Segmentation_Model.git
git push -u origin main
```

Check `git status` before committing. If `INTERVIEW_NOTES.md`, `data/` or `models/` appear
in the list, stop and fix `.gitignore` before continuing.

---

## 3. Enable GitHub Pages for the dashboard

The dashboard is a single self-contained HTML file with no external dependencies, so Pages
serves it directly.

1. Repository → **Settings** → **Pages**
2. **Source:** Deploy from a branch
3. **Branch:** `main`, folder **`/docs`**
4. Save.

After a minute the dashboard is live at
`https://21pravi.github.io/Customer_Segmentation_Model/`, which is the link already in the README.

`docs/index.html` is a copy of `outputs/dashboard.html`. Regenerate both with:

```bash
make pages
```

`docs/.nojekyll` is present so GitHub serves the file as-is rather than running it through
Jekyll.

---

## 4. Confirm CI passes

`.github/workflows/ci.yml` runs on every push to `main`. It lints with `ruff` and runs the
18-test suite on Python 3.10 and 3.12. Both steps pass locally:

```bash
ruff check src/ *.py tests/     # All checks passed!
pytest tests/ -q                # 18 passed in 3.89s
```

The tests deliberately assert **invariants**, not exact numbers — that the AON filter removes
every short-tenure subscriber, that segment rank tracks revenue monotonically, that the
control group lands within 0.5% of 3% in every segment, that accuracy collapses to the
majority-class baseline at threshold 1.0. Asserting `silhouette == 0.4156` would fail on the
next scikit-learn point release; asserting that no negative revenue survives preprocessing
will not.

If the badge goes red, that is the suite doing its job.

---

## 5. Optional polish

**Pin the repository** on your GitHub profile (`github.com/21pravi`) so it appears above the
fold.

**Add a status badge** to the README once CI has run at least once:

```markdown
[![CI](https://github.com/21pravi/Customer_Segmentation_Model/actions/workflows/ci.yml/badge.svg)](https://github.com/21pravi/Customer_Segmentation_Model/actions/workflows/ci.yml)
```

**Link it from LinkedIn** (`linkedin.com/in/praviveek-ray`) with the dashboard URL rather
than the repo URL. Recruiters click a working dashboard; far fewer click a source tree.

**Tag a release** once you are happy with it:

```bash
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

---

## If you later swap in real data

Do **not** commit a real subscriber extract, and do not commit results derived from one
without clearing it with your employer. `data/` is gitignored, which protects you by default.

The synthetic-data notice at the top of the README and the banner on the dashboard are load-
bearing. If you point the pipeline at a real extract, `USE_SYNTHETIC=False` flips both
automatically — `build_dashboard_data.py` reads the flag and changes the banner text. Verify
it did before publishing anything.

---

## File manifest

| File | Purpose |
|---|---|
| `README.md` | Repository landing page |
| `METHODOLOGY.md` | Assumptions, decisions, limitations — the document to hand a client |
| `INTERVIEW_NOTES.md` | **Private.** Gitignored. |
| `LICENSE` | MIT |
| `.gitignore` | Excludes private notes, ~140 MB of regenerable artifacts |
| `.gitattributes` | Stops Linguist labelling the repo as HTML; normalises line endings |
| `pyproject.toml` | Project metadata, ruff and pytest configuration |
| `requirements.txt` | Pinned dependencies |
| `Makefile` | `make all`, `make pages`, `make clean` |
| `.github/workflows/ci.yml` | Lint + test on 3.10 and 3.12 |
| `tests/test_pipeline.py` | 18 invariant tests |
| `docs/index.html` | GitHub Pages entry point |
| `docs/.nojekyll` | Serve HTML as-is |
