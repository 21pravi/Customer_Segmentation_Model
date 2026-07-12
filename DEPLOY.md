# Deploying to `Customer_Segmentation_Model`

This zip is pre-configured for a repo named **`Customer_Segmentation_Model`**. Every URL in
`README.md` already points to that name — no editing required.

---

## What was wrong with the previous upload

1. The dashboard link pointed at a repo name (`telecom-segmentation`) that didn't match the
   actual repository (`Customer_Segmentation_Model`) — GitHub Pages URLs are built from the
   repo name, so the link 404'd. **Fixed in this zip:** both URLs now read
   `21pravi.github.io/Customer_Segmentation_Model/` and
   `github.com/21pravi/Customer_Segmentation_Model.git`.
2. The project was uploaded as loose files plus a zip, with no `src/`, no `docs/`, and
   `ci.yml` sitting at the repo root where GitHub Actions can't find it. **Fixed:** this
   archive extracts to the correct folder tree directly.

---

## Deploy

```bash
# delete everything in the existing repo first, then from this unzipped folder:
cd Customer_Segmentation_Model

git init
git branch -M main
git add .
git commit -m "Publish full project structure"
git remote add origin https://github.com/21pravi/Customer_Segmentation_Model.git
git push -u origin main --force
```

`--force` overwrites the broken contents currently on GitHub.

## Enable the live dashboard

**Settings → Pages**
- Source: Deploy from a branch
- Branch: `main`, folder **`/docs`**
- Save

Wait about a minute. Live at:
**`https://21pravi.github.io/Customer_Segmentation_Model/`**

---

## Before you push, run the checker

```bash
bash check_before_push.sh
```

It verifies `src/`, `docs/index.html`, and `.github/workflows/ci.yml` are all present and in
the right place, that no zip or private notes are staged, that the dashboard has real data
rather than the `__DATA__` placeholder, and — the check that matters most here — that the
Pages URL inside `README.md` matches whatever repo you're actually pushing to. Once you've set
`git remote add origin ...`, run it again; it should print all green.

---

## Do not commit `INTERVIEW_NOTES.md`

It isn't in this zip on purpose — it's your private interview prep and must stay off GitHub.
Keep your local copy from the earlier delivery.
