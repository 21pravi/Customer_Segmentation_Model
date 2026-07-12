#!/usr/bin/env bash
# check_before_push.sh
# Run this from the project root BEFORE `git push`. It catches the mistakes that
# broke the first upload: missing src/, a committed zip, private notes staged,
# wrong folder structure, or a dashboard/Pages URL that will 404.
set -u

fail=0
ok()   { printf "  \033[32mOK\033[0m   %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; fail=1; }
warn() { printf "  \033[33mWARN\033[0m %s\n" "$1"; }

echo "== structure =="
[ -d src ]                         && ok "src/ package present" || bad "src/ is MISSING — the whole pipeline lives here"
[ -f src/config.py ]               && ok "src/config.py present" || bad "src/config.py MISSING"
[ -f docs/index.html ]             && ok "docs/index.html present (Pages entry point)" || bad "docs/index.html MISSING — Pages will 404"
[ -f docs/.nojekyll ]              && ok "docs/.nojekyll present" || warn "docs/.nojekyll missing — add it so Pages serves HTML as-is"
[ -f .github/workflows/ci.yml ]    && ok ".github/workflows/ci.yml in correct path" || bad "ci.yml not at .github/workflows/ — Actions will not run it"
[ -f tests/test_pipeline.py ]      && ok "tests/ present" || warn "tests/ missing"

echo "== hygiene =="
if ls ./*.zip >/dev/null 2>&1; then bad "a .zip is in the repo root — commit the unzipped tree, not an archive"; else ok "no zip archives in root"; fi
[ -f .gitignore ]                  && ok ".gitignore present" || bad ".gitignore MISSING"
if [ -f INTERVIEW_NOTES.md ]; then
  if git check-ignore -q INTERVIEW_NOTES.md 2>/dev/null; then ok "INTERVIEW_NOTES.md is gitignored"; else bad "INTERVIEW_NOTES.md is NOT ignored — it will be published"; fi
fi

echo "== dashboard integrity =="
if [ -f docs/index.html ]; then
  if grep -q "__DATA__" docs/index.html; then bad "docs/index.html still has the __DATA__ placeholder — run: python build_dashboard_data.py && make pages"; else ok "docs/index.html has real data (no placeholder)"; fi
fi

echo "== Pages URL consistency =="
# The Pages URL is https://<user>.github.io/<REPO_NAME>/. It must match the repo
# you actually push to. Detect the repo name from the git remote if set.
remote="$(git remote get-url origin 2>/dev/null || echo '')"
if [ -n "$remote" ]; then
  repo="$(basename "${remote%.git}")"
  ok "git remote repo name: $repo"
  linked="$(grep -o 'github\.io/[A-Za-z0-9_.-]*' README.md | head -1 | cut -d/ -f2)"
  if [ -n "$linked" ]; then
    if [ "$linked" = "$repo" ]; then
      ok "README Pages link matches repo name ($repo)"
    else
      bad "README links github.io/$linked but repo is $repo — the dashboard link will 404. Fix one to match the other."
    fi
  fi
else
  warn "no git remote set yet — after 'git remote add origin ...', the Pages URL must be https://<user>.github.io/<that-repo-name>/"
fi

echo
if [ "$fail" -eq 0 ]; then
  printf "\033[32mAll critical checks passed. Safe to push.\033[0m\n"
else
  printf "\033[31mFix the FAIL items above before pushing.\033[0m\n"; exit 1
fi
