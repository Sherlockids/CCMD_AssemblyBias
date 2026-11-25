Scope: explicitly selected paper calculation/plot notebooks, their local Python dependencies and compact result files directly loaded for plotting. No check/test notebooks, raw catalogues, random points, MCMC chains or audit runs. Notebook output stripping applies only to Git storage; working notebooks remain unchanged.
After cloning, configure:
git config --local filter.abcode-notebook.clean "python .git-tools/strip_notebook.py"
git config --local filter.abcode-notebook.smudge cat
git config --local filter.abcode-notebook.required true

Corrfunc is an external dependency; the nested code/Corrfunc repository is not included.
