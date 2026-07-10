#!/usr/bin/env bash
# Launch the classroom with a CONSISTENT numpy.
#
# Why PYTHONNOUSERSITE=1: on this machine the app runs under anaconda's Python,
# whose pyarrow/pandas are compiled for numpy 1.x — but ~/.local/ has numpy 2.x,
# which leaks in via user site-packages and makes pyarrow crash with
# "numpy.core.multiarray failed to import". Telling Python to ignore user
# site-packages makes anaconda use its own consistent numpy 1.26 and everything
# imports cleanly. (Alternative permanent fix: `pip install --user "numpy<2"`.)
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
exec streamlit run app.py "$@"
