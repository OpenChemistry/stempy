#!/usr/bin/env bash
set -ev

# Keep the tested builder version when upgrading the host Python.
python -m pip install cibuildwheel==2.23.4

if [[ $RUNNER_OS == "Windows" ]]; then
    .github/scripts/install_eigen_windows.sh
elif [[ $RUNNER_OS == "macOS" ]]; then
    brew install eigen ninja
fi
