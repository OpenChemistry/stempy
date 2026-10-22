#!/usr/bin/env bash
set -ev

python -m pip install cibuildwheel==4.2.1

if [[ $RUNNER_OS == "Windows" ]]; then
    .github/scripts/install_eigen_windows.sh
elif [[ $RUNNER_OS == "macOS" ]]; then
    brew install eigen ninja
fi
