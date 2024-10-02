#!/usr/bin/env bash
set -ev

pip install cibuildwheel

if [[ $RUNNER_OS == "Windows" ]]; then
    .github/scripts/install_eigen_windows.sh
elif [[ $RUNNER_OS == "macOS" ]]; then
    brew install eigen ninja
fi
