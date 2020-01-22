#!/usr/bin/env bash

# We use --implict-namespaces here so that 'stempy.' prepends the
# module names. Otherwise, it will do things like 'import io', which
# imports the system 'io' library rather than 'stempy.io'.
# Extra arguments at the end are exclude patterns
sphinx-apidoc --implicit-namespaces -f -o source/ ../python/stempy ../python/stempy/pipeline

make html
