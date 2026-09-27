#!/bin/sh
set -eu
for f in project.yml RIFE60GhostGuard/ContentView.swift RIFE60GhostGuard/RIFEVideoProcessor.swift .github/workflows/build-unsigned-ipa.yml; do
  test -f "$f" || { echo "missing $f"; exit 1; }
done
echo "Project structure OK"
