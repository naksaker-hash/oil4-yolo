#!/bin/zsh
cd "$(dirname "$0")"
while pgrep -f "^.venv/bin/python 10_fetch_extra" >/dev/null; do sleep 60; done
.venv/bin/python 11_fetch_confounders.py burn > runs/fetch_burn.log 2>&1
.venv/bin/python 11_fetch_confounders.py shadow > runs/fetch_shadow.log 2>&1
echo done > runs/fetch2.DONE
