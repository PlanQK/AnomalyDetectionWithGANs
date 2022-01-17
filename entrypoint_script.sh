#!/bin/bash
if [ "$backend" == "rigetti" ]
  then
    qvm -S > /dev/null 2>&1 &
    quilc -S > /dev/null 2>&1 &
fi
python /quantum-anomaly/run_simple.py