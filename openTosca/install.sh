#!/bin/bash

echo "Unpacking Archiv"
IFS=';' read -ra NAMES <<< "$DAs";
for i in "${NAMES[@]}"; do
  echo "KeyValue-Pair: "
  echo $i
  IFS=',' read -ra entry <<< "$i";
    echo "Key: "
    echo ${entry[0]}
    echo "Value: "
    echo ${entry[1]}

  # find the tar.gz
  if [[ "${entry[1]}" == *.tar.gz ]];
  then
    # unpack
	tar -xf $CSAR${entry[1]} -C /
  fi
done
echo "Finished Unpacking"
echo "Installing Dependencies"

cd QuantumClassifierDocker
pip install -r requirements.txt
cp src/* .

echo "Finished Installing Dependencies"
