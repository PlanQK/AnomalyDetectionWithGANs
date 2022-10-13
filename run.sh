# #!/bin/bash

git_root=`git rev-parse --show-toplevel`
path_prefix="input2/"
path_to_json="test_classical"

file="${git_root}/${path_prefix}${path_to_json}.json"

echo "using ${file} as input"

cat $file | jq '.data' > $git_root/input/data.json
cat $file | jq '.params' > $git_root/input/params.json

docker rm -f qgan
docker run -it \
  -e BASE64_ENCODED=false \
  -v $git_root/input/data.json:/var/input/data/data.json \
  -v $git_root/input/params.json:/var/input/params/params.json \
  --name qgan planqk-service