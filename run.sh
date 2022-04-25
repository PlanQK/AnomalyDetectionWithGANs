# #!/bin/bash

git_root=`git rev-parse --show-toplevel`
path_prefix="input/"
path_to_json="small_input_test"

file="${git_root}/${path_prefix}${path_to_json}.json"

echo "using ${file} as input"

data=`cat $file | jq '.data' | base64`
params=`cat $file | jq '.params' | base64`

echo "parameters: ${params}"
echo "data: ${data}"

docker rm -f qgan
docker run -it -e INPUT_DATA="${data}" -e INPUT_PARAMS="${params}" --name qgan planqk-service