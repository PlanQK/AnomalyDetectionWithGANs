from flask import Flask
from flask import request
from program import run
from werkzeug.utils import secure_filename
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os

app = Flask(__name__)

@app.route('/start', methods=['POST'])
def start_optimization():
    #network = request.args['network']
    #config = request.args['config']
    #cli_params_dict = parse_cli_params(request.args.get('cli_params', ""))

    json_data = request.get_json()
    

    config = json_data['params']
    data = json_data['data']

    f = io.StringIO()
    with redirect_stderr(f):
        result = run(data=data, params=config)
    response = {
        'result':result.to_json(),
        'logs':f.getvalue()
    } 
    return json.dumps(response)



#port = os.environ['Port']
port = 8080

app.run(host='0.0.0.0', port=port)