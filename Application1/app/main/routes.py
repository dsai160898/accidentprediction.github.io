from flask import render_template, request, Blueprint, jsonify
from app.api_call_pred import api_call
import datetime
import json
import traceback

main = Blueprint('main', __name__)

@main.route("/")
@main.route("/home")
def home():
    return render_template('index.html')

@main.route("/interaction")
def interaction():
    return render_template('interaction.html')

@main.route("/abstract")
def abstract():
    return render_template('abstract.html')

@main.route("/map")
def map():
    return render_template('predictionmap1.html')


@main.route('/prediction', methods=['POST'])
def prediction():
    try:
        req_data = request.get_json()
        origin = req_data['origin']
        destination = req_data['destination']
        date_time = req_data['datetime']

        tm = datetime.datetime.strptime(date_time,'%Y/%m/%d %H:%M').strftime('%Y-%m-%dT%H:%M')

        out = api_call(origin, destination, tm)

        return json.dumps(out)

    except:

        return jsonify({'trace': traceback.format_exc()})