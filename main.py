from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, g
import plotly
import json
from deepnoiseapp import DeepNoiseApp
import numpy as np

app = Flask(__name__, static_url_path="/")
app.config['SECRET_KEY'] = "sdfklas0lk42j"
app.debug = True
deep_noise_app = DeepNoiseApp(app)

@dataclass
class User:
    id: int
    username: str
    password: str

users = [
	User(1, "admin", "123456"),
]

@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        user = [u for u in users if u.id == session['user_id']][0]
        g.user = user

def parseInterval(request):
    if request.form.get('QvInterval'):
        QvInterval = int(request.form.get('QvInterval'))
    else:
        QvInterval = 1
    if request.form.get('QvNum'):
        QvNum = int(request.form.get('QvNum'))
    else:
        QvNum = 1
    if request.form.get('DPInterval'):
        DPInterval = int(request.form.get('DPInterval'))
    else:
        DPInterval = 1
    if request.form.get('DPNum'):
        DPNum = int(request.form.get('DPNum'))
    else:
        DPNum = 1
    if request.form.get('RPMInterval'):
        RPMInterval = int(request.form.get('RPMInterval'))
    else:
        RPMInterval = 1
    if request.form.get('RPMNum'):
        RPMNum = int(request.form.get('RPMNum'))
    else:
        RPMNum = 1
    return QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Qv = int(request.form.get('Qv'))
        DP = int(request.form.get('DP'))
        RPM = int(request.form.get('RPM'))
        type = int(request.form.get('type'))
        method = request.form.get('mode')
        QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum = parseInterval(request)
        Qv = np.array(range(Qv, Qv+QvInterval*QvNum, QvInterval))
        DP = np.array(range(DP, DP+DPInterval*DPNum, DPInterval))
        RPM = np.array(range(RPM, RPM+RPMInterval*RPMNum, RPMInterval))
        data = deep_noise_app.inputArrange(Qv, DP, RPM)
        input = {'data': data, 'type': type, 'method': method}
        pred = deep_noise_app.predict(input)
        Qv_fig, DP_fig, RPM_fig = deep_noise_app.dBA_postprocess(Qv, DP, RPM, pred)
        graphs = [
            dict(
                data=Qv_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='Qv'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            ),
            dict(
                data=DP_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='DP'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            ),
            dict(
                data=RPM_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='RPM'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            )
        ]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        res = {
            'fig' : graphJSON
        }
        return jsonify(res)
    else:
        return render_template('layout/index.html')

@app.route('/3octave', methods=['POST'])
def octave_3():
    Qv = int(request.form.get('Qv'))
    DP = int(request.form.get('DP'))
    RPM = int(request.form.get('RPM'))
    type = int(request.form.get('type'))
    method = request.form.get('mode')
    QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum = parseInterval(request)
    Qv = np.array(range(Qv, Qv+QvInterval*QvNum, QvInterval))
    DP = np.array(range(DP, DP+DPInterval*DPNum, DPInterval))
    RPM = np.array(range(RPM, RPM+RPMInterval*RPMNum, RPMInterval))
    data = deep_noise_app.inputArrange(Qv, DP, RPM)
    input = {'data': data, 'type': type, 'method': method}
    pred_fft = deep_noise_app.predict_3octave(input)
    octave_fig = deep_noise_app.octave3_postprocess(Qv, DP, RPM, pred_fft)
    graphs = [
        dict(
            data=octave_fig,
            layout=dict(
                title='1/3 octave',
                xaxis=dict(
                    title=dict(
                        text='Hz'
                    )
                ),
                yaxis=dict(
                    title=dict(
                        text='dBA'
                    )
                )
            )
    )]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    res = {
            'fig' : graphJSON
    }
    return jsonify(res)

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 登录操作
        session.pop('user_id', None)
        username = request.form.get("username", None)
        password = request.form.get("password", None)
        user = [u for u in users if u.username==username]
        if len(user) > 0:
            user = user[0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        
    return render_template("login.html")

@app.route("/profile")
def profile():
    if not g.user:
        return redirect(url_for('login'))
    
    return render_template("profile.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
