from flask import Flask, render_template, request
from flask import redirect, url_for
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
#from lstm import build_model
import plotly
import plotly.offline as po
from get_graph import candle_stick

app = Flask(__name__)

@app.route('/dashboard/')
def dashboard(name=None):
    #chart = candle_stick(search, start)
    return render_template("dashboard.html", name=name)

@app.route('/login/')
def login(name=None):
    #chart = candle_stick(search, start)
    return render_template("login.html", name=name)

database = {'abc': '123', 'Prajyot': 'P12345', 'karthik': '416'}


@app.route('/form_login', methods=['POST', 'GET'])
def form_login():
    name1 = request.form['username']
    pwd = request.form['password']
    if name1 not in database:
        return render_template('login.html', info='Invalid User')
    else:
        if database[name1] != pwd:
            return render_template('login.html', info='Invalid Password')
        else:
            return render_template('home.html', name=name1)


@app.route('/', methods=["GET", "POST"])
def home(name=None):
    try:
        if request.method == 'POST':
            search = request.form['search']
            #start = request.form['start']
            result = candle_stick(search,datetime.date.today())
            if result == True:
                return redirect("http://127.0.0.1:8050/")
                # return redirect(url_for("/login.html/"))
            else:
                return render_template('login.html', sign='notfound')
        else:
            return render_template("login.html", name=name)
    except Exception as e:
        return render_template("login.html", sign=e)


@app.route('/about/')
def about(name=None):
    return render_template("about.html", name=name)

@app.route('/mon')
def mon(name=None):
    return redirect('http://localhost:8050',code=301)


#ticker_symbol = 'NFLX'
#today = datetime.date.today()

def full_plot(y_inv, ytest_inv, ypred_inv):
    plt.plot(np.arange(0, len(y_inv)), y_inv, 'g', label="history")
    plt.plot(np.arange(len(y_inv), len(y_inv) + len(ytest_inv)), ypred_inv, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

    return plt


if __name__ == "__main__":
    app.run(debug=True)
