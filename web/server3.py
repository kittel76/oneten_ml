# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from flask import Flask
from flask import request

from cf_with_rbm import dbio

app = Flask(__name__)

@app.route('/api/relPrdList')
def relPrdList():


    dbio.g

    return "kkk"


if __name__ == '__main__':

    app.run("0.0.0.0", "8000")


