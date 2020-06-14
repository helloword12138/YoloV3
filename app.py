from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import argparse
from people_flow import *
import cv2
import time

#导入数据库模块
import pymysql
from flask import Flask
from flask import render_template
#导入前台请求的request模块
from flask import request
import traceback
from flask import jsonify
from datetime import timedelta

db = pymysql.connect(host="localhost", user="root", passwd='root', db='test')
cursor = db.cursor()
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/sigin')
def login1():
    return render_template('login.html')

@app.route('/regist')
def regist():
    return render_template('register.html')


@app.route('/sucess')
def sucess():
    return render_template('seccesful.html')

# 获取注册请求及处理
@app.route('/regist', methods=['POST'])
def getRigistRequest():
    email = request.form.get('email')
    nickname = request.form.get('nickname')
    password = request.form.get('password')
    account = request.form.get('username')

    db = pymysql.connect(host="localhost", user="root", passwd='root', db='test')
    cursor = db.cursor()
    sql = "select email from PeopleCheck where email=" + "'" + email + "'";
    cursor.execute(sql)
    results = cursor.fetchall()
    if len(results) == 1:
        return jsonify({'code': 101})
    else:
        cursor = db.cursor()
        sql2 = "select email from PeopleCheck where nickname=" + "'" + nickname + "'";
        cursor.execute(sql2)
        results = cursor.fetchall()
        if len(results) == 1:
            return jsonify({'code': 100})
        else:
            print(account + nickname + password + email)
            cursor = db.cursor()
            sql_insert = """
                             insert into PeopleCheck(PHONEID,NICKNAME,PASWORD,EMAIL) values('%s','%s','%s','%s')
                                                           """
            cursor.execute(sql_insert % (account, nickname, password, email))
            db.commit()
            return jsonify({'code': 200})
    db.close()


# 获取登录参数及处理
@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    sql = """ select PHONEID, PASWORD from PeopleCheck where PHONEID='%s' and PASWORD='%s' """ % (username, password)
    cursor.execute(sql)
    results = cursor.fetchone()

    print(request)
    if results:
        return jsonify({'code': 200})
        db.close()
    else:
        return jsonify({'code': 100})
        db.close()

@app.route('/xxx', methods=['POST', 'GET'])
def picture():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = basepath + '/static/yuan/01.jpg'

        output_path = basepath + '/static/img/xx.jpg'

        f.save(upload_path)
        #ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        parser.add_argument(
            "--input", nargs='?', type=str, required=False, default=upload_path,
            help="Video input path"
        )
        parser.add_argument(
            "--output", nargs='?', type=str, default= output_path,
            help="[Optional] picture output path"
        )
        FLAGS = parser.parse_args()
        detect_img(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

        return redirect(url_for('picture'))

    if request.method == 'GET':
        return render_template('picture.html')


@app.route('/index', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = basepath + '/video/' + f.filename
        output_path = basepath + '/static/img/demo2.mp4'
        out = basepath + '/static/yuan/demo.mp4'
        f.save(out)
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        parser.add_argument(
            "--input", nargs='?', type=str, required=False, default=upload_path,
            help="Video input path"
        )
        parser.add_argument(
            "--output", nargs='?', type=str, default= output_path,
            help="[Optional] Video output path"
        )
        FLAGS = parser.parse_args()
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

        return redirect(url_for('video'))
    return render_template('video.html', val1=time.time())


@app.route('/stream', methods=['POST', 'GET'])
def video2():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    FLAGS = parser.parse_args()
    detect_video1(YOLO(**vars(FLAGS)))
    return render_template('seccesful.html')



if __name__ == '__main__':
    app.run(debug=True,port = 6666)