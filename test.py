from flask import Flask
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def hello():
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0')