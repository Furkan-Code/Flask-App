from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')

def start():
    return "The MBSA Server is running"
@app.route('/mbsa')

def mbsa():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)