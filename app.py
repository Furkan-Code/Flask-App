from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

model_path = "C:\\Users\\Furkan\\Desktop\\content\\outputs"
tokenizer_path = "C:\\Users\\Furkan\\Desktop\\content\\outputs"
# Model ve tokenizer yükleniyor
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


@app.route('/')

def start():
    return "The MBSA Server is running"
@app.route('/mbsa')

def mbsa():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])

def analyze():
    data = request.get_json()
    text = data['text']
    # Metni sınıflandırıp sonucu döndür
    result = nlp(text)
    # Sonucun içerisinde belirli bir afet türünü arayarak 'yes' veya 'no' döndür
    response = 'yes' if result[0]['label'] == 'LABEL_0' else 'no'
    return jsonify({'prediction': response})

if __name__ == '__main__':
    app.run(debug=True)