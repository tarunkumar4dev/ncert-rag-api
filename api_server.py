from flask import Flask, jsonify, request
from rag_system import query_rag

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "API Working"})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    answer = query_rag(data.get("question", ""))
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()
