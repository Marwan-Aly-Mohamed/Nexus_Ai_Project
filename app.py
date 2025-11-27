import sys
import os

# 1. Add current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
# Import both the main Engine and the Chatbot
from nexus_engine import engine, chatbot

app = Flask(__name__)

# ==========================================
# WEB ROUTES
# ==========================================

@app.route('/')
def index():
    """Renders the main dashboard."""
    return render_template('index.html')

# ==========================================
# API ROUTES
# ==========================================

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main Analysis Endpoint.
    Triggers DLinear (Price) and FinBERT (News) pipelines.
    """
    data = request.json
    ticker = data.get('ticker')
    company = data.get('company', '')
    ceo = data.get('ceo', '')
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    print(f"📡 Analysis Request received for: {ticker}")
    
    try:
        # Run the full analysis pipeline
        result = engine.get_signal(ticker, company, ceo)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Server Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    GenAI Chatbot Endpoint.
    Receives user query, retrieves specific CSV context, calls Gemini.
    """
    data = request.json
    ticker = data.get('ticker')
    query = data.get('query')
    
    # Validation
    if not ticker or not query:
        return jsonify({"error": "Missing ticker or query"}), 400
    
    print(f"💬 Chat Query for [{ticker}]: {query}")
    
    try:
        # Generate response using the Chatbot class
        response_text = chatbot.generate_response(ticker, query)
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        print(f"❌ Chat Server Error: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================
# SERVER ENTRY POINT
# ==========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING NEXUS AI SERVER v4.0")
    print("   - DLinear Price Model: Active")
    print("   - FinBERT Sentiment:   Active")
    print("   - Gemini Chatbot:      Active")
    print(f"🌍 Open your browser to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False)