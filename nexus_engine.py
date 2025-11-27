import os
import sys
import pandas as pd
import numpy as np
import torch
import yfinance as yf
import feedparser
import dateparser
import datetime
import re
import urllib.parse
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from darts import TimeSeries
from darts.models import DLinearModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from sklearn.preprocessing import RobustScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Stock_Decision_System", "models")

# 🛠️ PATH TO YOUR STABLE 3-FEATURE MODEL
DLINEAR_DIR = os.path.join(MODEL_PATH, "dlinear_nexus", "dlinear_model_2025_3D_Horizon.pt")
FINBERT_DIR = os.path.join(MODEL_PATH, "finbert_ultimate")
DATA_DIR = os.path.join(BASE_DIR, "data")
KEY_FILE = os.path.join(BASE_DIR, "gemini_key.txt")

# Ensure Data Directory Exists
os.makedirs(DATA_DIR, exist_ok=True)

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Constants (MATCHING YOUR TRAINING CODE EXACTLY)
LOOKBACK = 60
HORIZON = 3   # 3-Day Trajectory
MAX_LEN = 64

# ==============================================================================
# CLASS: NEXUS CHATBOT
# ==============================================================================
class NexusChatbot:
    def __init__(self):
        self.api_key = self._load_key()
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Using Flash for speed/reliability
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                print("✅ Gemini AI Connected")
            except Exception as e:
                print(f"⚠️ Gemini Configuration Error: {e}")

    def _load_key(self):
        try:
            if os.path.exists(KEY_FILE):
                with open(KEY_FILE, 'r') as f: return f.read().strip()
        except: return None
        return None

    def generate_response(self, ticker, user_query):
        if not self.model: return "Error: Gemini API is not configured."
        
        csv_path = os.path.join(DATA_DIR, f"{ticker.upper()}_news.csv")
        if not os.path.exists(csv_path): return "Please run analysis first."
        
        try:
            df = pd.read_csv(csv_path)
            df = df.sort_values(by='Final Weight Score', ascending=False)
            context_df = df[['Headline', 'Publisher', 'Final Weight Score', 'Bert Class', 'Snippet', 'Date']]
            context_data = context_df.head(40).to_string(index=False)
        except Exception as e: return f"Error reading data: {str(e)}"

        system_prompt = f"""
        ### ROLE
        Senior Equity Research Strategist for Nexus Capital.
        
        ### DATA (INTERNAL)
        {context_data}

        ### USER QUERY
        "{user_query}"

        ### INSTRUCTIONS
        Synthesize the news into a professional financial answer. 
        Highlight key drivers. Use bolding for emphasis (which renders as Gold).
        """
        try:
            chat = self.model.start_chat(history=[])
            response = chat.send_message(system_prompt)
            return response.text
        except Exception as e: return f"API Error: {e}"

chatbot = NexusChatbot()

# ==============================================================================
# CLASS: NEXUS ENGINE
# ==============================================================================
class NexusEngine:
    def __init__(self):
        print("\n" + "="*60)
        print("⚙️ INITIALIZING NEXUS ENGINE v5.4 (JSON SAFE + STABLE 3D)")
        print("="*60)
        self.load_models()

    def load_models(self):
        # Load DLinear
        if os.path.exists(DLINEAR_DIR):
            map_loc = None if torch.cuda.is_available() else torch.device('cpu')
            self.price_model = DLinearModel.load(DLINEAR_DIR, map_location=map_loc)
            self.price_model.trainer_params["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
            print("✅ DLinear Price Model Loaded")
        else:
            print(f"⚠️ DLinear Missing: {DLINEAR_DIR}")
            self.price_model = None

        # Load FinBERT
        if os.path.exists(os.path.join(FINBERT_DIR, "config.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_DIR)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_DIR).to(device)
            print("✅ FinBERT Loaded")
        else:
            print("⚠️ FinBERT Missing")
            self.sentiment_model = None

    # --- HELPERS ---
    def _clean_ceo_name(self, name): return name.split()[0] if name else ""
    def _clean_company_name_for_search(self, name): return name.split()[0] if name else ""
    
    def fetch_company_metadata(self, ticker):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return info.get('shortName', ticker), ""
        except: return ticker, ""

    # --- NEWS FETCHING ---
    def fetch_yahoo_news(self, ticker):
        try:
            rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(rss)
            return [{'source': 'Yahoo', 'headline': e.title, 'date': dateparser.parse(e.published) or datetime.datetime.now(), 'snippet': e.summary if 'summary' in e else ""} for e in feed.entries]
        except: return []

    def fetch_google_news_rss(self, query):
        try:
            q = urllib.parse.quote_plus(query)
            rss = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss)
            return [{'source': 'Google', 'headline': e.title, 'date': dateparser.parse(e.published) or datetime.datetime.now(), 'snippet': ""} for e in feed.entries]
        except: return []

    # --- SCORING ---
    def calculate_initial_relevance(self, row, ticker, company, ceo):
        text = (str(row['headline']) + " " + str(row['snippet'])).lower()
        score = 1
        if company and company.lower() in text: score = 3
        if ticker.lower() in text: score = 4
        return score

    def apply_aging(self, score, pub_date):
        if pub_date.tzinfo is None: pub_date = pub_date.replace(tzinfo=datetime.timezone.utc)
        delta = datetime.datetime.now(datetime.timezone.utc) - pub_date
        return score * np.exp(-0.1 * max(0, delta.total_seconds()/86400))

    def predict_text(self, text_list):
        self.sentiment_model.eval()
        all_preds, all_confs = [], []
        for i in range(0, len(text_list), 32):
            batch = text_list[i:i+32]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.nn.functional.softmax(self.sentiment_model(**inputs).logits, dim=-1).cpu().numpy()
                all_preds.extend(np.argmax(probs, axis=1))
                all_confs.extend(np.max(probs, axis=1))
        return np.array(all_preds), np.array(all_confs)

    # --- ANALYZE NEWS ---
    def analyze_news_pipeline(self, ticker, company, ceo):
        if not company: company, ceo = self.fetch_company_metadata(ticker)
        print(f"\n📰 NEWS ANALYSIS: [{ticker}]")
        
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(self.fetch_yahoo_news, ticker)
            f2 = ex.submit(self.fetch_google_news_rss, f"{company} {ticker} finance")
            raw = f1.result() + f2.result()

        if not raw: return {'decision': 'HOLD', 'strength': 0, 'stats': {'positive':0, 'neutral':0, 'negative':0}}

        df = pd.DataFrame(raw)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.drop_duplicates(subset=['headline'], keep='first', inplace=True)
        df.sort_values(by='date', ascending=False, inplace=True)

        df['final_score'] = df.apply(lambda x: self.apply_aging(self.calculate_initial_relevance(x, ticker, company, ceo)*25, x['date']), axis=1)
        
        print(f"   🧠 Sentiment Inference on {len(df)} items...")
        df['full'] = df['headline'].astype(str) + ". " + df['snippet'].astype(str)
        p_idx, p_conf = self.predict_text(df['full'].tolist())
        df['class_label'] = pd.Series(p_idx).map({0:'Negative', 1:'Neutral', 2:'Positive'})
        
        # Save CSV
        csv_path = os.path.join(DATA_DIR, f"{ticker.upper()}_news.csv")
        df.rename(columns={'headline':'Headline', 'date':'Date', 'source':'Publisher', 'final_score':'Final Weight Score', 'class_label':'Bert Class', 'snippet':'Snippet'}, inplace=True)
        df[['Headline', 'Date', 'Publisher', 'Final Weight Score', 'Bert Class', 'Snippet']].to_csv(csv_path, index=False)

        sums = df.groupby('Bert Class')['Final Weight Score'].sum()
        pos, neu, neg = sums.get('Positive', 0), sums.get('Neutral', 0), sums.get('Negative', 0)
        
        if neu >= 1.5 * (pos + neg): d, s = "HOLD", 0
        elif pos > neg: d, s = "BUY", 2
        else: d, s = "SELL", -2
        
        return {'decision': d, 'strength': s, 'stats': {'positive': float(pos), 'neutral': float(neu), 'negative': float(neg)}}

    # --- ANALYZE PRICES (STABLE 3-FEATURE) ---
    def analyze_prices(self, ticker):
        if not self.price_model: return None
        try:
            ticker = ticker.upper().strip()
            print(f"\n📉 PRICE ANALYSIS: [{ticker}]")
            
            # 1. Fetch Data (6mo)
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            
            if len(df) < 90:
                print(f"   ❌ Insufficient history ({len(df)} rows)")
                return None

            # 2. Features (MATCHING YOUR STABLE MODEL: 3 FEATURES)
            # Only 3 Features: LogReturns, LogVol, Volatility, RSI
            df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['LogVol'] = np.log1p(df['Volume'])
            df['Volatility'] = df['LogReturns'].rolling(20).std().fillna(0)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)

            df.dropna(inplace=True)
            current_price = float(df['Close'].iloc[-1])

            # 3. Prepare Input (3 Covariates ONLY)
            covariates = ['LogVol', 'Volatility', 'RSI']
            ts = TimeSeries.from_dataframe(df, value_cols=['LogReturns'] + covariates, fill_missing_dates=True, freq='B')
            ts = MissingValuesFiller().transform(ts).astype(np.float32)

            # Scale
            scaler_in = Scaler(RobustScaler())
            ts_scaled = scaler_in.fit_transform(ts)
            scaler_out = Scaler(RobustScaler())
            scaler_out.fit(ts['LogReturns'])

            # 4. Predict (Horizon 3)
            pred_scaled = self.price_model.predict(n=HORIZON, series=ts_scaled['LogReturns'], past_covariates=ts_scaled[covariates], verbose=False)
            
            # 5. Decode
            pred_inv = scaler_out.inverse_transform(pred_scaled)
            log_traj = np.sum(pred_inv.values())
            pred_pct = (np.exp(log_traj) - 1) * 100
            pred_price = current_price * (1 + (pred_pct / 100))

            # 6. Score (3D Thresholds)
            if pred_pct > 3.0: d, s = "STRONG BUY", 3
            elif pred_pct > 0.5: d, s = "BUY", 2
            elif pred_pct < -3.0: d, s = "STRONG SELL", -3
            elif pred_pct < -0.5: d, s = "SELL", -2
            else: d, s = "HOLD", 0

            print(f"   💹 Forecast: {pred_pct:+.2f}%")

            # 7. Chart Data (JSON SAFE FIXES)
            # Convert pandas/numpy types to native python floats for JSON serialization
            chart_dates = [d.strftime('%Y-%m-%d') for d in df.index[-30:]]
            chart_prices = [float(x) for x in df['Close'].tail(30).tolist()]
            
            step = (pred_price - current_price) / 3
            # Construct projection points ensuring they are floats
            proj_prices = [
                float(current_price + step), 
                float(current_price + (step*2)), 
                float(pred_price)
            ]
            
            last_date = df.index[-1]
            proj_dates = []
            for i in range(1, 4):
                proj_dates.append((last_date + pd.tseries.offsets.BusinessDay(i)).strftime('%Y-%m-%d'))

            return {
                "current_price": float(round(current_price, 2)),
                "predicted_price": float(round(pred_price, 2)),
                "pct_change": float(round(pred_pct, 2)),
                "decision": d,
                "strength": s,
                "chart_data": {
                    "history_dates": chart_dates,
                    "history_prices": chart_prices,
                    "proj_dates": proj_dates,
                    "proj_prices": proj_prices
                }
            }

        except Exception as e:
            print(f"   ❌ Price Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # --- MASTER SIGNAL ---
    def get_signal(self, ticker, company, ceo):
        print(f"\n{'='*60}\n  🎯 ANALYZING: {ticker}\n{'='*60}")
        price = self.analyze_prices(ticker)
        news = self.analyze_news_pipeline(ticker, company, ceo)
        
        if not price: return {"error": f"Price Analysis Failed for {ticker}"}

        # Scoring Logic
        score = price['strength'] + news['strength']
        if score >= 4: sig, emo = "STRONG BUY", "🚀"
        elif score >= 1: sig, emo = "BUY", "🟢"
        elif score <= -4: sig, emo = "STRONG SELL", "📉"
        elif score <= -1: sig, emo = "SELL", "🔴"
        else: sig, emo = "HOLD", "⚪"

        return {
            "ticker": ticker.upper(),
            "signal": sig, "emoji": emo, "score": int(score),
            "price_data": price, "news_data": news
        }

engine = NexusEngine()