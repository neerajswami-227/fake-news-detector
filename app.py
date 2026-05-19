"""
Flask Web Application – Bilingual, Both Models Loaded at Startup
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import re
import nltk
from scraper import NewsScraper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
from functools import wraps
from flask_cors import CORS

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-strong-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fake_news.db').replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
scraper = NewsScraper()

# ------------------------------- DATABASE MODELS ---------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    history = db.relationship('CheckHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class CheckHistory(db.Model):
    __tablename__ = 'check_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(500))
    news_text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    checked_at = db.Column(db.DateTime, default=datetime.utcnow)

class ReportedNews(db.Model):
    __tablename__ = 'reported_news'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    news_text = db.Column(db.Text, nullable=False)
    reason = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)

# ------------------------------- HELPER FUNCTIONS ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = {'said', 'says', 'say', 'told', 'according', 'also', 'would', 'could', 'may'}
stop_words.update(custom_stopwords)

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

# ------------------------------- LOAD MODELS AT STARTUP -------------------------------
import joblib
print("Loading English model...")
model_en = joblib.load('models/model.joblib')
vectorizer_en = joblib.load('models/vectorizer.joblib')
print("✅ English model loaded")

# Load Hindi model (if available)
try:
    model_hi = joblib.load('models/hindi_model.joblib')
    vectorizer_hi = joblib.load('models/hindi_vectorizer.joblib')
    print("✅ Hindi model loaded")
except Exception as e:
    print(f"⚠️ Hindi model not found: {e}. Hindi will fallback to translation.")
    model_hi = None
    vectorizer_hi = None

def predict_english(text):
    try:
        processed = preprocess_text(text)
        X = vectorizer_en.transform([processed])
        pred = model_en.predict(X)[0]
        prob = model_en.predict_proba(X)[0]
        result = 'FAKE' if pred == 1 else 'REAL'
        conf = max(prob) * 100
        return result, conf
    except Exception as e:
        print(f"English prediction error: {e}")
        fake_keywords = ['shocking', 'conspiracy', 'exposed', 'you won\'t believe']
        if any(kw in text.lower() for kw in fake_keywords):
            return 'FAKE', 75.0
        return 'REAL', 70.0

def predict_hindi(text):
    if model_hi is None or vectorizer_hi is None:
        # Fallback to translation
        from language_utils import translate_to_english
        translated = translate_to_english(text)
        return predict_english(translated)
    try:
        from hindi_preprocess import preprocess_hindi
        processed = preprocess_hindi(text)
        X = vectorizer_hi.transform([processed])
        pred = model_hi.predict(X)[0]
        prob = model_hi.predict_proba(X)[0]
        result = 'FAKE' if pred == 1 else 'REAL'
        conf = max(prob) * 100
        return result, conf
    except Exception as e:
        print(f"Hindi prediction error: {e}, falling back to translation")
        from language_utils import translate_to_english
        translated = translate_to_english(text)
        return predict_english(translated)

# ------------------------------- ROUTES ---------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')
        if len(text) < 20:
            return jsonify({'error': 'Minimum 20 characters required'}), 400

        if language == 'hi':
            result, conf = predict_hindi(text)
        elif language == 'en':
            result, conf = predict_english(text)
        else:  # auto-detect
            from language_utils import detect_language
            lang = detect_language(text)
            if lang == 'hi':
                result, conf = predict_hindi(text)
            else:
                result, conf = predict_english(text)

        # Save history (deduplication within 2 seconds)
        if current_user.is_authenticated:
            last = CheckHistory.query.filter_by(
                user_id=current_user.id,
                news_text=text[:500]
            ).order_by(CheckHistory.checked_at.desc()).first()
            if not (last and (datetime.utcnow() - last.checked_at).total_seconds() < 2):
                entry = CheckHistory(
                    user_id=current_user.id,
                    title=None,
                    news_text=text[:1000],
                    result=result,
                    confidence=conf
                )
                db.session.add(entry)
                db.session.commit()

        return jsonify({
            'result': result,
            'confidence': round(conf, 1),
            'word_count': len(text.split()),
            'reading_time': max(1, len(text.split()) // 200)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        language = data.get('language', 'auto')
        if not url:
            return jsonify({'error': 'URL required'}), 400

        scraped = scraper.scrape_article(url)
        if not scraped['success']:
            return jsonify({'error': scraped.get('error', 'Scraping failed')}), 400

        if language == 'hi':
            result, conf = predict_hindi(scraped['text'])
        elif language == 'en':
            result, conf = predict_english(scraped['text'])
        else:
            from language_utils import detect_language
            lang = detect_language(scraped['text'])
            if lang == 'hi':
                result, conf = predict_hindi(scraped['text'])
            else:
                result, conf = predict_english(scraped['text'])

        if current_user.is_authenticated:
            last = CheckHistory.query.filter_by(
                user_id=current_user.id,
                news_text=scraped['text'][:500]
            ).order_by(CheckHistory.checked_at.desc()).first()
            if not (last and (datetime.utcnow() - last.checked_at).total_seconds() < 2):
                entry = CheckHistory(
                    user_id=current_user.id,
                    title=scraped.get('title', '')[:200],
                    news_text=scraped['text'][:1000],
                    result=result,
                    confidence=conf
                )
                db.session.add(entry)
                db.session.commit()

        scraped['prediction'] = result
        scraped['confidence'] = round(conf, 1)
        scraped['word_count'] = len(scraped['text'].split())
        return jsonify(scraped)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== AUTHENTICATION ROUTES (unchanged) =====
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user, remember=request.form.get('remember', False))
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        if password != confirm:
            flash('Passwords do not match', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    history = CheckHistory.query.filter_by(user_id=current_user.id)\
        .order_by(CheckHistory.checked_at.desc()).limit(50).all()
    total = len(history)
    fake = sum(1 for h in history if h.result == 'FAKE')
    real = total - fake
    avg_conf = sum(h.confidence for h in history) / total if total else 0
    stats = {'total_checks': total, 'fake_count': fake, 'real_count': real, 'avg_confidence': avg_conf}
    return render_template('dashboard.html', history=history, stats=stats)

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Admin access required', 'danger')
        return redirect(url_for('index'))
    stats = {
        'total_users': User.query.count(),
        'total_checks': CheckHistory.query.count(),
        'total_reports': ReportedNews.query.count(),
        'pending_reports': ReportedNews.query.filter_by(status='pending').count()
    }
    recent = CheckHistory.query.order_by(CheckHistory.checked_at.desc()).limit(20).all()
    reports = ReportedNews.query.filter_by(status='pending').limit(20).all()
    return render_template('admin.html', stats=stats, recent_checks=recent, pending_reports=reports)

@app.route('/report', methods=['POST'])
@login_required
def report_news():
    try:
        data = request.get_json()
        report = ReportedNews(
            user_id=current_user.id,
            news_text=data.get('text', '')[:2000],
            reason=data.get('reason', '')[:500]
        )
        db.session.add(report)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'english_model_loaded': True,
        'hindi_model_loaded': model_hi is not None
    })

# ------------------------------- DATABASE INIT -------------------------------
with app.app_context():
    db.create_all()
    try:
        db.engine.execute('ALTER TABLE check_history ADD COLUMN title TEXT')
        print("✅ Added title column")
    except:
        pass
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin created (admin/admin123)")

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)