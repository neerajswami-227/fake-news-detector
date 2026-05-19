"""
Fake News Detection – Final (Auto‑fix missing column, works with any classifier)
"""

import re, os, pickle, joblib, nltk, time
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from scraper import NewsScraper
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fake_news.db').replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
scraper = NewsScraper()

# ------------------------------- Database Models -------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    history = db.relationship('CheckHistory', backref='user', lazy=True)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class CheckHistory(db.Model):
    __tablename__ = 'check_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(500))   # will be added automatically if missing
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------------- Helper: Add missing column -------------------------------
def add_column_if_not_exists(table, column, col_type):
    """Add a column to the table if it doesn't exist (SQLite + PostgreSQL)."""
    with app.app_context():
        try:
            # Check if column exists
            if 'sqlite' in str(db.engine.url):
                # SQLite pragma
                cursor = db.engine.execute(f"PRAGMA table_info({table})")
                cols = [row[1] for row in cursor]
                if column not in cols:
                    db.engine.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    print(f"✅ Added column '{column}' to {table}")
            else:
                # PostgreSQL
                cursor = db.engine.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name='{table}' AND column_name='{column}'
                """)
                if not cursor.fetchone():
                    db.engine.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                    print(f"✅ Added column '{column}' to {table}")
        except Exception as e:
            print(f"⚠️ Could not add column {column}: {e}")

# ------------------------------- NLP Helpers -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update({'said', 'says', 'say', 'told', 'according', 'also', 'would', 'could', 'may'})

def preprocess_english(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

# ------------------------------- Model Loading -------------------------------
def load_model_file(path):
    """Load .joblib first, then .pkl"""
    joblib_path = path.replace('.pkl', '.joblib')
    if os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    elif os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Model file not found: {path}")

print("Loading English model...")
try:
    model_en = load_model_file('models/model.pkl')
    vectorizer_en = load_model_file('models/vectorizer.pkl')
    print("✅ English model loaded")
except Exception as e:
    print(f"❌ CRITICAL: English model not loaded – {e}")
    exit(1)

print("Loading Hindi model (optional)...")
try:
    model_hi = load_model_file('models/hindi_model.pkl')
    vectorizer_hi = load_model_file('models/hindi_vectorizer.pkl')
    print("✅ Hindi model loaded")
except Exception:
    model_hi = None
    vectorizer_hi = None
    print("⚠️ Hindi model not found – will use translation fallback.")

def predict_english(text):
    """Works for both LogisticRegression (has predict_proba) and PassiveAggressive (has decision_function)."""
    processed = preprocess_english(text)
    X = vectorizer_en.transform([processed])
    pred = model_en.predict(X)[0]
    
    # Confidence extraction
    if hasattr(model_en, 'predict_proba'):
        prob = model_en.predict_proba(X)[0]
        confidence = max(prob) * 100
    elif hasattr(model_en, 'decision_function'):
        decision = model_en.decision_function(X)
        if decision.ndim == 1:
            decision = decision[0]
        else:
            decision = decision[0][1] if decision.shape[1] > 1 else decision[0]
        # Map decision value to [0,100] using sigmoid (rough)
        from math import exp
        sigmoid = 1 / (1 + exp(-decision))
        confidence = sigmoid * 100
    else:
        confidence = 75.0  # fallback
    
    result = 'FAKE' if pred == 1 else 'REAL'
    print(f"🔮 English prediction: {result} with {confidence:.1f}% confidence")
    return result, confidence

def predict_hindi(text):
    if model_hi is None or vectorizer_hi is None:
        from language_utils import translate_to_english
        translated = translate_to_english(text)
        return predict_english(translated)
    try:
        from hindi_preprocess import preprocess_hindi
        processed = preprocess_hindi(text)
        X = vectorizer_hi.transform([processed])
        pred = model_hi.predict(X)[0]
        if hasattr(model_hi, 'predict_proba'):
            prob = model_hi.predict_proba(X)[0]
            confidence = max(prob) * 100
        elif hasattr(model_hi, 'decision_function'):
            decision = model_hi.decision_function(X)
            if decision.ndim == 1:
                decision = decision[0]
            else:
                decision = decision[0][1] if decision.shape[1] > 1 else decision[0]
            from math import exp
            confidence = (1 / (1 + exp(-decision))) * 100
        else:
            confidence = 75.0
        result = 'FAKE' if pred == 1 else 'REAL'
        print(f"🔮 Hindi prediction: {result} with {confidence:.1f}% confidence")
        return result, confidence
    except Exception as e:
        print(f"Hindi model error: {e}, falling back to translation")
        from language_utils import translate_to_english
        translated = translate_to_english(text)
        return predict_english(translated)

# ------------------------------- Duplicate Prevention -------------------------------
_recent = defaultdict(float)

def is_duplicate(key, seconds=5):
    now = time.time()
    if key in _recent and now - _recent[key] < seconds:
        return True
    _recent[key] = now
    return False

# ------------------------------- Template filter -------------------------------
@app.template_filter('to_ist')
def to_ist_filter(utc_dt):
    if utc_dt is None:
        return ''
    ist_dt = utc_dt + timedelta(hours=5, minutes=30)
    return ist_dt.strftime('%b %d, %Y %I:%M %p')

# ------------------------------- Routes -------------------------------
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

        cache_key = f"u{current_user.id if current_user.is_authenticated else 0}_t{hash(text[:200])}"
        if is_duplicate(cache_key):
            return jsonify({'error': 'Duplicate request ignored'}), 429

        if language == 'hi':
            result, confidence = predict_hindi(text)
        elif language == 'en':
            result, confidence = predict_english(text)
        else:
            from language_utils import detect_language
            if detect_language(text) == 'hi':
                result, confidence = predict_hindi(text)
            else:
                result, confidence = predict_english(text)

        if current_user.is_authenticated:
            recent = CheckHistory.query.filter(
                CheckHistory.user_id == current_user.id,
                CheckHistory.news_text == text[:1000],
                CheckHistory.checked_at > datetime.utcnow() - timedelta(seconds=60)
            ).first()
            if not recent:
                entry = CheckHistory(
                    user_id=current_user.id,
                    title=None,
                    news_text=text[:1000],
                    result=result,
                    confidence=confidence
                )
                db.session.add(entry)
                db.session.commit()
                print("✅ History saved (predict)")
            else:
                print("⏱️ Duplicate prediction ignored (database)")

        return jsonify({
            'result': result,
            'confidence': round(confidence, 1),
            'word_count': len(text.split()),
            'reading_time': max(1, len(text.split()) // 200)
        })
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        language = data.get('language', 'auto')
        if not url:
            return jsonify({'error': 'URL required'}), 400

        cache_key = f"u{current_user.id if current_user.is_authenticated else 0}_u{url}"
        if is_duplicate(cache_key):
            return jsonify({'error': 'Duplicate request ignored'}), 429

        scraped = scraper.scrape_article(url)
        if not scraped['success']:
            return jsonify({'error': scraped.get('error', 'Scraping failed')}), 400

        if language == 'hi':
            result, confidence = predict_hindi(scraped['text'])
        elif language == 'en':
            result, confidence = predict_english(scraped['text'])
        else:
            from language_utils import detect_language
            if detect_language(scraped['text']) == 'hi':
                result, confidence = predict_hindi(scraped['text'])
            else:
                result, confidence = predict_english(scraped['text'])

        if current_user.is_authenticated:
            recent = CheckHistory.query.filter(
                CheckHistory.user_id == current_user.id,
                CheckHistory.news_text.like(f'%{url}%'),
                CheckHistory.checked_at > datetime.utcnow() - timedelta(seconds=60)
            ).first()
            if not recent:
                entry = CheckHistory(
                    user_id=current_user.id,
                    title=scraped.get('title', '')[:200],
                    news_text=scraped['text'][:1000],
                    result=result,
                    confidence=confidence
                )
                db.session.add(entry)
                db.session.commit()
                print("✅ History saved (scrape)")
            else:
                print("⏱️ Duplicate scrape ignored (database)")

        scraped['prediction'] = result
        scraped['confidence'] = round(confidence, 1)
        scraped['word_count'] = len(scraped['text'].split())
        return jsonify(scraped)
    except Exception as e:
        print(f"Scrape error: {e}")
        return jsonify({'error': 'Server error'}), 500

# ------------------------------- Authentication (keep your templates) -------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user, remember=request.form.get('remember', False))
            flash('Welcome back!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username/password', 'danger')
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
            flash('Username exists', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email registered', 'danger')
        elif len(password) < 6:
            flash('Password too short', 'danger')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful!', 'success')
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
    last_10 = history[:10][::-1]
    trend_labels = [h.checked_at.strftime('%b %d') for h in last_10]
    trend_data = [round(h.confidence, 1) for h in last_10]
    return render_template('dashboard.html', history=history, stats=stats,
                           trend_labels=trend_labels, trend_data=trend_data)

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Admin only', 'danger')
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
        'hindi_model_loaded': model_hi is not None,
        'total_predictions': CheckHistory.query.count()
    })

# ------------------------------- Database & column fix -------------------------------
with app.app_context():
    db.create_all()
    # Ensure 'title' column exists
    add_column_if_not_exists('check_history', 'title', 'TEXT')
    # Create admin user if missing
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created (admin/admin123)")

if __name__ == '__main__':
    print("\n🚀 Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)