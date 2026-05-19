import os, re, time, joblib, nltk
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from sqlalchemy import inspect, text
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------------------- Add missing column -------------------------------
def add_title_column():
    with app.app_context():
        insp = inspect(db.engine)
        if 'check_history' in insp.get_table_names():
            if 'title' not in [c['name'] for c in insp.get_columns('check_history')]:
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE check_history ADD COLUMN title TEXT'))
                    conn.commit()
                print("✅ Added title column")

# ------------------------------- NLP -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update({'said', 'says', 'say', 'told', 'according', 'also', 'would', 'could', 'may'})

def preprocess_english(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ------------------------------- Load Logistic Regression Model (REQUIRED) -------------------------------
print("="*50)
print("LOADING LOGISTIC REGRESSION MODEL")
print("="*50)
model_path = 'models/model.joblib'
vec_path = 'models/vectorizer.joblib'
if not os.path.exists(model_path):
    # Try .pkl fallback
    model_path = 'models/model.pkl'
    vec_path = 'models/vectorizer.pkl'
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Vectorizer loaded from {vec_path}")
except Exception as e:
    print(f"❌ FATAL: Could not load model: {e}")
    print("Please run train_model.py and ensure model files exist.")
    exit(1)

def predict_english(text):
    processed = preprocess_english(text)
    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    result = 'FAKE' if pred == 1 else 'REAL'
    conf = max(prob) * 100
    print(f"   Prediction: {result} with {conf:.1f}%")
    return result, conf

# Hindi model (optional – if missing, auto-translate)
try:
    hindi_model = joblib.load('models/hindi_model.joblib')
    hindi_vec = joblib.load('models/hindi_vectorizer.joblib')
    print("✅ Hindi model loaded")
    def predict_hindi(text):
        from hindi_preprocess import preprocess_hindi
        proc = preprocess_hindi(text)
        X = hindi_vec.transform([proc])
        pred = hindi_model.predict(X)[0]
        prob = hindi_model.predict_proba(X)[0]
        return ('FAKE' if pred == 1 else 'REAL'), max(prob)*100
except:
    print("⚠️ Hindi model missing – will use translation")
    from language_utils import translate_to_english
    def predict_hindi(text):
        en = translate_to_english(text)
        return predict_english(en)

def predict_auto(text):
    from language_utils import detect_language
    lang = detect_language(text)
    if lang == 'hi':
        return predict_hindi(text)
    else:
        return predict_english(text)

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
    ist = utc_dt + timedelta(hours=5, minutes=30)
    return ist.strftime('%b %d, %Y %I:%M %p')

# ------------------------------- Routes -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    language = data.get('language', 'auto')
    if len(text) < 20:
        return jsonify({'error': 'Minimum 20 characters'}), 400

    key = f"u{current_user.id if current_user.is_authenticated else 0}_t{hash(text[:200])}"
    if is_duplicate(key):
        return jsonify({'error': 'Duplicate'}), 429

    if language == 'hi':
        result, conf = predict_hindi(text)
    elif language == 'en':
        result, conf = predict_english(text)
    else:
        result, conf = predict_auto(text)

    if current_user.is_authenticated:
        # DB duplicate check (60s)
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
                confidence=conf
            )
            db.session.add(entry)
            db.session.commit()

    return jsonify({
        'result': result,
        'confidence': round(conf, 1),
        'word_count': len(text.split()),
        'reading_time': max(1, len(text.split())//200)
    })

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url', '').strip()
    language = data.get('language', 'auto')
    if not url:
        return jsonify({'error': 'URL required'}), 400

    key = f"u{current_user.id if current_user.is_authenticated else 0}_u{url}"
    if is_duplicate(key):
        return jsonify({'error': 'Duplicate'}), 429

    scraped = scraper.scrape_article(url)
    if not scraped.get('success'):
        return jsonify({'error': scraped.get('error', 'Scraping failed')}), 400

    if language == 'hi':
        result, conf = predict_hindi(scraped['text'])
    elif language == 'en':
        result, conf = predict_english(scraped['text'])
    else:
        result, conf = predict_auto(scraped['text'])

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
                confidence=conf
            )
            db.session.add(entry)
            db.session.commit()

    scraped['prediction'] = result
    scraped['confidence'] = round(conf, 1)
    scraped['word_count'] = len(scraped['text'].split())
    return jsonify(scraped)

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
    history = CheckHistory.query.filter_by(user_id=current_user.id).order_by(CheckHistory.checked_at.desc()).limit(50).all()
    total = len(history)
    fake = sum(1 for h in history if h.result == 'FAKE')
    real = total - fake
    avg_conf = sum(h.confidence for h in history) / total if total else 0
    stats = {'total_checks': total, 'fake_count': fake, 'real_count': real, 'avg_confidence': avg_conf}
    last_10 = history[:10][::-1]
    trend_labels = [h.checked_at.strftime('%b %d') for h in last_10]
    trend_data = [round(h.confidence, 1) for h in last_10]
    return render_template('dashboard.html', history=history, stats=stats, trend_labels=trend_labels, trend_data=trend_data)

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        flash('Admin only', 'danger')
        return redirect(url_for('index'))
    stats = {'total_users': User.query.count(), 'total_checks': CheckHistory.query.count(), 'total_reports': ReportedNews.query.count(), 'pending_reports': ReportedNews.query.filter_by(status='pending').count()}
    recent = CheckHistory.query.order_by(CheckHistory.checked_at.desc()).limit(20).all()
    reports = ReportedNews.query.filter_by(status='pending').limit(20).all()
    return render_template('admin.html', stats=stats, recent_checks=recent, pending_reports=reports)

@app.route('/report', methods=['POST'])
@login_required
def report_news():
    data = request.get_json()
    report = ReportedNews(user_id=current_user.id, news_text=data.get('text','')[:2000], reason=data.get('reason','')[:500])
    db.session.add(report)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/api/stats')
def api_stats():
    return jsonify({'english_model_loaded': True, 'total_predictions': CheckHistory.query.count()})

# ------------------------------- Initialization -------------------------------
with app.app_context():
    db.create_all()
    add_title_column()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin created")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)