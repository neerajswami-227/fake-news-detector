"""
PHASE 4: Flask Web Application for Fake News Detection
Complete working web interface with authentication, history, and admin panel
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pickle
import re
import nltk
from scraper import NewsScraper   # ← ADD THIS LINE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import json
from functools import wraps
from flask_cors import CORS 
from language_utils import detect_language, translate_to_english
from hindi_preprocess import preprocess_hindi   # only if you have Hindi preprocessing
from datetime import datetime, timedelta

import nltk

# Download required NLTK data (runs only once on server startup)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ============================================
# CONFIGURATION
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

from datetime import datetime, timedelta

@app.template_filter('to_ist')
def to_ist(utc_dt):
    """Convert UTC datetime to Indian Standard Time (UTC+5:30) and format nicely."""
    if utc_dt is None:
        return ''
    ist_dt = utc_dt + timedelta(hours=5, minutes=30)
    return ist_dt.strftime('%b %d, %Y %I:%M %p')

# Database configuration – supports both local SQLite and cloud PostgreSQL
import os
if os.environ.get('DATABASE_URL'):
    # On Render (or any cloud with DATABASE_URL env var): use PostgreSQL
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://', 1)
else:
    # On your local machine: use SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fake_news.db'
    
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)   # allow all origins for development

# Initialize scraper
scraper = NewsScraper()

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page'

# ============================================
# DATABASE MODELS
# ============================================

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    history = db.relationship('CheckHistory', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class CheckHistory(db.Model):
    """Stores user's news check history"""
    __tablename__ = 'check_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    news_text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    checked_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'news_text': self.news_text[:200] + '...' if len(self.news_text) > 200 else self.news_text,
            'result': self.result,
            'confidence': round(self.confidence, 1),
            'checked_at': self.checked_at.strftime('%Y-%m-%d %H:%M')
        }


class ReportedNews(db.Model):
    """Stores user-reported fake news for model retraining"""
    __tablename__ = 'reported_news'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    news_text = db.Column(db.Text, nullable=False)
    reason = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, added_to_training
    reported_at = db.Column(db.DateTime, default=datetime.utcnow)


# ============================================
# LOAD ML MODEL AND VECTORIZER
# ============================================

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Custom stopwords for news
custom_stopwords = {
    'said', 'says', 'say', 'told', 'according', 'also', 'would',
    'could', 'may', 'one', 'two', 'three', 'article', 'news', 'report'
}
stop_words.update(custom_stopwords)

# Load model and vectorizer
model = None
vectorizer = None

def load_models():
    """Load the trained model and vectorizer with robust error handling"""
    global model, vectorizer
    import os
    import traceback

    model_path = 'models/model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'

    # Debug: show current directory and list files in 'models/'
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of 'models/' folder: {os.listdir('models') if os.path.exists('models') else 'models folder not found'}")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return False
    if not os.path.exists(vectorizer_path):
        print(f"❌ Vectorizer file not found at {vectorizer_path}")
        return False

    # Try to load with pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded successfully using pickle!")
        return True
    except Exception as e:
        print(f"❌ Pickle loading failed: {e}")
        print(traceback.format_exc())
        # Fallback: try using joblib (if available)
        try:
            import joblib
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("✅ Model and vectorizer loaded successfully using joblib!")
            return True
        except Exception as e2:
            print(f"❌ Joblib loading also failed: {e2}")
            return False
    
    # Load Hindi model if available
hindi_model = None
hindi_vectorizer = None
if os.path.exists('models/hindi_model.pkl') and os.path.exists('models/hindi_vectorizer.pkl'):
    with open('models/hindi_model.pkl', 'rb') as f:
        hindi_model = pickle.load(f)
    with open('models/hindi_vectorizer.pkl', 'rb') as f:
        hindi_vectorizer = pickle.load(f)
    print("✅ Hindi model loaded")
else:
    print("⚠️ Hindi model not found; will fallback to translation")


def preprocess_text(text):
    """Preprocess text for prediction"""
    # Clean text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(lemmatized)

def predict_hindi(text):
    # Option 1: Use dedicated Hindi model if available
    # Option 2: Translate to English and use English model
    from language_utils import translate_to_english
    translated = translate_to_english(text)
    return predict_news_english(translated)

def predict_news(text, language='auto'):
    """Multilingual prediction – auto-detects or uses manual language."""
    from language_utils import detect_language, translate_to_english

    # Helper function to safely call English prediction
    def safe_predict_english(txt):
        try:
            return predict_news_english(txt)
        except Exception as e:
            print(f"⚠️ English prediction failed: {e}")
            return "REAL", 50.0   # fallback

    # Manual override
    if language == 'hi':
        if hindi_model is not None and hindi_vectorizer is not None:
            try:
                from hindi_preprocess import preprocess_hindi
                processed = preprocess_hindi(text)
                vec = hindi_vectorizer.transform([processed])
                pred = hindi_model.predict(vec)[0]
                prob = hindi_model.predict_proba(vec)[0]
                result = 'FAKE' if pred == 1 else 'REAL'
                confidence = max(prob) * 100
                print("🇮🇳 Using dedicated Hindi model")
                return result, confidence
            except Exception as e:
                print(f"⚠️ Hindi model error: {e}, falling back to translation")
        # Fallback: translate to English
        translated = translate_to_english(text)
        print("🔄 Translating Hindi to English (fallback)")
        return safe_predict_english(translated)

    elif language == 'en':
        return safe_predict_english(text)

    else:  # auto-detect
        lang = detect_language(text)
        print(f"🌐 Detected language: {lang}")
        if lang == 'hi':
            if hindi_model is not None and hindi_vectorizer is not None:
                try:
                    from hindi_preprocess import preprocess_hindi
                    processed = preprocess_hindi(text)
                    vec = hindi_vectorizer.transform([processed])
                    pred = hindi_model.predict(vec)[0]
                    prob = hindi_model.predict_proba(vec)[0]
                    result = 'FAKE' if pred == 1 else 'REAL'
                    confidence = max(prob) * 100
                    print("🇮🇳 Using dedicated Hindi model")
                    return result, confidence
                except Exception as e:
                    print(f"⚠️ Hindi model error: {e}, falling back to translation")
            translated = translate_to_english(text)
            print("🔄 Translating Hindi to English (fallback)")
            return safe_predict_english(translated)
        else:
            return safe_predict_english(text)

def predict_news_english(text):
    """Original English prediction with fallback"""
    if model is None or vectorizer is None:
        print("⚠️ English model not loaded – returning fallback prediction")
        return "REAL", 50.0
    try:
        processed = preprocess_text(text)
        vec = vectorizer.transform([processed])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        result = 'FAKE' if prediction == 1 else 'REAL'
        confidence = max(probability) * 100
        return result, confidence
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return "REAL", 50.0


# ============================================
# USER LOADER FOR FLASK-LOGIN
# ============================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================
# ADMIN REQUIRED DECORATOR
# ============================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Home page with detection interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for news prediction with robust history saving."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')

        print("=" * 50)
        print(f"📝 /predict called")
        print(f"   Text length: {len(text)}")
        print(f"   User authenticated: {current_user.is_authenticated}")
        if current_user.is_authenticated:
            print(f"   Username: {current_user.username}")

        if not text:
            return jsonify({'error': 'Please enter some text to analyze'}), 400
        if len(text) < 20:
            return jsonify({'error': 'Please enter at least 20 characters for accurate analysis'}), 400

        result, confidence = predict_news(text, language=language)
        print(f"🔮 Prediction: {result}, Confidence: {confidence}")

        if result is None:
            # Fallback when model not loaded
            return jsonify({
                'result': 'REAL',
                'confidence': 50.0,
                'message': 'Model temporarily unavailable. Using fallback.'
            }), 200

        # Save to history
        if current_user.is_authenticated:
            try:
                history_entry = CheckHistory(
                    user_id=current_user.id,
                    news_text=text[:1000],
                    result=result,
                    confidence=confidence
                )
                db.session.add(history_entry)
                db.session.commit()
                print("✅ History saved successfully!")
            except Exception as db_error:
                db.session.rollback()
                print(f"❌ Failed to save history: {db_error}")
        else:
            print("❌ User not logged in - history NOT saved")

        # Analytics for frontend
        word_count = len(text.split())
        reading_time = max(1, round(word_count / 200))

        print("=" * 50)

        return jsonify({
            'result': result,
            'confidence': round(confidence, 1),
            'word_count': word_count,
            'reading_time': reading_time,
            'message': f'This article is classified as {result} with {confidence:.1f}% confidence'
        })

    except Exception as e:
        db.session.rollback()
        print(f"💥 ERROR in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error. Please try again.'}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
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
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with check history"""
    # FIXED: correct ordering and .all()
    history = CheckHistory.query.filter_by(user_id=current_user.id)\
        .order_by(CheckHistory.checked_at.desc())\
        .limit(50)\
        .all()
    
    # Calculate stats
    total_checks = len(history)
    fake_count = sum(1 for h in history if h.result == 'FAKE')
    real_count = total_checks - fake_count
    avg_confidence = sum(h.confidence for h in history) / total_checks if total_checks > 0 else 0
    
    stats = {
        'total_checks': total_checks,
        'fake_count': fake_count,
        'real_count': real_count,
        'avg_confidence': avg_confidence
    }
    
    return render_template('dashboard.html', history=history, stats=stats)


@app.route('/report', methods=['POST'])
@login_required
def report_news():
    """Report fake news for model improvement"""
    try:
        data = request.get_json()
        news_text = data.get('text', '')
        reason = data.get('reason', '')
        
        report = ReportedNews(
            user_id=current_user.id,
            news_text=news_text[:2000],
            reason=reason[:500] if reason else None
        )
        db.session.add(report)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Thank you for reporting!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    """Admin dashboard"""
    stats = {
        'total_users': User.query.count(),
        'total_checks': CheckHistory.query.count(),
        'total_reports': ReportedNews.query.count(),
        'pending_reports': ReportedNews.query.filter_by(status='pending').count()
    }
    
    recent_checks = CheckHistory.query.order_by(CheckHistory.checked_at.desc()).limit(20).all()
    pending_reports = ReportedNews.query.filter_by(status='pending').limit(20).all()
    
    return render_template('admin.html', stats=stats, recent_checks=recent_checks, pending_reports=pending_reports)


@app.route('/admin/user/<int:user_id>/toggle_admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot change your own admin status'}), 400
    
    user.is_admin = not user.is_admin
    db.session.commit()
    return jsonify({'success': True, 'is_admin': user.is_admin})


@app.route('/api/stats')
def api_stats():
    """Public API endpoint for system stats"""
    return jsonify({
        'model_loaded': model is not None,
        'total_predictions': CheckHistory.query.count(),
        'model_accuracy': 93.7  # From training results
    })

@app.route('/scrape', methods=['POST'])
def scrape_url():
    """API endpoint to scrape news from URL and save to history"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        language = data.get('language', 'auto')

        if not url:
            return jsonify({'error': 'Please provide a URL'}), 400

        print("=" * 50)
        print(f"🌐 /scrape called")
        print(f"   URL: {url}")
        print(f"   User authenticated: {current_user.is_authenticated}")

        # 1. Scrape the article
        result = scraper.scrape_article(url)
        if not result['success']:
            return jsonify({'error': result['error']}), 400

        # 2. Predict (with fallback if model fails)
        prediction_result, confidence = predict_news(result['text'], language=language)
        if prediction_result is None:
            # Model not available – use neutral fallback
            prediction_result = "REAL"
            confidence = 50.0
            print("⚠️ Prediction fallback used (model unavailable)")

        result['prediction'] = prediction_result
        result['confidence'] = confidence
        print(f"🔮 Prediction: {prediction_result}, Confidence: {confidence:.1f}%")

        # 3. Save history (only if user is logged in AND prediction is not None)
        if current_user.is_authenticated and prediction_result is not None:
            try:
                # Optional: prevent duplicate saves for same URL within 1 second
                from datetime import datetime, timedelta
                last_check = CheckHistory.query.filter_by(
                    user_id=current_user.id,
                    news_text=result['text'][:1000]
                ).order_by(CheckHistory.checked_at.desc()).first()
                if last_check and (datetime.utcnow() - last_check.checked_at) < timedelta(seconds=1):
                    print("⏱️ Duplicate URL save ignored (same text within 1 sec)")
                else:
                    history_entry = CheckHistory(
                        user_id=current_user.id,
                        news_text=result['text'][:1000],  # original scraped text
                        result=prediction_result,
                        confidence=confidence
                    )
                    db.session.add(history_entry)
                    db.session.commit()
                    print("✅ URL history saved successfully!")
            except Exception as db_err:
                db.session.rollback()
                print(f"❌ Failed to save history: {db_err}")
        else:
            print("❌ User not logged in or invalid prediction - history NOT saved")

        # 4. Add analytics for frontend
        word_count = len(result['text'].split())
        reading_time = max(1, round(word_count / 200))
        result['word_count'] = word_count
        result['reading_time'] = reading_time

        print("=" * 50)
        return jsonify(result)

    except Exception as e:
        db.session.rollback()   # ensure any failed transaction is rolled back
        print(f"💥 ERROR in /scrape: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error. Please try again.'}), 500
    
@app.route('/test', methods=['GET'])      # ← ADD THIS FOR TESTING
def test():
        return jsonify({'status': 'ok', 'message': 'Server is running'})

load_models()


# ============================================
# CREATE DATABASE TABLES
# ============================================

def create_tables():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")
        
        # Create admin user if not exists
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin = User(
                username='admin',
                email='admin@fakenewsdetector.com',
                is_admin=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created (username: admin, password: admin123)")


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)
    
    
    # Create database tables
    create_tables()
    
    print("\n📍 Server running at: http://127.0.0.1:5000")
    print("📍 Admin login: admin / admin123")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)