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

# ============================================
# CONFIGURATION
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

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
    """Load the trained model and vectorizer"""
    global model, vectorizer
    model_path = 'models/model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded successfully!")
        return True
    else:
        print("❌ Model files not found. Please run train_model.py first")
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

    # Manual override
    if language == 'hi':
        # Option A: use Hindi model if available
        if hindi_model is not None:
            from hindi_preprocess import preprocess_hindi
            processed = preprocess_hindi(text)
            vec = hindi_vectorizer.transform([processed])
            pred = hindi_model.predict(vec)[0]
            prob = hindi_model.predict_proba(vec)[0]
            result = 'FAKE' if pred == 1 else 'REAL'
            confidence = max(prob) * 100
            print("🇮🇳 Using dedicated Hindi model")
            return result, confidence
        else:
            # Fallback: translate to English
            translated = translate_to_english(text)
            print("🔄 Translating Hindi to English (fallback)")
            return predict_news_english(translated)

    elif language == 'en':
        return predict_news_english(text)

    else:  # auto-detect
        lang = detect_language(text)
        print(f"🌐 Detected language: {lang}")
        if lang == 'hi':
            if hindi_model is not None:
                from hindi_preprocess import preprocess_hindi
                processed = preprocess_hindi(text)
                vec = hindi_vectorizer.transform([processed])
                pred = hindi_model.predict(vec)[0]
                prob = hindi_model.predict_proba(vec)[0]
                result = 'FAKE' if pred == 1 else 'REAL'
                confidence = max(prob) * 100
                print("🇮🇳 Using dedicated Hindi model")
                return result, confidence
            else:
                translated = translate_to_english(text)
                print("🔄 Translating Hindi to English (fallback)")
                return predict_news_english(translated)
        else:
            return predict_news_english(text)

def predict_news_english(text):
    """Original English prediction logic (unchanged)"""
    if model is None or vectorizer is None:
        return None, None
    
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    
    result = 'FAKE' if prediction == 1 else 'REAL'
    confidence = max(probability) * 100
    
    return result, confidence


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
    """API endpoint for news prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')   # ← NEW: get language from request

        # Debug prints
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
            return jsonify({'error': 'Model not loaded. Please contact administrator'}), 500
        
        # ✅ SAVE TO HISTORY - THIS IS THE CRITICAL PART
        if current_user.is_authenticated:
            print(f"💾 Saving history for user: {current_user.username} (ID: {current_user.id})")
            history_entry = CheckHistory(
                user_id=current_user.id,
                news_text=text[:1000],
                result=result,
                confidence=confidence
            )
            db.session.add(history_entry)
            db.session.commit()
            print("✅ History saved successfully!")
        else:
            print("❌ User not logged in - history NOT saved")
        
        print("=" * 50)
        
        return jsonify({
            'result': result,
            'confidence': round(confidence, 1),
            'message': f'This article is classified as {result} with {confidence:.1f}% confidence'
        })
        
    except Exception as e:
        print(f"💥 ERROR in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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
        
        # Scrape the article
        result = scraper.scrape_article(url)
        
        if not result['success']:
            return jsonify({'error': result['error']}), 400
        
        # Preprocess and predict
        # Use the multilingual prediction function
        prediction_result, confidence = predict_news(result['text'], language=language)
        result['prediction'] = prediction_result
        result['confidence'] = confidence

        print(f"🔮 Prediction: {result['prediction']}, Confidence: {result['confidence']}")
        
        # ✅ SAVE TO HISTORY (same as /predict)
        if current_user.is_authenticated:
            print(f"💾 Saving URL result for user: {current_user.username} (ID: {current_user.id})")
            history_entry = CheckHistory(
                user_id=current_user.id,
                news_text=result['text'][:1000],  # Save the scraped text
                result=result['prediction'],
                confidence=result['confidence']
            )
            db.session.add(history_entry)
            db.session.commit()
            print("✅ URL history saved successfully!")
        else:
            print("❌ User not logged in - history NOT saved")
        
        print("=" * 50)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"💥 ERROR in /scrape: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/test', methods=['GET'])      # ← ADD THIS FOR TESTING
def test():
        return jsonify({'status': 'ok', 'message': 'Server is running'})


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
    
    # Load ML model
    load_models()
    
    # Create database tables
    create_tables()
    
    print("\n📍 Server running at: http://127.0.0.1:5000")
    print("📍 Admin login: admin / admin123")
    print("\n" + "=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)