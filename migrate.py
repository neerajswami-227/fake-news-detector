# migrate.py
from app import app, db

with app.app_context():
    try:
        db.engine.execute('ALTER TABLE check_history ADD COLUMN title TEXT')
        print("✅ Title column added successfully")
    except Exception as e:
        if 'duplicate column name' in str(e).lower():
            print("⚠️ Title column already exists")
        else:
            print(f"❌ Error: {e}")