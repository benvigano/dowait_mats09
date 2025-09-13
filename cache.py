"""
Cache management for all experiment data using SQLite for robustness.
Handles generation cache, error detection cache, and generalization cache.
"""

import os
import sqlite3
import hashlib
import time
from datetime import datetime
import pytz

# --- Configuration ---
CACHE_DIR = "cache"
CACHE_DB_FILE = os.path.join(CACHE_DIR, "cache.db")

# --- Utility Functions ---

def get_timestamp_in_rome():
    """Returns the current timestamp in 'Europe/Rome' timezone."""
    rome_tz = pytz.timezone('Europe/Rome')
    return datetime.now(rome_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

def print_timestamped_message(message):
    """Prints a message with a Rome timestamp."""
    print(f"[{get_timestamp_in_rome()}] {message}")

def _init_cache_db():
    """Initialize the SQLite cache database with all required tables."""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    # Generation cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generation_cache (
            cache_key TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            generated_text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            model_id TEXT NOT NULL
        )
    ''')
    
    # Error detection cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS error_detection_cache (
            cache_key TEXT PRIMARY KEY,
            problem_text TEXT NOT NULL,
            incorrect_cot TEXT NOT NULL,
            error_sentence TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            api_model TEXT NOT NULL
        )
    ''')
    
    # Generalization cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generalization_cache (
            cache_key TEXT PRIMARY KEY,
            problem_text TEXT NOT NULL,
            multiplier REAL NOT NULL,
            steered_solution TEXT NOT NULL,
            steered_answer TEXT NOT NULL,
            is_corrected INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Generation Cache ---

def _get_cache_key(prompt, model_id):
    """Generate a deterministic cache key for any prompt."""
    # Use a more structured format: model_hash::prompt_hash
    model_hash = hashlib.md5(model_id.encode('utf-8')).hexdigest()[:8]
    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:16]
    return f"{model_hash}::{prompt_hash}"

def save_to_generation_cache(prompt, generated_text, model_id):
    """Save a generation result to the cache."""
    _init_cache_db()
    cache_key = _get_cache_key(prompt, model_id)
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO generation_cache 
            (cache_key, prompt, generated_text, timestamp, model_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (cache_key, prompt, generated_text, get_timestamp_in_rome(), model_id))
        
        conn.commit()
    except Exception as e:
        print_timestamped_message(f"Warning: Could not save to generation cache: {e}")
    finally:
        conn.close()

def get_from_generation_cache(prompt, model_id):
    """Retrieve a generation result from the cache."""
    _init_cache_db()
    cache_key = _get_cache_key(prompt, model_id)
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT generated_text FROM generation_cache WHERE cache_key = ?
        ''', (cache_key,))
        
        result = cursor.fetchone()
        if result:
            return result[0]
    except Exception as e:
        print_timestamped_message(f"Warning: Could not read from generation cache: {e}")
    finally:
        conn.close()
    
    return None

def clear_generation_cache():
    """Clear the entire generation cache."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM generation_cache')
        conn.commit()
        print_timestamped_message("Generation cache cleared.")
    except Exception as e:
        print_timestamped_message(f"Warning: Could not clear generation cache: {e}")
    finally:
        conn.close()

def get_generation_cache_stats():
    """Get statistics about the generation cache."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT COUNT(*) FROM generation_cache')
        count = cursor.fetchone()[0]
        
        cursor.execute('SELECT DISTINCT model_id FROM generation_cache')
        models = [row[0] for row in cursor.fetchall()]
        
        return {'count': count, 'models': models}
    except Exception as e:
        print_timestamped_message(f"Warning: Could not get generation cache stats: {e}")
        return {'count': 0, 'models': []}
    finally:
        conn.close()

# --- Error Detection Cache ---

def _get_error_cache_key(problem_text, incorrect_cot):
    """Generate a deterministic cache key for error detection."""
    key_string = f"error_detection::{problem_text}::{incorrect_cot}"
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def save_to_error_cache_by_key(cache_key, problem_text, incorrect_cot, error_sentence, api_model="claude-3-5-sonnet-20241022"):
    """Save an error detection result by cache key, ensuring all fields are present."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO error_detection_cache 
            (cache_key, problem_text, incorrect_cot, error_sentence, timestamp, api_model)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (cache_key, problem_text, incorrect_cot, error_sentence, get_timestamp_in_rome(), api_model))
        
        conn.commit()
    except Exception as e:
        print_timestamped_message(f"Warning: Could not save to error detection cache: {e}")
    finally:
        conn.close()

def get_from_error_cache_by_key(cache_key):
    """Retrieve an error detection result by cache key."""
    _init_cache_db()
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT error_sentence FROM error_detection_cache WHERE cache_key = ?
        ''', (cache_key,))
        
        result = cursor.fetchone()
        if result:
            return result[0]
    except Exception as e:
        print_timestamped_message(f"Warning: Could not read from error detection cache: {e}")
    finally:
        conn.close()
    
    return None

# --- Generalization Cache ---

def _get_generalization_cache_key(problem_text, multiplier):
    """Generate a deterministic cache key for a generalization test run."""
    key_string = f"generalization::{problem_text}::multiplier_{multiplier}"
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

def save_to_generalization_cache(problem_text, multiplier, steered_solution, steered_answer, is_corrected):
    """Save a generalization test result to the cache."""
    _init_cache_db()
    cache_key = _get_generalization_cache_key(problem_text, multiplier)
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO generalization_cache 
            (cache_key, problem_text, multiplier, steered_solution, steered_answer, is_corrected, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (cache_key, problem_text, multiplier, steered_solution, steered_answer, int(is_corrected), get_timestamp_in_rome()))
        
        conn.commit()
    except Exception as e:
        print_timestamped_message(f"Warning: Could not save to generalization cache: {e}")
    finally:
        conn.close()

def get_from_generalization_cache(problem_text, multiplier):
    """Retrieve a generalization test result from the cache."""
    _init_cache_db()
    cache_key = _get_generalization_cache_key(problem_text, multiplier)
    
    conn = sqlite3.connect(CACHE_DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT problem_text, multiplier, steered_solution, steered_answer, is_corrected, timestamp
            FROM generalization_cache WHERE cache_key = ?
        ''', (cache_key,))
        
        result = cursor.fetchone()
        if result:
            return {
                'problem_text': result[0],
                'multiplier': result[1],
                'steered_solution': result[2],
                'steered_answer': result[3],
                'is_corrected': bool(result[4]),
                'timestamp': result[5]
            }
    except Exception as e:
        print_timestamped_message(f"Warning: Could not read from generalization cache: {e}")
    finally:
        conn.close()
    
    return None

# Initialize cache database on import
try:
    _init_cache_db()
    stats = get_generation_cache_stats()
    if stats['count'] > 0:
        print_timestamped_message(f"SQLite cache initialized. Found {stats['count']} cached generations for models: {', '.join(stats['models'])}")
    else:
        print_timestamped_message("SQLite cache initialized with empty database.")
except Exception as e:
    print_timestamped_message(f"Warning: Could not initialize cache database: {e}")