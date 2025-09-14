"""
SQLite database for experiment results.
Single table with one row per problem containing complete baseline + intervention data.
"""

import sqlite3
import json
import os
from datetime import datetime
import pytz

def get_timestamp_in_rome():
    """Returns the current timestamp in 'Europe/Rome' timezone."""
    rome_tz = pytz.timezone('Europe/Rome')
    return datetime.now(rome_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

def create_database(db_path):
    """Create the SQLite database and tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            timestamp TEXT,
            experiment_parameters TEXT,
            problem TEXT,
            ground_truth_answer TEXT,
            baseline_raw_response TEXT,
            baseline_correct INTEGER,
            baseline_error_line_number INTEGER,
            baseline_error_line_content TEXT,
            intervention_raw_input_prompt TEXT,
            intervention_raw_response TEXT,
            intervention_correct INTEGER
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nnsight_steering_results (
            timestamp TEXT,
            model_id TEXT,
            layer INTEGER,
            steering_strength REAL,
            train_pairs_count INTEGER,
            validation_pairs_count INTEGER,
            success_rate REAL,
            successful_corrections INTEGER,
            steering_vector_norm REAL,
            validation_results TEXT
        )
    """)
    
    
    conn.commit()
    conn.close()

def get_db_connection(results_dir):
    """Get database connection for the experiment results directory."""
    db_path = os.path.join(results_dir, "experiment_results.db")
    
    # Create database if it doesn't exist
    if not os.path.exists(db_path):
        create_database(db_path)
    
    return sqlite3.connect(db_path)

def insert_baseline_result(results_dir, experiment_params, problem, ground_truth_answer, 
                          baseline_raw_response, baseline_correct, baseline_error_line_number, 
                          baseline_error_line_content):
    """Insert or update baseline results for a problem."""
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    # Check if this problem already exists
    cursor.execute("SELECT rowid FROM experiment_results WHERE problem = ?", (problem,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing row
        cursor.execute("""
            UPDATE experiment_results 
            SET timestamp = ?, experiment_parameters = ?, ground_truth_answer = ?,
                baseline_raw_response = ?, baseline_correct = ?, 
                baseline_error_line_number = ?, baseline_error_line_content = ?
            WHERE problem = ?
        """, (
            get_timestamp_in_rome(),
            json.dumps(experiment_params),
            ground_truth_answer,
            baseline_raw_response,
            baseline_correct,
            baseline_error_line_number,
            baseline_error_line_content,
            problem
        ))
    else:
        # Insert new row
        cursor.execute("""
            INSERT INTO experiment_results 
            (timestamp, experiment_parameters, problem, ground_truth_answer,
             baseline_raw_response, baseline_correct, baseline_error_line_number, 
             baseline_error_line_content, intervention_raw_input_prompt, 
             intervention_raw_response, intervention_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            get_timestamp_in_rome(),
            json.dumps(experiment_params),
            problem,
            ground_truth_answer,
            baseline_raw_response,
            baseline_correct,
            baseline_error_line_number,
            baseline_error_line_content,
            None,  # intervention_raw_input_prompt
            None,  # intervention_raw_response  
            None   # intervention_correct
        ))
    
    conn.commit()
    conn.close()

def insert_intervention_result(results_dir, problem, intervention_raw_input_prompt, 
                              intervention_raw_response, intervention_correct):
    """Update intervention results for an existing problem."""
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE experiment_results 
        SET intervention_raw_input_prompt = ?, intervention_raw_response = ?, 
            intervention_correct = ?, timestamp = ?
        WHERE problem = ?
    """, (
        intervention_raw_input_prompt,
        intervention_raw_response,
        intervention_correct,
        get_timestamp_in_rome(),
        problem
    ))
    
    conn.commit()
    conn.close()



def get_experiment_results(results_dir):
    """Get all experiment results as a list of dictionaries."""
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiment_results")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        result_dict = dict(zip(columns, row))
        # Parse experiment_parameters JSON
        if result_dict['experiment_parameters']:
            result_dict['experiment_parameters'] = json.loads(result_dict['experiment_parameters'])
        results.append(result_dict)
    
    conn.close()
    return results


def print_database_summary(results_dir):
    """Print a summary of the database contents."""
    try:
        conn = get_db_connection(results_dir)
        cursor = conn.cursor()
        
        # Total problems
        cursor.execute("SELECT COUNT(*) FROM experiment_results")
        total = cursor.fetchone()[0]
        
        # Baseline results
        cursor.execute("SELECT COUNT(*) FROM experiment_results WHERE baseline_raw_response IS NOT NULL")
        baseline_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM experiment_results WHERE baseline_correct = 1")
        baseline_correct = cursor.fetchone()[0]
        
        # Intervention results
        cursor.execute("SELECT COUNT(*) FROM experiment_results WHERE intervention_raw_response IS NOT NULL")
        intervention_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM experiment_results WHERE intervention_correct = 1")
        intervention_correct = cursor.fetchone()[0]
        
        print(f"📊 Database Summary:")
        print(f"   Total problems: {total}")
        print(f"   Baseline completed: {baseline_count} ({baseline_correct} correct)")
        if intervention_count > 0:
            print(f"   Interventions completed: {intervention_count} ({intervention_correct} corrected)")
            intervention_rate = (intervention_correct / intervention_count * 100) if intervention_count > 0 else 0
            print(f"   Intervention success rate: {intervention_rate:.1f}%")
        
        conn.close()
        
    except Exception as e:
        print(f"Could not read database summary: {e}")

def insert_nnsight_steering_result(results_dir, model_id, layer, steering_strength, train_pairs_count, 
                                  validation_pairs_count, success_rate, successful_corrections, 
                                  steering_vector_norm, validation_results):
    """Insert NNsight steering experiment results into database."""
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO nnsight_steering_results 
        (timestamp, model_id, layer, steering_strength, train_pairs_count, validation_pairs_count,
         success_rate, successful_corrections, steering_vector_norm, validation_results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        get_timestamp_in_rome(),
        model_id,
        layer,
        steering_strength,
        train_pairs_count,
        validation_pairs_count,
        success_rate,
        successful_corrections,
        steering_vector_norm,
        json.dumps(validation_results) if validation_results else None
    ))
    
    conn.commit()
    conn.close()