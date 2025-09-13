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
        CREATE TABLE IF NOT EXISTS activation_patching_results (
            timestamp TEXT,
            model_id TEXT,
            patching_setup TEXT,
            component_name TEXT,
            max_recovery REAL,
            mean_recovery REAL,
            min_recovery REAL,
            best_layer INTEGER,
            best_position INTEGER,
            recovery_matrix TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS steering_results (
            timestamp TEXT,
            model_id TEXT,
            layer INTEGER,
            train_problems_count INTEGER,
            validation_problems_count INTEGER,
            success_rate REAL,
            successful_corrections INTEGER,
            steering_vector TEXT,
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

def insert_activation_patching_result(results_dir, model_id, patching_setup, component_name, 
                                     recovery_matrix):
    """Insert activation patching results for a component."""
    import numpy as np
    
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    # Calculate statistics
    data_array = np.array(recovery_matrix)
    max_recovery = float(np.max(data_array))
    mean_recovery = float(np.mean(data_array))
    min_recovery = float(np.min(data_array))
    
    # Find best position
    best_pos = np.unravel_index(np.argmax(data_array), data_array.shape)
    best_layer = int(best_pos[0])
    best_position = int(best_pos[1])
    
    # Convert patching_setup to JSON-serializable format
    setup_for_json = {}
    for key, value in patching_setup.items():
        if hasattr(value, 'to_dict'):  # DataFrame or Series
            setup_for_json[key] = {
                'type': 'dataframe',
                'shape': getattr(value, 'shape', None),
                'columns': getattr(value, 'columns', None).tolist() if hasattr(value, 'columns') else None,
                'summary': f"{type(value).__name__} with {len(value)} rows" if hasattr(value, '__len__') else str(type(value))
            }
        else:
            setup_for_json[key] = value
    
    cursor.execute("""
        INSERT INTO activation_patching_results 
        (timestamp, model_id, patching_setup, component_name, max_recovery, 
         mean_recovery, min_recovery, best_layer, best_position, recovery_matrix)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        get_timestamp_in_rome(),
        model_id,
        json.dumps(setup_for_json),
        component_name,
        max_recovery,
        mean_recovery,
        min_recovery,
        best_layer,
        best_position,
        json.dumps(data_array.tolist())
    ))
    
    conn.commit()
    conn.close()

def insert_steering_result(results_dir, model_id, layer, train_problems_count, validation_problems_count,
                          success_rate, successful_corrections, steering_vector, validation_results):
    """Insert steering experiment results."""
    import numpy as np
    
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO steering_results 
        (timestamp, model_id, layer, train_problems_count, validation_problems_count,
         success_rate, successful_corrections, steering_vector, validation_results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        get_timestamp_in_rome(),
        model_id,
        layer,
        train_problems_count,
        validation_problems_count,
        success_rate,
        successful_corrections,
        json.dumps(steering_vector.tolist() if hasattr(steering_vector, 'tolist') else None),
        json.dumps(validation_results)
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

def get_activation_patching_results(results_dir):
    """Get all activation patching results as a list of dictionaries."""
    conn = get_db_connection(results_dir)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM activation_patching_results")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        result_dict = dict(zip(columns, row))
        # Parse JSON fields
        if result_dict['patching_setup']:
            result_dict['patching_setup'] = json.loads(result_dict['patching_setup'])
        if result_dict['recovery_matrix']:
            result_dict['recovery_matrix'] = json.loads(result_dict['recovery_matrix'])
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
        
        # Activation patching results
        cursor.execute("SELECT COUNT(*) FROM activation_patching_results")
        patching_count = cursor.fetchone()[0]
        
        # Steering results
        cursor.execute("SELECT COUNT(*) FROM steering_results")
        steering_count = cursor.fetchone()[0]
        
        print(f"📊 Database Summary:")
        print(f"   Total problems: {total}")
        print(f"   Baseline completed: {baseline_count} ({baseline_correct} correct)")
        if intervention_count > 0:
            print(f"   Interventions completed: {intervention_count} ({intervention_correct} corrected)")
        if patching_count > 0:
            print(f"   Activation patching components: {patching_count}")
            
            # Show best recovery for each component
            cursor.execute("SELECT component_name, max_recovery, best_layer, best_position FROM activation_patching_results ORDER BY max_recovery DESC")
            patching_results = cursor.fetchall()
            for component, max_rec, layer, pos in patching_results:
                print(f"     {component}: Max recovery {max_rec:.3f} at Layer {layer}, Position {pos}")
        
        if steering_count > 0:
            print(f"   Steering experiments: {steering_count}")
            
            # Show steering results
            cursor.execute("SELECT layer, success_rate, successful_corrections, validation_problems_count FROM steering_results ORDER BY success_rate DESC")
            steering_results = cursor.fetchall()
            for layer, success_rate, corrections, total_val in steering_results:
                print(f"     Layer {layer}: {success_rate:.1f}% success ({corrections}/{total_val})")
        
        conn.close()
        
    except Exception as e:
        print(f"Could not read database summary: {e}")