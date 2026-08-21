import sqlite3
import json
import datetime
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "attendance.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    # 1. classes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            class_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            subject TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 2. students
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            prn TEXT NOT NULL,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id),
            UNIQUE(class_id, prn)
        )
    """)
    # 3. attendance_sessions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            date DATE NOT NULL,
            photo_path TEXT,
            total_present INTEGER DEFAULT 0,
            total_absent INTEGER DEFAULT 0,
            total_unknown INTEGER DEFAULT 0,
            taken_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (class_id),
            UNIQUE(class_id, date)
        )
    """)
    # 4. attendance_records
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT CHECK( status IN ('P', 'A', 'U') ),
            confidence_score REAL,
            FOREIGN KEY (session_id) REFERENCES attendance_sessions (session_id),
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    """)
    
    # Try to migrate old DB if it's the 1-table format
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(students)")
        cols = [r['name'] for r in cur.fetchall()]
        if 'class_id' not in cols and 'embedding' in cols:
            print("Migrating old database...")
            # Create a default class
            cur.execute("INSERT INTO classes (class_name, subject, description) VALUES (?, ?, ?)", ("Legacy Class", "General", "Auto-created during migration"))
            class_id = cur.lastrowid
            
            # We need to recreate the table, so we rename the old one
            cur.execute("ALTER TABLE students RENAME TO old_students")
            init_db() # re-run to create new tables
            
            # Copy data
            cur.execute(f"INSERT INTO students (class_id, prn, name, embedding) SELECT {class_id}, prn, name, embedding FROM old_students")
            cur.execute("DROP TABLE old_students")
            conn.commit()
            print("Migration successful.")
    except Exception as e:
        pass # Not a migration scenario
        
    conn.commit()
    conn.close()

# --- Classes ---
def create_class(class_name, subject="", description=""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO classes (class_name, subject, description) VALUES (?, ?, ?)",
        (class_name, subject, description)
    )
    conn.commit()
    class_id = cur.lastrowid
    conn.close()
    return class_id

def get_all_classes():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM classes ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_class(class_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM classes WHERE class_id = ?", (class_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

# --- Students ---
def add_student(class_id, prn, name, embedding_np):
    conn = get_connection()
    embedding_json = json.dumps(embedding_np.tolist())
    conn.execute(
        "INSERT OR REPLACE INTO students (class_id, prn, name, embedding) VALUES (?, ?, ?, ?)",
        (class_id, prn, name, embedding_json)
    )
    conn.commit()
    conn.close()

def get_students_for_class(class_id):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students WHERE class_id = ?", (class_id,)).fetchall()
    conn.close()
    
    students = []
    for r in rows:
        d = dict(r)
        d['embedding'] = np.array(json.loads(d['embedding']), dtype=np.float32)
        students.append(d)
    return students

# --- Attendance ---
def save_attendance_session(class_id, date, present_list, absent_list, unknown_count):
    conn = get_connection()
    cur = conn.cursor()
    
    # 1. Upsert Session
    cur.execute("""
        INSERT INTO attendance_sessions (class_id, date, total_present, total_absent, total_unknown) 
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(class_id, date) DO UPDATE SET 
            total_present=excluded.total_present,
            total_absent=excluded.total_absent,
            total_unknown=excluded.total_unknown,
            taken_at=CURRENT_TIMESTAMP
    """, (class_id, date, len(present_list), len(absent_list), unknown_count))
    
    # Get session_id
    cur.execute("SELECT session_id FROM attendance_sessions WHERE class_id=? AND date=?", (class_id, date))
    session_id = cur.fetchone()['session_id']
    
    # 2. Delete old records for this session if we are re-taking
    cur.execute("DELETE FROM attendance_records WHERE session_id=?", (session_id,))
    
    # 3. Insert Present
    for p in present_list:
        cur.execute(
            "INSERT INTO attendance_records (session_id, student_id, status, confidence_score) VALUES (?, ?, 'P', ?)",
            (session_id, p['student_id'], p['score'])
        )
        
    # 4. Insert Absent
    for a in absent_list:
        cur.execute(
            "INSERT INTO attendance_records (session_id, student_id, status, confidence_score) VALUES (?, ?, 'A', NULL)",
            (session_id, a['student_id'])
        )
        
    conn.commit()
    conn.close()
    return session_id

def get_attendance_report(class_id, date):
    conn = get_connection()
    query = """
        SELECT s.prn, s.name, r.status, r.confidence_score
        FROM attendance_records r
        JOIN attendance_sessions sess ON r.session_id = sess.session_id
        JOIN students s ON r.student_id = s.student_id
        WHERE sess.class_id = ? AND sess.date = ?
        ORDER BY s.prn
    """
    rows = conn.execute(query, (class_id, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_session_dates(class_id):
    conn = get_connection()
    rows = conn.execute("SELECT date FROM attendance_sessions WHERE class_id = ? ORDER BY date DESC", (class_id,)).fetchall()
    conn.close()
    return [r['date'] for r in rows]

# Initialize DB on load
init_db()
