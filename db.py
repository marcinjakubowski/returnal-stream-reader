
import sqlite3
import logging

class NoRunException(Exception):
    pass

class ReturnalDb():
    def __init__(self):
        # autocommit
        self.db = sqlite3.connect('returnal.db', isolation_level=None)
        self.cur = self.db.cursor()
        self.run = None
        self.logger = logging.getLogger('DB')

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS run
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME DEFAULT (DATETIME('now')),
                file_id TEXT
            )
        """)

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS progress
            (
                run_id INT REFERENCES run(id),
                time INT,
                phase INT,
                room INT,
                score INT,
                multi DECIMAL(5, 2),
                at DATETIME DEFAULT (DATETIME('now'))
            )
        """)
    
    def start_new_run(self, file_id):
        self.logger = logging.getLogger(file_id)
        self.logger.info("  DB  | Starting new run")
        self.cur.execute("INSERT INTO run (file_id) VALUES (?)", (file_id, ))
        self.run = self.cur.lastrowid
        return self.run

    def record(self, time, phase, room, score, multi):
        if self.run is None:
            raise NoRunException()

        self.logger.info(f"  DB  | Recording progress @{phase}/{room} => {score} @ {multi}")            
        self.cur.execute(
            "INSERT INTO progress (run_id, time, phase, room, score, multi) VALUES (?, ?, ?, ?, ?, ?)", 
            (self.run, int(time), phase, room, score, multi)
        )


    



