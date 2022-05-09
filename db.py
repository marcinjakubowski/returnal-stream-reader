import sqlite3

class ReturnalDb():
    def __init__(self):
        # autocommit
        self.db = sqlite3.connect('returnal.db', isolation_level=None)
        self.cur = self.db.cursor()
        self.run = None

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS run
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at DATETIME DEFAULT (DATETIME('now'))
            )
        """)

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS progress
            (
                run_id INT REFERENCES run(id),
                phase INT,
                room INT,
                score INT,
                multi DECIMAL(5, 2),
                at DATETIME DEFAULT (DATETIME('now'))
            )
        """)
    
    def start_new_run(self):
        print("[DB] Starting new run")
        self.cur.execute("INSERT INTO run DEFAULT VALUES")
        self.run = self.cur.lastrowid
        return self.run

    def record(self, phase, room, score, multi):
        if self.run is None:
            self.start_new_run()

        print(f"[DB] Recording progress @{phase}/{room} => {score} @ {multi}")            
        self.cur.execute(
            "INSERT INTO progress (run_id, phase, room, score, multi) VALUES (?, ?, ?, ?, ?)", 
            (self.run, phase, room, score, multi)
        )


    



