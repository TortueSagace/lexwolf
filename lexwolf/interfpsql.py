import psycopg2

# Establish connection
conn = psycopg2.connect(
    dbname="chess4",
    user="postgres",
    password="pswd",
    host="localhost"
)

cur = conn.cursor()

cur.execute("REINDEX INDEX idx_chessgame_gin;")
cur.execute("SELECT getboard(g,15) FROM games limit 1;")
rows = cur.fetchall()
for row in rows:
    for i in  range(len(row)):
        print(row[i].strip('()').replace('"','').strip('()'))
cur.close()
conn.close()    
