# Ejecuta esto para ver ejemplos de hashes
import sqlite3
conn = sqlite3.connect('scryfall_db.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT name, phash FROM cards WHERE name LIKE '%Black Lotus%' OR name LIKE '%Island%' LIMIT 5")
results = cursor.fetchall()
for name, phash in results:
    print(f"{name}: {phash[:50]}...")
conn.close()
