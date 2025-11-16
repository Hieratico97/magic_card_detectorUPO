import sqlite3
import json
import os

JSON_FILE_NAME = 'default-cards.json'  # El archivo que descargaste
DB_NAME = 'scryfall_db.sqlite'

# --- Columnas que nos interesan ---
# phash: El "perceptual hash" (huella digital) del arte.
# name: El nombre de la carta.
# set: El código de la edición (ej. 'lea', 'mkm').
# set_name: El nombre de la edición (ej. 'Limited Edition Alpha').
# illustration_id: Un ID único para cada ilustración (agrupa reimpresiones).
# cardmarket_id: El ID que usa Cardmarket (¡clave para la API!).

def create_database():
    # Borra la BBDD antigua si existe para empezar de cero
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Crear la tabla
    cursor.execute('''
    CREATE TABLE cards (
        scryfall_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        set_code TEXT NOT NULL,
        set_name TEXT NOT NULL,
        phash TEXT,
        illustration_id TEXT,
        cardmarket_id INTEGER
    )
    ''')

    # Crear un índice en el 'phash' para búsquedas súper rápidas
    # (Guardaremos los hashes como texto)
    cursor.execute('CREATE INDEX idx_phash ON cards (phash)')

    print(f"Base de datos '{DB_NAME}' creada. Empezando a poblar desde '{JSON_FILE_NAME}'...")

    # Cargar el archivo JSON
    with open(JSON_FILE_NAME, 'r', encoding='utf-8') as f:
        all_cards = json.load(f)

    total_cards = len(all_cards)
    print(f"Se encontraron {total_cards} cartas en el JSON. Insertando...")

    cards_to_insert = []
    
    for i, card in enumerate(all_cards):
        
        # Filtramos objetos que no son cartas (ej. 'vanguard_avatar')
        if card.get('layout') == 'vanguard_avatar':
            continue

        # Extraemos los datos. Usamos .get() para evitar errores si falta un campo
        data = (
            card.get('id'),
            card.get('name'),
            card.get('set'),
            card.get('set_name'),
            card.get('phash'),  # El phash pre-calculado por Scryfall
            card.get('illustration_id'),
            card.get('cardmarket_id')
        )

        # Solo añadimos cartas que tengan un phash (¡esencial!)
        if data[4]: # Si el campo 'phash' no está vacío
            cards_to_insert.append(data)

        if (i + 1) % 5000 == 0:
            print(f"  Procesadas {i + 1} / {total_cards} cartas...")

    # Insertar todos los datos en la base de datos de golpe (mucho más rápido)
    print(f"Insertando {len(cards_to_insert)} cartas con pHash en la BBDD...")
    cursor.executemany(
        'INSERT INTO cards (scryfall_id, name, set_code, set_name, phash, illustration_id, cardmarket_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
        cards_to_insert
    )

    # Guardar cambios y cerrar
    conn.commit()
    conn.close()

    print(f"¡Éxito! Base de datos '{DB_NAME}' creada con {len(cards_to_insert)} cartas.")

if __name__ == '__main__':
    create_database()