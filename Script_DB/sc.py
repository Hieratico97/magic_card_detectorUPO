import sqlite3
import imagehash

def verify_island_hashes():
    conn = sqlite3.connect('scryfall_db.sqlite')
    cursor = conn.cursor()
    
    # Buscar todas las Islas en la base de datos
    cursor.execute("""
        SELECT name, set_code, set_name, phash 
        FROM cards 
        WHERE name LIKE '%Island%' AND phash IS NOT NULL
        LIMIT 10
    """)
    
    islands = cursor.fetchall()
    print(f"Encontradas {len(islands)} variantes de Island:")
    
    for name, set_code, set_name, phash_str in islands:
        print(f"\n{name} | {set_code} | {set_name}")
        print(f"Hash: {phash_str[:50]}...")
        
        # Verificar que el hash se puede convertir correctamente
        try:
            hash_obj = imagehash.hex_to_hash(phash_str)
            print(f"✓ Hash válido - longitud: {len(phash_str)}")
        except Exception as e:
            print(f"✗ Error con hash: {e}")
    
    conn.close()

verify_island_hashes()