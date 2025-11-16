import json
import os

def verify_json_file():
    json_path = 'Script_DB/default-cards.json'
    
    if not os.path.exists(json_path):
        print(f"❌ El archivo JSON no existe en: {json_path}")
        return False
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON válido. Número de cartas: {len(data)}")
        
        # Verificar algunas cartas de ejemplo
        print("\nEjemplos del JSON:")
        for i in range(min(3, len(data))):
            card = data[i]
            print(f"  {i+1}. {card.get('name', 'Sin nombre')} | {card.get('set', 'Sin set')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return False

verify_json_file()