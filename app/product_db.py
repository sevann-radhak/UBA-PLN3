import json


def get_class_name(pred_idx, mapping_path):
    """
    Recupera el nombre de la clase a partir del índice predicho.

    Args:
        pred_idx (int): El índice numérico predicho por el modelo.
        mapping_path (str): La ruta al archivo JSON que contiene el mapeo de clases.

    Returns:
        str: El nombre de la clase correspondiente o un mensaje de error.
    """
    try:
        # Abrir y cargar el archivo de mapeo JSON
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
            
        # El archivo JSON usa claves de cadena, por lo que convertimos el índice
        # para hacer la búsqueda. Usamos .get() para evitar errores si el índice no existe.
        class_name = class_mapping.get(str(pred_idx), "Clase no encontrada")
        class_name = class_name.split('-')[-1]  # Obtener solo el nombre principal antes de la coma
        return class_name

    except FileNotFoundError:
        return f"Error: Archivo de mapeo '{mapping_path}' no encontrado."
    except json.JSONDecodeError:
        return f"Error: Archivo de mapeo '{mapping_path}' no es un JSON válido."