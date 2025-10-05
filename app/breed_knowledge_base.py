"""
Base de conocimiento estructurada para razas de perros
Implementa RAG híbrido según conceptos de Clase 1
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from app.documents import Document

@dataclass
class BreedInfo:
    """Información estructurada de una raza de perro"""
    breed_id: str
    breed_name: str
    characteristics: str
    temperament: str
    care_requirements: str
    health_issues: str
    training_advice: str
    origin: str
    size: str
    energy_level: str
    life_expectancy: str
    grooming_needs: str

class DogBreedKnowledgeBase:
    """Base de conocimiento para razas de perros"""
    
    def __init__(self, class_mapping_path: str = "./class_mapping.json"):
        self.class_mapping_path = class_mapping_path
        self.class_mapping = self.load_class_mapping()
        self.breed_database = self.create_breed_database()
    
    def load_class_mapping(self) -> Dict[str, str]:
        """Cargar mapeo de clases desde JSON"""
        with open(self.class_mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_breed_database(self) -> Dict[str, BreedInfo]:
        """Crear base de datos estructurada de razas"""
        breed_db = {}
        
        for breed_id, breed_name in self.class_mapping.items():
            # Extraer nombre limpio de la raza
            clean_name = breed_name.split('-')[-1].replace('_', ' ').title()
            
            # Crear información estructurada
            breed_info = BreedInfo(
                breed_id=breed_id,
                breed_name=clean_name,
                characteristics=self.get_breed_characteristics(clean_name),
                temperament=self.get_breed_temperament(clean_name),
                care_requirements=self.get_care_requirements(clean_name),
                health_issues=self.get_health_issues(clean_name),
                training_advice=self.get_training_advice(clean_name),
                origin=self.get_breed_origin(clean_name),
                size=self.get_breed_size(clean_name),
                energy_level=self.get_energy_level(clean_name),
                life_expectancy=self.get_life_expectancy(clean_name),
                grooming_needs=self.get_grooming_needs(clean_name)
            )
            
            breed_db[breed_id] = breed_info
        
        return breed_db
    
    def get_breed_characteristics(self, breed_name: str) -> str:
        """Obtener características físicas de la raza"""
        # Base de datos de características por raza
        characteristics_db = {
            "Chihuahua": "Perro pequeño, cabeza redonda, ojos grandes y expresivos, orejas grandes y erectas, cuerpo compacto y musculoso.",
            "Golden Retriever": "Perro mediano-grande, pelaje dorado y denso, cabeza ancha, ojos marrones y amigables, cuerpo musculoso y bien proporcionado.",
            "Labrador Retriever": "Perro mediano-grande, pelaje corto y denso, cabeza ancha, ojos expresivos, cuerpo musculoso y atlético.",
            "German Shepherd": "Perro grande, pelaje doble, cabeza alargada, orejas erectas, cuerpo musculoso y ágil.",
            "French Bulldog": "Perro pequeño-mediano, cabeza cuadrada, orejas de murciélago, cuerpo compacto y musculoso.",
            "Poodle": "Perro de tamaño variable, pelaje rizado y denso, cabeza alargada, orejas colgantes, cuerpo elegante y proporcionado.",
            "Beagle": "Perro mediano, pelaje tricolor, cabeza redonda, orejas largas y colgantes, cuerpo compacto y musculoso.",
            "Rottweiler": "Perro grande, pelaje negro con marcas marrones, cabeza ancha, orejas triangulares, cuerpo musculoso y poderoso.",
            "Siberian Husky": "Perro mediano-grande, pelaje doble y denso, ojos azules o marrones, orejas triangulares, cuerpo ágil y atlético.",
            "Bulldog": "Perro mediano, cabeza grande y cuadrada, hocico corto, cuerpo compacto y musculoso, piel suelta y arrugada."
        }
        
        return characteristics_db.get(breed_name, f"Características físicas específicas de {breed_name}.")
    
    def get_breed_temperament(self, breed_name: str) -> str:
        """Obtener temperamento de la raza"""
        temperament_db = {
            "Chihuahua": "Valiente, leal, alerta, puede ser territorial, ideal para apartamentos.",
            "Golden Retriever": "Amigable, inteligente, leal, excelente con niños, muy sociable.",
            "Labrador Retriever": "Amigable, activo, inteligente, excelente con familias, muy sociable.",
            "German Shepherd": "Leal, valiente, inteligente, excelente guardián, necesita liderazgo firme.",
            "French Bulldog": "Amigable, tranquilo, adaptable, excelente para apartamentos, muy cariñoso.",
            "Poodle": "Inteligente, activo, leal, excelente con familias, muy sociable y juguetón.",
            "Beagle": "Amigable, curioso, activo, excelente con niños, muy sociable y juguetón.",
            "Rottweiler": "Leal, valiente, confiado, excelente guardián, necesita socialización temprana.",
            "Siberian Husky": "Amigable, activo, independiente, excelente con familias, muy sociable.",
            "Bulldog": "Amigable, tranquilo, leal, excelente con familias, muy cariñoso y paciente."
        }
        
        return temperament_db.get(breed_name, f"Temperamento característico de {breed_name}.")
    
    def get_care_requirements(self, breed_name: str) -> str:
        """Obtener requisitos de cuidado de la raza"""
        care_db = {
            "Chihuahua": "Ejercicio moderado, cepillado semanal, cuidado dental, protección del frío, alimentación de alta calidad.",
            "Golden Retriever": "Ejercicio diario intenso, cepillado frecuente, cuidado dental, baño regular, alimentación balanceada.",
            "Labrador Retriever": "Ejercicio diario intenso, cepillado regular, cuidado dental, baño ocasional, alimentación controlada.",
            "German Shepherd": "Ejercicio diario intenso, cepillado frecuente, cuidado dental, baño regular, alimentación de alta calidad.",
            "French Bulldog": "Ejercicio moderado, cepillado semanal, cuidado dental, protección del calor, alimentación balanceada.",
            "Poodle": "Ejercicio diario, cepillado diario, cuidado dental, baño regular, alimentación de alta calidad.",
            "Beagle": "Ejercicio diario intenso, cepillado semanal, cuidado dental, baño ocasional, alimentación controlada.",
            "Rottweiler": "Ejercicio diario intenso, cepillado semanal, cuidado dental, baño ocasional, alimentación de alta calidad.",
            "Siberian Husky": "Ejercicio diario intenso, cepillado frecuente, cuidado dental, baño ocasional, alimentación de alta calidad.",
            "Bulldog": "Ejercicio moderado, cepillado semanal, cuidado dental, baño ocasional, alimentación balanceada."
        }
        
        return care_db.get(breed_name, f"Requisitos de cuidado específicos para {breed_name}.")
    
    def get_health_issues(self, breed_name: str) -> str:
        """Obtener problemas de salud comunes de la raza"""
        health_db = {
            "Chihuahua": "Problemas dentales, luxación de rótula, hipoglucemia, problemas cardíacos, hidrocefalia.",
            "Golden Retriever": "Displasia de cadera, problemas oculares, cáncer, problemas cardíacos, alergias.",
            "Labrador Retriever": "Displasia de cadera, problemas oculares, obesidad, problemas cardíacos, alergias.",
            "German Shepherd": "Displasia de cadera, problemas oculares, problemas digestivos, alergias, problemas de comportamiento.",
            "French Bulldog": "Problemas respiratorios, problemas oculares, problemas de piel, problemas cardíacos, obesidad.",
            "Poodle": "Problemas oculares, problemas de piel, problemas cardíacos, alergias, problemas dentales.",
            "Beagle": "Problemas oculares, problemas de oído, obesidad, problemas cardíacos, alergias.",
            "Rottweiler": "Displasia de cadera, problemas oculares, problemas cardíacos, cáncer, problemas de comportamiento.",
            "Siberian Husky": "Problemas oculares, problemas de piel, problemas cardíacos, alergias, problemas de comportamiento.",
            "Bulldog": "Problemas respiratorios, problemas oculares, problemas de piel, problemas cardíacos, obesidad."
        }
        
        return health_db.get(breed_name, f"Problemas de salud comunes en {breed_name}.")
    
    def get_training_advice(self, breed_name: str) -> str:
        """Obtener consejos de entrenamiento para la raza"""
        training_db = {
            "Chihuahua": "Entrenamiento positivo, socialización temprana, paciencia, refuerzo positivo, evitar castigos.",
            "Golden Retriever": "Entrenamiento positivo, socialización temprana, ejercicio mental, refuerzo positivo, consistencia.",
            "Labrador Retriever": "Entrenamiento positivo, socialización temprana, ejercicio mental, refuerzo positivo, consistencia.",
            "German Shepherd": "Entrenamiento firme pero positivo, socialización temprana, liderazgo claro, ejercicio mental, consistencia.",
            "French Bulldog": "Entrenamiento positivo, socialización temprana, paciencia, refuerzo positivo, evitar sobreesfuerzo.",
            "Poodle": "Entrenamiento positivo, socialización temprana, ejercicio mental, refuerzo positivo, consistencia.",
            "Beagle": "Entrenamiento positivo, socialización temprana, ejercicio mental, refuerzo positivo, consistencia.",
            "Rottweiler": "Entrenamiento firme pero positivo, socialización temprana, liderazgo claro, ejercicio mental, consistencia.",
            "Siberian Husky": "Entrenamiento positivo, socialización temprana, ejercicio mental, refuerzo positivo, consistencia.",
            "Bulldog": "Entrenamiento positivo, socialización temprana, paciencia, refuerzo positivo, evitar sobreesfuerzo."
        }
        
        return training_db.get(breed_name, f"Consejos de entrenamiento específicos para {breed_name}.")
    
    def get_breed_origin(self, breed_name: str) -> str:
        """Obtener origen de la raza"""
        origin_db = {
            "Chihuahua": "México, siglo XIX, descendiente de perros precolombinos.",
            "Golden Retriever": "Escocia, siglo XIX, desarrollado para la caza.",
            "Labrador Retriever": "Canadá, siglo XIX, desarrollado para la caza acuática.",
            "German Shepherd": "Alemania, siglo XIX, desarrollado para pastoreo y trabajo.",
            "French Bulldog": "Francia, siglo XIX, descendiente del Bulldog inglés.",
            "Poodle": "Alemania, siglo XV, desarrollado para la caza acuática.",
            "Beagle": "Inglaterra, siglo XIX, desarrollado para la caza de liebres.",
            "Rottweiler": "Alemania, siglo XIX, desarrollado para pastoreo y trabajo.",
            "Siberian Husky": "Siberia, siglo XIX, desarrollado para trineo y trabajo.",
            "Bulldog": "Inglaterra, siglo XIX, desarrollado para combate con toros."
        }
        
        return origin_db.get(breed_name, f"Origen histórico de {breed_name}.")
    
    def get_breed_size(self, breed_name: str) -> str:
        """Obtener tamaño de la raza"""
        size_db = {
            "Chihuahua": "Pequeño (15-25 cm, 1-3 kg)",
            "Golden Retriever": "Mediano-Grande (55-61 cm, 25-34 kg)",
            "Labrador Retriever": "Mediano-Grande (55-62 cm, 25-36 kg)",
            "German Shepherd": "Grande (55-65 cm, 22-40 kg)",
            "French Bulldog": "Pequeño-Mediano (30-35 cm, 8-14 kg)",
            "Poodle": "Variable (Toy: 24-28 cm, Mini: 28-35 cm, Estándar: 45-60 cm)",
            "Beagle": "Mediano (33-41 cm, 9-11 kg)",
            "Rottweiler": "Grande (56-69 cm, 35-60 kg)",
            "Siberian Husky": "Mediano-Grande (51-60 cm, 16-27 kg)",
            "Bulldog": "Mediano (31-40 cm, 18-25 kg)"
        }
        
        return size_db.get(breed_name, f"Tamaño característico de {breed_name}.")
    
    def get_energy_level(self, breed_name: str) -> str:
        """Obtener nivel de energía de la raza"""
        energy_db = {
            "Chihuahua": "Baja-Media (ejercicio moderado diario)",
            "Golden Retriever": "Alta (ejercicio intenso diario)",
            "Labrador Retriever": "Alta (ejercicio intenso diario)",
            "German Shepherd": "Alta (ejercicio intenso diario)",
            "French Bulldog": "Baja-Media (ejercicio moderado diario)",
            "Poodle": "Media-Alta (ejercicio regular diario)",
            "Beagle": "Alta (ejercicio intenso diario)",
            "Rottweiler": "Media-Alta (ejercicio regular diario)",
            "Siberian Husky": "Muy Alta (ejercicio muy intenso diario)",
            "Bulldog": "Baja (ejercicio ligero diario)"
        }
        
        return energy_db.get(breed_name, f"Nivel de energía característico de {breed_name}.")
    
    def get_life_expectancy(self, breed_name: str) -> str:
        """Obtener expectativa de vida de la raza"""
        life_db = {
            "Chihuahua": "12-20 años",
            "Golden Retriever": "10-12 años",
            "Labrador Retriever": "10-12 años",
            "German Shepherd": "9-13 años",
            "French Bulldog": "10-12 años",
            "Poodle": "12-15 años",
            "Beagle": "12-15 años",
            "Rottweiler": "8-10 años",
            "Siberian Husky": "12-14 años",
            "Bulldog": "8-10 años"
        }
        
        return life_db.get(breed_name, f"Expectativa de vida de {breed_name}.")
    
    def get_grooming_needs(self, breed_name: str) -> str:
        """Obtener necesidades de aseo de la raza"""
        grooming_db = {
            "Chihuahua": "Cepillado semanal, baño mensual, cuidado dental regular",
            "Golden Retriever": "Cepillado diario, baño mensual, cuidado dental regular",
            "Labrador Retriever": "Cepillado semanal, baño mensual, cuidado dental regular",
            "German Shepherd": "Cepillado diario, baño mensual, cuidado dental regular",
            "French Bulldog": "Cepillado semanal, baño mensual, cuidado dental regular",
            "Poodle": "Cepillado diario, baño semanal, cuidado dental regular",
            "Beagle": "Cepillado semanal, baño mensual, cuidado dental regular",
            "Rottweiler": "Cepillado semanal, baño mensual, cuidado dental regular",
            "Siberian Husky": "Cepillado diario, baño mensual, cuidado dental regular",
            "Bulldog": "Cepillado semanal, baño mensual, cuidado dental regular"
        }
        
        return grooming_db.get(breed_name, f"Necesidades de aseo de {breed_name}.")
    
    def create_breed_documents(self) -> List[Document]:
        """Crear documentos para el pipeline RAG"""
        documents = []
        
        for breed_id, breed_info in self.breed_database.items():
            # Crear documento estructurado
            breed_text = f"""
            Raza: {breed_info.breed_name}
            ID: {breed_info.breed_id}
            
            Características Físicas: {breed_info.characteristics}
            
            Temperamento: {breed_info.temperament}
            
            Requisitos de Cuidado: {breed_info.care_requirements}
            
            Problemas de Salud Comunes: {breed_info.health_issues}
            
            Consejos de Entrenamiento: {breed_info.training_advice}
            
            Origen: {breed_info.origin}
            
            Tamaño: {breed_info.size}
            
            Nivel de Energía: {breed_info.energy_level}
            
            Expectativa de Vida: {breed_info.life_expectancy}
            
            Necesidades de Aseo: {breed_info.grooming_needs}
            """
            
            doc = Document(
                id=f"breed_{breed_id}",
                text=breed_text.strip(),
                source="breed_database",
                page=1
            )
            documents.append(doc)
        
        return documents

