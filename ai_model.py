import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class GestureAIModel:
    def __init__(self, model_path='gesture_model.joblib', scaler_path='scaler.joblib'):
        """
        Inicializa o modelo de IA para reconhecimento de gestos
        
        Args:
            model_path: Caminho para salvar/carregar o modelo
            scaler_path: Caminho para salvar/carregar o scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()
        
        # Carrega o modelo e o scaler se existirem
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess_landmarks(self, landmarks):
        """
        Pré-processa os pontos de referência para o modelo
        
        Args:
            landmarks: Lista de pontos de referência da mão
            
        Returns:
            features: Array numpy com as features processadas
        """
        try:
            if landmarks is None or len(landmarks) < 21:
                return None
                
            # Converte para array numpy
            landmarks_array = np.array(landmarks)
            
            # Extrai coordenadas x e y
            x_coords = landmarks_array[:, 1].astype(float)
            y_coords = landmarks_array[:, 2].astype(float)
            
            # Normaliza as coordenadas
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1
            
            x_coords = (x_coords - x_min) / x_range
            y_coords = (y_coords - y_min) / y_range
            
            # Calcula distâncias entre pontos importantes
            thumb_index_dist = np.sqrt((x_coords[4] - x_coords[8])**2 + (y_coords[4] - y_coords[8])**2)
            index_middle_dist = np.sqrt((x_coords[8] - x_coords[12])**2 + (y_coords[8] - y_coords[12])**2)
            
            # Calcula ângulos entre dedos
            angles = []
            for i in range(0, 20, 4):
                if i + 4 < len(landmarks):
                    v1 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
                    v2 = np.array([x_coords[i+3] - x_coords[i+1], y_coords[i+3] - y_coords[i+1]])
                    
                    # Evita divisão por zero
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 > 0 and norm_v2 > 0:
                        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                        # Limita o valor para evitar erros numéricos
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                    else:
                        angle = 0
                    angles.append(angle)
            
            # Combina todas as features
            features = np.concatenate([
                x_coords,
                y_coords,
                [thumb_index_dist, index_middle_dist],
                angles
            ])
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Erro no pré-processamento: {str(e)}")
            return None
    
    def train(self, X, y):
        """
        Treina o modelo com dados de treinamento
        
        Args:
            X: Features de treinamento
            y: Labels de treinamento
        """
        # Pré-processa os dados
        X_processed = []
        y_processed = []
        
        print("Pré-processando dados de treinamento...")
        for i, landmarks in enumerate(X):
            if i % 1000 == 0:
                print(f"Processando amostra {i+1}/{len(X)}")
            features = self.preprocess_landmarks(landmarks)
            if features is not None:
                X_processed.append(features)
                y_processed.append(y[i])
        
        if not X_processed:
            raise ValueError("Nenhum dado válido para treinamento")
        
        X_processed = np.array(X_processed)
        y_processed = np.array(y_processed)
        
        print(f"Dados processados: {len(X_processed)} amostras válidas")
        
        # Normaliza os dados
        print("Normalizando dados...")
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Treina o modelo
        print("Treinando modelo...")
        self.model.fit(X_scaled, y_processed)
        
        # Salva o modelo e o scaler
        print("Salvando modelo...")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def predict(self, landmarks):
        """
        Faz a predição do gesto
        
        Args:
            landmarks: Lista de pontos de referência da mão
            
        Returns:
            prediction: Gesto predito
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")
            
        # Pré-processa os dados
        features = self.preprocess_landmarks(landmarks)
        if features is None:
            return None
            
        # Normaliza os dados
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Faz a predição
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def predict_proba(self, landmarks):
        """
        Retorna as probabilidades de cada gesto
        
        Args:
            landmarks: Lista de pontos de referência da mão
            
        Returns:
            probabilities: Probabilidades de cada gesto
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")
            
        # Pré-processa os dados
        features = self.preprocess_landmarks(landmarks)
        if features is None:
            return None
            
        # Normaliza os dados
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Calcula as probabilidades
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return probabilities 