from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import joblib
import os

class ModelTrainer:
    def __init__(self, model_class, param_grid, cv=5):
        self.base_model = model_class()
        self.param_grid = param_grid
        self.cv = cv
        self.best_model = None
        self.best_params = None
    
    def optimize_hyperparameters(self, X, y):
        """
        Roda o GridSearch para encontrar a melhor configuração.
        """

        if not self.param_grid:
            self.best_params = {}
            self.best_model = self.base_model
            return
        
        search = GridSearchCV(
            estimator = self.base_model,
            param_grid = self.param_grid,
            cv=self.cv,
            scoring = 'accuracy',
            n_jobs = -1
        )
        search.fit(X,y)

        self.best_params = search.best_params_
        self.best_model = search.best_estimator_
        print(f"Melhores parâmetros: {self.best_params}")
    
    def train(self, X, y):
        """
        Treina o modelo final (usando os melhores parâmetros já encontrados).
        """

        if self.best_model is None:
            self.best_model = clone(self.base_model)
        
        print("Treinando modelo final...")
        self.best_model.fit(X, y)
        
    def predict(self, X):
        """Faz predições."""
        if self.best_model is None:
            raise Exception("O modelo ainda não foi treinado. Execute .train() primeiro.")
        return self.best_model.predict(X)

    def save(self, filepath):
        """Salva o modelo treinado em disco."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"Modelo salvo em {filepath}")