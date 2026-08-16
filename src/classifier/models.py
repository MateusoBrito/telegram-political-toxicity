# src/models.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIER_REGISTRY = {
    'knn': KNeighborsClassifier,
    'svm': SVC,
    'dt': DecisionTreeClassifier,
}

def get_classifier(name: str):
    """Retorna a classe do classificador pelo nome."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Classificador '{name}' não suportado. "
            f"Disponíveis: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    return CLASSIFIER_REGISTRY[name]