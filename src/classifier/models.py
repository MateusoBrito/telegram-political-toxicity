from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#from sklearn.naive_bayes import GaussianNB

# É necessário ter instalado: pip install xgboost lightgbm
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

CLASSIFIER_REGISTRY = {
    #'knn': KNeighborsClassifier,
    #'svm': SVC,
    'dt': DecisionTreeClassifier,
    'lr': LogisticRegression,
    'rf': RandomForestClassifier,
    #'xgb': XGBClassifier,
    #'lgbm': LGBMClassifier,
    #'mlp': MLPClassifier,
    #'nb': GaussianNB 
}

def get_classifier(name: str):
    """Retorna a classe do classificador pelo nome."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Classificador '{name}' não suportado. "
            f"Disponíveis: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    return CLASSIFIER_REGISTRY[name]