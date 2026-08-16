from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit

def get_validation_strategy(strategy_name, n_splits=5, random_state=42):
    if strategy_name == 'stratified_kfold':
        # Mantém a proporção das classes em cada dobra (Ideal para classificação)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    elif strategy_name == 'kfold':
        # K-Fold simples (Aleatório, não olha as classes)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    elif strategy_name == 'holdout':
        # Simula um Holdout (70/30, 80/20) usando ShuffleSplit com 1 única repetição.
        # test_size=0.2 significa 20% para teste.
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        
    else:
        raise ValueError(f"Estratégia de validação '{strategy_name}' não suportada.")