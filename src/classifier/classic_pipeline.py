import os
import datetime
import numpy as np
from src.classifier.trainer import ModelTrainer
from src.classifier.evaluation import evaluate_model, save_results, plot_confusion_matrix


def run_classic_pipeline(
    X, y_encoded, splitter, model_class, param_grid,
    model_name, embedding_name, dataset_path, output_dir, class_names
):
    """
    Pipeline de classificação clássica com embeddings pré-computados.

    Args:
        X (np.ndarray): Embeddings (n_samples, dim).
        y_encoded (np.ndarray): Labels codificados.
        splitter: Estratégia de validação cruzada (sklearn).
        model_class: Classe do classificador (ex: SVC).
        param_grid (dict): Grid de hiperparâmetros para GridSearchCV.
        model_name (str): Nome do modelo (para log).
        dataset_path (str): Nome do dataset (para log).
        output_dir (str): Diretório de saída para resultados.
        class_names (list): Nomes das classes.

    Returns:
        list: Lista de acurácias por fold.
    """
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_encoded)):
        print(f"\n--- Fold {fold + 1} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Treinamento e Otimização (GridSearchCV já retreina o best_estimator)
        trainer = ModelTrainer(model_class, param_grid, cv=3)
        print("Otimizando parâmetros...")
        trainer.optimize_hyperparameters(X_train, y_train)

        # Predição e Avaliação
        y_pred = trainer.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, labels=class_names)

        print(f"Acurácia: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")

        # Log de resultados
        fold_info = {
            'run_id': datetime.datetime.now().isoformat(),
            'dataset': os.path.basename(dataset_path),
            'embedding': embedding_name,
            'model': model_name,
            'fold': fold + 1,
            'best_params': str(trainer.best_params)
        }

        log_path = os.path.join(output_dir, 'experiment_log.csv')
        save_results(metrics, fold_info, log_path)

        # Salvamento de plots
        plots_path = os.path.join(output_dir, 'plots')
        cm_path = os.path.join(plots_path, f'cm_{model_name}_{fold+1}.png')
        plot_confusion_matrix(y_test, y_pred, labels=class_names, filepath=cm_path)

        fold_results.append(metrics['accuracy'])

    if fold_results:
        print(f"\n=== Média Final: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f}) ===")

    return fold_results