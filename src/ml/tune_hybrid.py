import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

os.makedirs('models/optuna_logs', exist_ok=True)
os.makedirs('models/best_models_cnn_lstm', exist_ok=True)


def create_objective(X_train, y_train):

    def objective(trial):
        filters = trial.suggest_categorical('filters', [32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 3, 5)
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        model.add(Conv1D(filters=filters, 
                         kernel_size=kernel_size, 
                         activation='relu'))
        
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

        history = model.fit(
            X_train, y_train,
            validation_split=0.2, 
            epochs=40, 
            batch_size=batch_size,
            callbacks=[early_stop, pruning_callback],
            verbose=0 
        )

        return min(history.history['val_loss'])
    
    return objective


if __name__ == "__main__":
    
    motores = ['FD001', 'FD002', 'FD003', 'FD004']
    resultados_finais = []

    for motor in motores:
        print(f"\n{'='*50}")
        print(f"INICIANDO OTIMIZAÇÃO OPTUNA PARA O MOTOR: {motor}")
        print(f"{'='*50}\n")
        
        X_train = np.load(f"data/processed_data/X_train_{motor}.npz")['dados']
        y_train = np.load(f"data/processed_data/y_train_{motor}.npz")['dados']
        X_test = np.load(f"data/processed_data/X_test_{motor}.npz")['dados']
        y_test = np.load(f"data/processed_data/y_test_{motor}.npz")['dados']

        db_path = f"sqlite:///models/optuna_logs/optuna_study_{motor}.db"
        study = optuna.create_study(
            direction='minimize', 
            study_name=f"cnn_lstm_{motor}",
            storage=db_path,        
            load_if_exists=True,      
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5) 
        )
        
        print("Buscando as melhores arquiteturas (pode levar algum tempo)...")
        objetivo = create_objective(X_train, y_train)
        
        study.optimize(objetivo, n_trials=30)
        
        best_hps = study.best_params
        best_loss = study.best_value
        
        print(f"\nMelhores Hiperparâmetros para {motor} (MSE = {best_loss:.2f}):")
        for key, value in best_hps.items():
            print(f"  - {key}: {value}")
        
        print("\nTreinando o modelo final campeão...")
        
        best_model = Sequential()
        best_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        best_model.add(Conv1D(filters=best_hps['filters'], 
                              kernel_size=best_hps['kernel_size'], 
                              activation='relu'))
        best_model.add(LSTM(units=best_hps['lstm_units'], return_sequences=False))
        best_model.add(Dropout(best_hps['dropout_rate']))
        best_model.add(Dense(32, activation='relu'))
        best_model.add(Dense(1))

        best_model.compile(
            optimizer=Adam(learning_rate=best_hps['learning_rate']), 
            loss='mse', 
            metrics=['mae']
        )
        
        early_stop_final = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        
        best_model.fit(
            X_train, y_train, 
            validation_split=0.2, 
            epochs=50, 
            batch_size=best_hps['batch_size'],
            callbacks=[early_stop_final],
            verbose=0
        )

        y_pred = best_model.predict(X_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nConcluído {motor}! -> RMSE Final: {rmse:.2f} | MAE Final: {mae:.2f}")

        best_model.save(f"models/best_models_cnn_lstm/best_cnn_lstm_{motor}.keras")

        resultados_finais.append({
            'Motor': motor, 
            'RMSE': rmse, 
            'MAE': mae,
            'Filtros_Conv': best_hps['filters'],
            'Kernel_Conv': best_hps['kernel_size'],
            'Unidades_LSTM': best_hps['lstm_units'],
            'Batch_Size': best_hps['batch_size']
        })


    print("\n" + "="*50)
    print("RESUMO DOS CAMPEÕES - CNN-LSTM OTIMIZADA (OPTUNA)")
    print("="*50)
    df_final = pd.DataFrame(resultados_finais)
    print(df_final.to_markdown(index=False))

    df_final.to_csv("models/resultados_finais_cnn_lstm_optuna.csv", index=False)