import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
import keras_tuner as kt

os.makedirs('models/tuner_logs', exist_ok=True)
os.makedirs('models/best_models_cnn', exist_ok=True)


class MyHyperModel(kt.HyperModel):
    # Passamos a quantidade de sensores do motor atual para a classe na hora de criá-la
    def __init__(self, num_sensores):
        self.num_sensores = num_sensores
        
    def build(self, hp):
        model = Sequential()
        
        # Agora a rede se adapta ao número correto de sensores do motor!
        model.add(Input(shape=(30, self.num_sensores)))
        
        hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=128, step=32)
        hp_kernel = hp.Choice('kernel_size', values=[3, 5])
        
        model.add(Conv1D(
            filters=hp_filters_1, 
            kernel_size=hp_kernel, 
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        
        if hp.Boolean('use_second_conv'):
            hp_filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32)
            model.add(Conv1D(filters=hp_filters_2, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            
            if hp.Boolean('use_third_conv'):
                hp_filters_3 = hp.Int('filters_3', min_value=32, max_value=64, step=32)
                model.add(Conv1D(filters=hp_filters_3, kernel_size=3, activation='relu', padding='same'))
                model.add(MaxPooling1D(pool_size=2))
            
        model.add(Flatten())
        
        hp_dense = hp.Choice('dense_units', values=[32, 64, 128])
        hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        
        model.add(Dense(units=hp_dense, activation='relu'))
        model.add(Dropout(hp_dropout))
        model.add(Dense(1))
        
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

if __name__ == "__main__":
    
    motores = ['FD001', 'FD002', 'FD003', 'FD004']
    resultados_finais = []

    for motor in motores:
        print(f"\n{'='*50}")
        print(f"INICIANDO KerasTuner PARA O MOTOR: {motor}")
        print(f"{'='*50}\n")
        
        X_train = np.load(f"data/processed_data/X_train_{motor}.npz")['dados']
        y_train = np.load(f"data/processed_data/y_train_{motor}.npz")['dados']
        X_test = np.load(f"data/processed_data/X_test_{motor}.npz")['dados']
        y_test = np.load(f"data/processed_data/y_test_{motor}.npz")['dados']
        
        # Descobre quantos sensores esse motor específico tem (a 3ª dimensão do shape)
        qtd_sensores = X_train.shape[2]
        
        # Chama a classe passando a quantidade correta
        hypermodel = MyHyperModel(num_sensores=qtd_sensores)
        
        tuner = kt.Hyperband(
            hypermodel,
            objective='val_loss',
            max_epochs=40,
            factor=3,
            directory='models/tuner_logs',
            project_name=f'cnn_tuning_{motor}', 
            overwrite=False 
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("Iniciando a busca pelas melhores arquiteturas...")
        tuner.search(
            X_train, y_train, 
            epochs=40, 
            validation_split=0.2, 
            callbacks=[early_stop],
            verbose=1
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"\nMelhores Hiperparâmetros para {motor}:")
        print(f"- Filtros Conv 1: {best_hps.get('filters_1')}")
        print(f"- Lupa Conv 1: {best_hps.get('kernel_size')}")
        print(f"- Usou 2ª Camada Conv?: {best_hps.get('use_second_conv')}")
        
        if best_hps.get('use_second_conv'):
            print(f"  - Filtros Conv 2: {best_hps.get('filters_2')}")
            
            if best_hps.get('use_third_conv'):
                print(f"  - Usou 3ª Camada Conv?: Sim")
                print(f"    - Filtros Conv 3: {best_hps.get('filters_3')}")

        print(f"- Neurônios Densos: {best_hps.get('dense_units')}")
        print(f"- Taxa de Dropout: {best_hps.get('dropout_rate')}")
        print(f"- Learning Rate: {best_hps.get('learning_rate')}")
        
        print("\nTreinando o modelo final campeão...")
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(
            X_train, y_train, 
            epochs=50, 
            validation_split=0.2, 
            callbacks=[early_stop],
            verbose=0 
        )
        
        y_pred = best_model.predict(X_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nConcluído {motor}! -> RMSE Final: {rmse:.2f} | MAE Final: {mae:.2f}")
        
        best_model.save(f"models/best_models_cnn/best_cnn_{motor}.keras")
        
        resultados_finais.append({
            'Motor': motor, 
            'RMSE': rmse, 
            'MAE': mae,
            'Camadas_Conv': 3 if best_hps.get('use_third_conv') else (2 if best_hps.get('use_second_conv') else 1),
            'Filtros_1': best_hps.get('filters_1')
        })

    print("\n" + "="*50)
    print("RESUMO DOS CAMPEÕES - CNN 1D OTIMIZADA")
    print("="*50)
    df_final = pd.DataFrame(resultados_finais)
    print(df_final.to_markdown(index=False))

    df_final.to_csv("models/resultados_finais_cnn.csv", index=False)
    print("A pasta tuner_logs foi mantida conforme solicitado.")