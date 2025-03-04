import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
import numpy as np

# Example synthetic data
input_dim = 20
num_samples = 1000
X_train = np.random.randn(num_samples, input_dim)
y_train = (np.random.rand(num_samples) > 0.5).astype(int)

def build_model(hp):
    model = keras.Sequential()
    # Tune the number of units in the first Dense layer
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=256, step=32),
                           activation='relu', input_shape=(input_dim,)))
    # Tune dropout rate for regularization
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Tune the learning rate for the optimizer
    model.compile(optimizer=keras.optimizers.Adam(
                    hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='hyperparam_tuning_example'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Print the best hyperparameters found
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best units:", best_hps.get('units'))
print("Best dropout:", best_hps.get('dropout'))
print("Best learning rate:", best_hps.get('learning_rate'))
