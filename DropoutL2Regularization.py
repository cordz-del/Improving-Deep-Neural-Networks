from tensorflow.keras.regularizers import l2

model = keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Assuming X_train and y_train are defined
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
