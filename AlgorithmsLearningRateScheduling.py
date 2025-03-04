from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    # Decrease the learning rate by 10% every 5 epochs
    if epoch % 5 == 0 and epoch:
        return lr * 0.9
    return lr

# Example model architecture
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Choose an optimizer – you can switch between Adam or SGD with momentum
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# Alternatively, use:
# optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(scheduler)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler])
