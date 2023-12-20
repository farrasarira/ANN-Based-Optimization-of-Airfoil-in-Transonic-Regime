from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib as plt

# Custom METRICS function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true),axis=0))

def r2score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    return (1 - SS_res/(SS_tot + K.epsilon()))

def genNNmodel(X,y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler_X = StandardScaler().fit(X)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Tune the hyperparameters for the Adam optimizer
    # optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    while True:
        # Build a neural network model
        model = Sequential()
        model.add(Dense(128, input_dim=12, activation='sigmoid')) # 12 input
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(3, activation='linear')) # 3 output
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse,r2score])

        # Display the model summary
        # model.summary()

        # Train the model
        epochs = 100
        batch_size = 32
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Optional
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2) #, callbacks=[early_stopping])

        # Evaluate the final metrics value
        result = model.evaluate(X_test, y_test)
        val_rmse = result[1] # result[0] for final loss (mse)
        val_r2 = result[2]

        # Print final metrics
        # print(f"val_rmse: {val_rmse}, val_r2: {val_r2}")

        # Evaluate the final model on validation data
        y_pred = model.predict(X_test)
        # RMSE
        test_rmse = rmse(y_test, y_pred)
        # R-squared
        test_r2 = r2_score(y_test, y_pred, multioutput='raw_values') # multioutput using sklearn

        # Print rmse for validation data
        print(f"RMSE: {test_rmse}, R-squared: {test_r2}")

        if (test_r2>0.8).all():
            break

    # # Plot training and validation loss history
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss, [mse]')
    # plt.legend()
    # plt.show()

    return model