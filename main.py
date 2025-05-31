from read_data import read_and_preprocess
from models import build_dense_model, build_lstm_model, build_cnn_lstm_model
from evaluate_models import evaluate_model

SEQ_LEN = 24
X_train, y_train, X_test, y_test, scaler = read_and_preprocess("household_power_consumption.txt", SEQ_LEN)

input_shape = X_train.shape[1:]

models = {
    'Dense': build_dense_model(input_shape),
    'LSTM': build_lstm_model(input_shape),
    'CNN-LSTM': build_cnn_lstm_model(input_shape)
}

for name, model in models.items():
    print(f"\n🔧 Training {name} model...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)

    mse, mae, preds = evaluate_model(model, X_test, y_test)
    print(f"📊 {name} MSE: {mse:.4f}, MAE: {mae:.4f}")
