#!/usr/bin/env python3
# train_keras.py
import argparse
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, Callback
from tensorflow.keras.utils import Progbar


def load_npz(path):
    data = np.load(path)
    return (
        data['X_train'].astype(np.float32),
        data['y_train'].astype(np.int64),
        data['X_val'].astype(np.float32),
        data['y_val'].astype(np.int64),
        data['X_test'].astype(np.float32),
        data['y_test'].astype(np.int64)
    )


def build_mlp(input_dim):
    model = Sequential()
    # 5 gizli katman, her biri 50 ünite ve ReLU aktivasyon
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # tek logit çıkış
    return model


def capture_layer_outputs(model, X):
    """Keras modelinin her Dense katmanındaki ReLU sonrası aktivasyonları döndürür."""
    # Çıktısını almak istediğiniz katmanların output'larını listeleyin
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
    raw_outputs = activation_model.predict(X)  # her katmanın doğrusal çıkışı
    # ReLU aktivasyonunu uygulamak (Keras zaten uyguluyor, yine de toplu halde çıkarabiliriz)
    activations = {}
    # son katman logit, sigmoid çıkışı
    for idx, out in enumerate(raw_outputs[:-1]):
        activations[f"hidden_{idx+1}"] = out
    activations["logit"] = raw_outputs[-1]
    return activations


class LiveProgressCallback(Callback):
    """Terminalde canlı ilerleme ve metrik gösterimi için basit bir callback."""
    def __init__(self):
        super().__init__()
        self.progbar = None

    def on_train_begin(self, logs=None):
        print("Eğitim başladı.")

    def on_epoch_begin(self, epoch, logs=None):
        steps = self.params.get("steps")
        if steps is None:
            samples = self.params.get("samples") or 0
            batch_size = self.params.get("batch_size") or 1
            steps = samples // batch_size + int(samples % batch_size > 0)
        self.progbar = Progbar(target=steps, verbose=1)
        print(f"\nEpoch {epoch + 1}/{self.params.get('epochs', '?')}")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Göster: loss ve varsa accuracy
        values = [("loss", logs.get("loss"))]
        # Keras metrik adı 'accuracy' veya 'acc' olabilir
        if "accuracy" in logs:
            values.append(("accuracy", logs.get("accuracy")))
        elif "acc" in logs:
            values.append(("acc", logs.get("acc")))
        if self.progbar is not None:
            self.progbar.add(1, values=values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_str_parts = []
        if "val_loss" in logs:
            val_str_parts.append(f"val_loss: {logs['val_loss']:.4f}")
        if "val_accuracy" in logs:
            val_str_parts.append(f"val_accuracy: {logs['val_accuracy']:.4f}")
        elif "val_acc" in logs:
            val_str_parts.append(f"val_acc: {logs['val_acc']:.4f}")
        if val_str_parts:
            print(" - " + " - ".join(val_str_parts))


def main():
    parser = argparse.ArgumentParser(
        description="Eski Keras mimarisi ile DNN eğitimi")
    parser.add_argument("--data", type=str, required=True,
                        help=".npz veri seti yolu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str,
                        default="run_keras", help="Çıktı klasörü")
    args = parser.parse_args()

    # Veriyi yükle
    X_train, y_train, X_val, y_val, X_test, y_test = load_npz(args.data)

    # Modeli oluştur
    model = build_mlp(X_train.shape[1])
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )

    # Eğitim (val_split yerine doğrudan X_val kullanıyoruz)
    # Canlı ilerleme ve loglama için callback'ler
    tb_cb = TensorBoard(log_dir=args.out, update_freq='batch', write_graph=False)
    csv_cb = CSVLogger(os.path.join(args.out, "training_log.csv"), append=False)
    live_cb = LiveProgressCallback()

    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=[live_cb, tb_cb, csv_cb],
        verbose=0  # kendi ilerleme çubuğumuzu kullanıyoruz
    )

    # Test seti doğruluğu
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Katman aktivasyonlarını yakala
    acts = capture_layer_outputs(model, X_test)
    # X_test ve y_test'i de ekle
    acts["X_test"] = X_test
    acts["y_test"] = y_test
    # .npz dosyasına kaydet
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "layer_outputs_test.npz"),
        **acts
    )
    print(
        f"Katman aktivasyonları {args.out}/layer_outputs_test.npz dosyasına kaydedildi.")

    # Modelle ilgili metrikleri kaydet
    import json
    metrics = {
        "test_acc": float(test_acc),
        "test_loss": float(test_loss)
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
