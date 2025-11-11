import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


def analyze_and_save(model, trainer, X_train, y_train, X_val, y_val, X_test, y_test, save_dir="results", model_name="QNN"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train).numpy()
        pred_val = model(X_val).numpy()
        pred_test = model(X_test).numpy()

    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()

    metrics = {
        "MSE": [mean_squared_error(y_train, pred_train),
                mean_squared_error(y_val, pred_val),
                mean_squared_error(y_test, pred_test)],
        "MAE": [mean_absolute_error(y_train, pred_train),
                mean_absolute_error(y_val, pred_val),
                mean_absolute_error(y_test, pred_test)],
    }

    print("\n" + "="*50)
    print(f"KẾT QUẢ {model_name.upper()}")
    print("="*50)
    print(f"{'':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*50)
    for name in ["MSE", "MAE"]:
        print(f"{name:<10} {metrics[name][0]:.6f}    {metrics[name][1]:.6f}    {metrics[name][2]:.6f}")
    print("="*50)

    plt.figure(figsize=(12, 4))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label="Train Loss", alpha=0.8)
    plt.plot(trainer.val_losses, label="Val Loss", alpha=0.8)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training Curve")

    # Prediction vs True (Test)
    plt.subplot(1, 2, 2)
    plt.plot(y_test[:100], 'b-', label="True", linewidth=2)
    plt.plot(pred_test[:100], 'r--', label="Predict", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Sunspot")
    plt.legend()
    plt.title("Prediction vs True (Test)")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_results.pdf")
    plt.close()

    torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pth")

    print(f"Saved: {save_dir}/{model_name}_results.pdf")
    print(f"Saved model: {save_dir}/{model_name}_best.pth")

