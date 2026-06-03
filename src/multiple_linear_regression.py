import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# モデル定義
class MultipleLinearRegression(nnx.Module):
    # 入出力次元の定義
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(din, dout, rngs=rngs)

    # 順伝播の定義
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


@nnx.jit
# 訓練ステップの定義
def train_step(
    model: MultipleLinearRegression,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    # 損失関数の定義(平均二乗誤差)
    def loss_fn(model: MultipleLinearRegression) -> jax.Array:
        return jnp.mean((model(x) - y) ** 2)

    loss, grad = nnx.value_and_grad(loss_fn)(model)

    optimizer.update(model, grad)

    return loss


def main():
    # CSVデータの読み込みと変数代入
    testdata: pd.DataFrame = pd.read_csv(
        "datasets/multiple_linear_regressin_testdata.csv"
    )

    # 特徴量 (x1~x5) とターゲット (y) を抽出
    x_columns = ["x1", "x2", "x3", "x4", "x5"]
    x_data: jax.Array = jnp.array(testdata[x_columns].to_numpy())
    y_data: jax.Array = jnp.array(testdata["y"].to_numpy())[:, None]

    # データを 平均0, 分散1 に標準化 (特徴量ごとに標準化)s
    x_mean: jax.Array = x_data.mean(axis=0)
    x_std: jax.Array = x_data.std(axis=0)
    x_data = (x_data - x_mean) / x_std

    y_mean: jax.Array = y_data.mean()
    y_std: jax.Array = y_data.std()
    y_data = (y_data - y_mean) / y_std

    # 乱数シードの初期化
    rng = nnx.Rngs(42)

    # 重回帰モデルの初期化 (入力次元 5, 出力次元 1)
    model = MultipleLinearRegression(din=5, dout=1, rngs=rng)

    tx = optax.sgd(learning_rate=0.1)
    optimizer = nnx.Optimizer(model=model, tx=tx, wrt=nnx.Param)

    # 訓練ループの実行
    epochs_list = []
    losses_list = []
    for epoch in range(50):
        loss_value = train_step(model, optimizer, x_data, y_data)
        epochs_list.append(epoch)
        losses_list.append(float(loss_value))

        if epoch == 0 or epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss_value}")

    # 予測値の計算 (標準化スケール)
    y_pred: jax.Array = model(x_data)

    # 逆標準化して元のデータスケールに戻す
    x_orig: jax.Array = (x_data * x_std) + x_mean
    y_orig: jax.Array = (y_data * y_std) + y_mean
    y_pred_orig: jax.Array = (y_pred * y_std) + y_mean

    # プロット用の DataFrame を作成 (実測値 vs 予測値)
    plot_df = pd.DataFrame(
        {"y_actual": y_orig.squeeze(), "y_pred": y_pred_orig.squeeze()}
    )

    # seaborn のテーマを設定
    sns.set_theme(style="darkgrid")

    # matplotlib の描画領域を設定 [inch]
    plt.figure(figsize=(10, 6))

    # 実測値と予測値の散布図をプロット
    sns.scatterplot(
        data=plot_df,
        x="y_actual",
        y="y_pred",
        label="Predicted vs Actual",
        alpha=0.6,
        color="royalblue",
    )

    # 理想的な予測を示す対角線 (y = x) をプロット
    min_val = min(plot_df["y_actual"].min(), plot_df["y_pred"].min())
    max_val = max(plot_df["y_actual"].max(), plot_df["y_pred"].max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.title("Multiple Linear Regression Result (Actual vs Predicted)", fontsize=14)
    plt.xlabel("Actual y", fontsize=12)
    plt.ylabel("Predicted y", fontsize=12)
    plt.legend()

    # 保存ディレクトリの作成と画像の保存
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "multiple_linear_regression_result.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Regression plot successfully saved to {plot_path}")

    # 誤差(損失)をプロット
    loss_df = pd.DataFrame({"Epoch": epochs_list, "Loss": losses_list})

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=loss_df,
        x="Epoch",
        y="Loss",
        label="Loss",
        color="purple",
        marker="o",
        linewidth=2,
    )
    plt.title("Loss History (Multiple Linear Regression)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()

    loss_plot_path = os.path.join(output_dir, "multiple_linear_regression_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss plot successfully saved to {loss_plot_path}")

    # 訓練済みの重みとバイアスを出力
    print("\n--- Trained Parameters ---")
    kernel = model.linear.kernel
    bias_param = model.linear.bias
    if kernel is not None and bias_param is not None:
        weights = kernel[...]
        bias = bias_param[...]
        for i, col in enumerate(x_columns):
            print(f"Weight ({col}): {float(weights[i, 0]):.6f}")
        print(f"Bias: {float(bias[0]):.6f}")


if __name__ == "__main__":
    main()
