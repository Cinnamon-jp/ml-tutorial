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
class SimpleLinearRegression(nnx.Module):
    # 入出力次元の定義
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(din, dout, rngs=rngs)

    # 順伝播の定義
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


@nnx.jit
# 訓練ステップの定義
def train_step(
    model: SimpleLinearRegression,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    # 損失関数の定義(平均二乗誤差)
    def loss_fn(model: SimpleLinearRegression) -> jax.Array:
        return jnp.mean((model(x) - y) ** 2)

    loss, grad = nnx.value_and_grad(loss_fn)(model)

    optimizer.update(model, grad)

    return loss


def main() -> None:
    # CSVデータの読み込みと変数代入
    testdata: pd.DataFrame = pd.read_csv(
        "datasets/simple_linear_regression_testdata.csv"
    )
    # print(testdata.head())
    # 入力と出力を (N, 1) の二次元形状に変形
    x_data: jax.Array = jnp.array(testdata.iloc[:, 0].to_numpy())[:, None]
    y_data: jax.Array = jnp.array(testdata.iloc[:, 1].to_numpy())[:, None]

    # データを 平均0, 分散1 に標準化
    x_mean: jax.Array = x_data.mean()
    x_std: jax.Array = x_data.std()
    x_data = (x_data - x_mean) / x_std  # 配列の要素ごとに実行される

    y_mean: jax.Array = y_data.mean()
    y_std: jax.Array = y_data.std()
    y_data = (y_data - y_mean) / y_std  # 配列の要素ごとに実行される

    rng = nnx.Rngs(42)

    model = SimpleLinearRegression(din=1, dout=1, rngs=rng)

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

    # プロット用の DataFrame を作成
    plot_df = pd.DataFrame(
        {"x": x_orig.squeeze(), "y": y_orig.squeeze(), "y_pred": y_pred_orig.squeeze()}
    )

    # seaborn のテーマを設定
    sns.set_theme(style="darkgrid")

    # matplotlib の描画領域を設定 [inch]
    plt.figure(figsize=(10, 6))

    # 実際のデータを散布図としてプロット
    sns.scatterplot(
        data=plot_df, x="x", y="y", label="Original Data", alpha=0.6, color="royalblue"
    )

    # モデルによる回帰直線をプロット
    plot_df_sorted = plot_df.sort_values(by="x")
    sns.lineplot(
        data=plot_df_sorted,
        x="x",
        y="y_pred",
        label="Regression Line",
        color="crimson",
        linewidth=2.5,
    )

    plt.title("Simple Linear Regression Result (Flax NNX)", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()

    # 保存ディレクトリの作成と画像の保存
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "simple_linear_regression_result.png")
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
    plt.title("Loss History", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()

    loss_plot_path = os.path.join(output_dir, "simple_linear_regression_loss.png")
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
        print(f"Weight: {float(weights[0, 0]):.6f}")
        print(f"Bias: {float(bias[0]):.6f}")


if __name__ == "__main__":
    main()
