import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# モデルパラメータの初期化関数
def init_params(rng_key: jax.Array, din: int, dout: int) -> dict[str, jax.Array]:
    w_key, _ = jax.random.split(rng_key)
    # flax.nnx.Linear のデフォルト初期化(LeCun Normal)に合わせる
    # LeCun normal: variance = 1 / din. 今回 din=1 なので、std = 1.0 の標準正規分布
    w = jax.random.normal(w_key, (din, dout))
    b = jnp.zeros((dout,))
    return {"w": w, "b": b}


# 順伝播の定義 (モデルの予測)
def predict(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.dot(x, params["w"]) + params["b"]


# 損失関数の定義(平均二乗誤差)
def loss_fn(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)


@jax.jit
# 訓練ステップの定義
def train_step(
    params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    lr: float = 0.1,
) -> tuple[dict[str, jax.Array], jax.Array]:
    # 損失と勾配の計算
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # 勾配降下法(SGD)によるパラメータの更新
    params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return params, loss


def main():
    # CSVデータの読み込みと変数代入
    testdata: pd.DataFrame = pd.read_csv(
        "datasets/simple_linear_regression_testdata.csv"
    )

    # 入力と出力を (N, 1) の二次元形状に変形
    x_data: jax.Array = jnp.array(testdata.iloc[:, 0].to_numpy())[:, None]
    y_data: jax.Array = jnp.array(testdata.iloc[:, 1].to_numpy())[:, None]



    # PRNGキーの初期化
    key = jax.random.PRNGKey(42)

    # パラメータの初期化
    params = init_params(key, din=1, dout=1)

    # 訓練ループの実行
    epochs_list = []
    losses_list = []
    for epoch in range(50):
        params, loss_value = train_step(params, x_data, y_data)
        epochs_list.append(epoch)
        losses_list.append(float(loss_value))

        if epoch == 0 or epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss_value}")

    # 予測値の計算
    y_pred: jax.Array = predict(params, x_data)

    # 正規化を行わないため、元のデータスケールをそのまま使用
    x_orig: jax.Array = x_data
    y_orig: jax.Array = y_data
    y_pred_orig: jax.Array = y_pred

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

    plt.title("Simple Linear Regression Result (JAX only)", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()

    # 保存ディレクトリの作成と画像の保存
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "simple_linear_regression_jax_result.png")
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
    plt.title("Loss History (JAX only)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()

    loss_plot_path = os.path.join(output_dir, "simple_linear_regression_jax_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss plot successfully saved to {loss_plot_path}")

    # 訓練済みの重みとバイアスを出力
    print("\n--- Trained Parameters ---")
    weights = params["w"]
    bias = params["b"]
    print(f"Weight: {float(weights[0, 0]):.6f}")
    print(f"Bias: {float(bias[0]):.6f}")


if __name__ == "__main__":
    main()
