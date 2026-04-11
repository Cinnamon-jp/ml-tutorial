from jax import numpy as jnp
import jax
import pandas as pd
import seaborn as sns

# 単回帰モデル (y = wx + b)
def model(params: tuple[float, float], x: float) -> float:
    w, b = params
    return w * x + b

# 損失関数 (平均二乗誤差: MSE)
def loss_fn(params: tuple[float, float], x: float, y: float) -> float:
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# パラメータ更新
def update_params(params: tuple[float, float], x: float, y: float, learning_rate: float) -> tuple[float, float]:
    # 損失関数を偏微分し、現在の params における勾配を得る
    grads = jax.grad(loss_fn)(params, x, y)
    
    # 現在の重みとバイアス、計算された勾配を展開する
    w, b = params
    grad_w, grad_b = grads
    
    # 勾配降下法によってパラメータを更新する
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    
    return (new_w, new_b)

def main():
    print("Linear regression module imported!")

    # CSVデータの読み込みと変数代入
    df = pd.read_csv("datasets/insurance.csv").iloc[:, [0, 6]]
    print(df.head())
    x_data: jax.Array = jnp.array(df.iloc[:, 0].to_numpy())
    y_data: jax.Array = jnp.array(df.iloc[:, 1].to_numpy())
    print(x_data)
    print(y_data)

    # 乱数生成キーを作成
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    key_w, key_b = jax.random.split(subkey)

    # 重みとバイアスをランダムに初期化
    w = jax.random.normal(key_w, shape=())
    b = jax.random.normal(key_b, shape=())
    params = (w, b)

    # ハイパーパラメータの設定
    learning_rate = 0.01
    epochs = 100

    # 学習ループ
    print("Starting training...")
    for epoch in range(epochs):
        params = update_params(params, x_data, y_data, learning_rate)
        if (epoch + 1) % 10 == 0:
            current_loss = loss_fn(params, x_data, y_data)
            print(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")

if __name__ == "__main__":
    main()
