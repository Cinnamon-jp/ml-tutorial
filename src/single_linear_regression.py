import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import pandas as pd
import seaborn as sns

def main():
    # CSVデータの読み込みと変数代入
    df = pd.read_csv("datasets/insurance.csv").iloc[:, [0, 6]]
    print(df.head())
    # 入力と出力を (N, 1) の二次元形状に変形
    x_data: jax.Array = jnp.array(df.iloc[:, 0].to_numpy())[:, None]
    y_data: jax.Array = jnp.array(df.iloc[:, 1].to_numpy())[:, None]
    
    # データの標準化 (平均0, 分散1)
    x_mean = x_data.mean()
    x_std = x_data.std()
    x_data = (x_data - x_mean) / x_std
    
    y_mean = y_data.mean()
    y_std = y_data.std()
    y_data = (y_data - y_mean) / y_std
    
    # モデル定義
    class SingleLinearRegression(nnx.Module):
        # 入出力次元の定義
        def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
            self.linear = nnx.Linear(din, dout, rngs=rngs)
            
        # 順伝播の定義
        def __call__(self, x):
            return self.linear(x)
    
    rng = nnx.Rngs(0)
    
    model = SingleLinearRegression(din=1, dout=1, rngs=rng)
    
    tx = optax.sgd(learning_rate=0.1)
    optimizer = nnx.Optimizer(model=model, tx=tx, wrt=nnx.Param)
    
    @nnx.jit
    # 訓練ステップの定義
    def train_step(model, optimizer, x, y) -> float:
        # 損失関数の定義(平均二乗誤差)
        def loss_fn(model) -> float:
            return jnp.mean((model(x) - y) ** 2)
        
        loss, grad = nnx.value_and_grad(loss_fn)(model)
        
        optimizer.update(model, grad)

        return loss
    
    # 訓練ループの実行
    for epoch in range(1, 101):
        loss_value = train_step(model, optimizer, x_data, y_data)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss_value}")

if __name__ == "__main__":
    main()
