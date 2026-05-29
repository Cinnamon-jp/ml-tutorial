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
class SingleLinearRegression(nnx.Module):
    # 入出力次元の定義
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        
    # 順伝播の定義
    def __call__(self, x):
        return self.linear(x)

@nnx.jit
# 訓練ステップの定義
def train_step(model, optimizer, x, y) -> float:
    # 損失関数の定義(平均二乗誤差)
    def loss_fn(model) -> float:
        return jnp.mean((model(x) - y) ** 2)
    
    loss, grad = nnx.value_and_grad(loss_fn)(model)
    
    optimizer.update(model, grad)

    return loss

def main():
    # CSVデータの読み込みと変数代入
    testdata = pd.read_csv("datasets/testdata.csv")
    # print(testdata.head())
    # 入力と出力を (N, 1) の二次元形状に変形
    x_data: jax.Array = jnp.array(testdata.iloc[:, 0].to_numpy())[:, None]
    y_data: jax.Array = jnp.array(testdata.iloc[:, 1].to_numpy())[:, None]
    
    # データを 平均0, 分散1 に標準化
    x_mean: float = x_data.mean()
    x_std: float = x_data.std()
    x_data = (x_data - x_mean) / x_std  # 配列の要素ごとに実行される
    
    y_mean: float = y_data.mean()
    y_std: float = y_data.std()
    y_data = (y_data - y_mean) / y_std  # 配列の要素ごとに実行される
    
    rng = nnx.Rngs(42)
    
    model = SingleLinearRegression(din=1, dout=1, rngs=rng)
    
    tx = optax.sgd(learning_rate=0.1)
    optimizer = nnx.Optimizer(model=model, tx=tx, wrt=nnx.Param)
    
    # 訓練ループの実行
    for epoch in range(50):
        loss_value = train_step(model, optimizer, x_data, y_data)
        
        if epoch == 0 or epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss_value}")
    
    # 予測値の計算 (標準化スケール)
    y_pred: jax.Array = model(x_data)
    
    # 逆標準化して元のデータスケールに戻す
    x_orig: jax.Array = (x_data * x_std) + x_mean
    y_orig: jax.Array = (y_data * y_std) + y_mean
    y_pred_orig: jax.Array = (y_pred * y_std) + y_mean
    
    # プロット用の DataFrame を作成
    plot_df = pd.DataFrame({
        'x': x_orig.squeeze(),
        'y': y_orig.squeeze(),
        'y_pred': y_pred_orig.squeeze()
    })
    
    # seaborn のテーマを設定
    sns.set_theme(style="darkgrid")
    
    # matplotlib の描画領域を設定 [inch]
    plt.figure(figsize=(10, 6))
    
    # 実際のデータを散布図としてプロット
    sns.scatterplot(
        data=plot_df, 
        x='x', 
        y='y', 
        label='Original Data', 
        alpha=0.6, 
        color='royalblue'
    )
    
    # モデルによる回帰直線をプロット
    plot_df_sorted = plot_df.sort_values(by='x')
    sns.lineplot(
        data=plot_df_sorted, 
        x='x', 
        y='y_pred', 
        label='Regression Line', 
        color='crimson', 
        linewidth=2.5
    )
    
    plt.title('Single Linear Regression Result (Flax NNX)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    
    # 保存ディレクトリの作成と画像の保存
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "single_linear_regression_result.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Regression plot successfully saved to {plot_path}")

if __name__ == "__main__":
    main()
