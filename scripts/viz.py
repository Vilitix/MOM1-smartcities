import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv('data.csv')

# 数値データの抽出
ph_data = df['pH Test'].dropna()

# 基本統計量の表示
print(ph_data.describe())

# グラフ作成
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(ph_data, kde=True)
plt.title('pH Test Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(x=ph_data)
plt.title('pH Test Boxplot')

plt.tight_layout()
plt.show()