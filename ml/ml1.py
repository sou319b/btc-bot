import pandas as pd

# 労働時間のデータを辞書形式で作成
data = {
  'Aさんの労働時間' : [150, 155], # Aさんの労働時間列の作成
  'Bさんの労働時間' : [162, 170] # Bさんの労働時間列の作成
}

# 辞書データをDataFrameに変換
df = pd.DataFrame(data)
df # DataFrameの表示

#インタプリタの場合下記を実行
#print(df)

# DataFrameの型を確認 
print(type(df))

# DataFrameの形状を確認
df.shape

df.index = ['4月', '5月'] # dfのインデックスを変更
df # 表示

df.columns = ['Aさんの労働(h)', 'Bさんの労働(h)'] # 列名の変更
df # 表示