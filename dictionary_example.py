# 辞書の作成
user_info = {
    "名前": "田中",
    "年齢": 25,
    "趣味": ["読書", "サッカー"]
}

# 値の取得
print("基本的な値の取得:")
print(f"名前: {user_info['名前']}")
print(f"年齢: {user_info['年齢']}")
print(f"趣味: {user_info['趣味']}")

# 値の追加と変更
print("\n値の追加と変更:")
user_info["職業"] = "エンジニア"  # 新しいキーと値の追加
user_info["年齢"] = 26  # 既存の値の変更
print(user_info)

# 値の削除
print("\n値の削除:")
del user_info["趣味"]
print(user_info)

# 辞書のメソッド
print("\n辞書のメソッド:")
print(f"全てのキー: {user_info.keys()}")
print(f"全ての値: {user_info.values()}")
print(f"全てのキーと値のペア: {user_info.items()}")

# 辞書の中身確認
print("\n辞書の中身確認:")
if "名前" in user_info:
    print("名前のキーが存在します")

# 辞書内包表記の例
print("\n辞書内包表記:")
数字_の二乗 = {x: x**2 for x in range(5)}
print(数字_の二乗) 