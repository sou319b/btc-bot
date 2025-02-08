# どちらも同じ結果になります
str1 = "こんにちは"
str2 = 'こんにちは'
print(f"str1: {str1}")
print(f"str2: {str2}")
print(f"str1 == str2: {str1 == str2}")

# クォートを含む文字列の例
text1 = "彼は'こんにちは'と言いました"  # シングルクォートを含む場合
text2 = '彼は"さようなら"と言いました'  # ダブルクォートを含む場合
print(f"\ntext1: {text1}")
print(f"text2: {text2}")

# 三重クォートの例（複数行の文字列）
multi_line = """これは
複数行の
文字列です"""
print(f"\n複数行の文字列:")
print(multi_line) 