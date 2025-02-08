"""メモ
正しく動作した
deepseek-apiを使ったチャットボット
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# OpenAIクライアントの初期化（シンプル版）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat_with_bot(user_input):
    try:
        # チャットの完了リクエストを送信
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "あなたは親切で役立つAIアシスタントです。"},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # 応答を返す
        return response.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

def main():
    print("DeepSeekチャットボットへようこそ！")
    print("終了するには 'quit' または 'exit' と入力してください。")
    
    while True:
        user_input = input("\nあなた: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("チャットを終了します。ご利用ありがとうございました！")
            break
            
        response = chat_with_bot(user_input)
        print(f"\nボット: {response}")

if __name__ == "__main__":
    main()
