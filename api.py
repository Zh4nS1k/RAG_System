from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env
load_dotenv()

# Читаем ключ
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("❌ OpenAI API key not found in .env file")

print("✅ OpenAI key loaded successfully")
