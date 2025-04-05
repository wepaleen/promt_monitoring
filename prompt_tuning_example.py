import os
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
import time
import json

# Загрузка переменных окружения
load_dotenv()

# Инициализация клиентов
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

def create_prompt_v1(query):
    """Базовая версия промпта"""
    return f"""
    Пожалуйста, ответь на следующий вопрос максимально подробно и информативно.
    Вопрос: {query}
    """

def create_prompt_v2(query):
    """Оптимизированная версия промпта"""
    return f"""
    Контекст: Ты - эксперт в области технологий.
    
    Правила:
    1. Начни с краткого ответа (1-2 предложения)
    2. Приведи 2-3 конкретных примера
    3. Заверши практическим советом
    
    Вопрос: {query}
    """

def test_prompt(prompt_func, query, version):
    """Тестирование промпта с измерением метрик"""
    start_time = time.time()
    
    # Создаем trace в Langfuse
    trace = langfuse.trace(
        name=f"prompt_test_v{version}",
        input={"query": query}
    )
    
    try:
        # Создаем промпт
        prompt = prompt_func(query)
        
        # Вызываем OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Получаем ответ и метрики
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        execution_time = time.time() - start_time
        
        # Завершаем trace с результатами
        trace.end(
            output={
                "answer": answer,
                "tokens_used": tokens_used,
                "execution_time": execution_time
            }
        )
        
        return {
            "version": version,
            "answer": answer,
            "tokens_used": tokens_used,
            "execution_time": execution_time
        }
        
    except Exception as e:
        trace.end(error=str(e))
        return {"error": str(e)}

def compare_prompts(query):
    """Сравнение разных версий промптов"""
    results = []
    
    # Тестируем базовую версию
    results.append(test_prompt(create_prompt_v1, query, 1))
    
    # Тестируем оптимизированную версию
    results.append(test_prompt(create_prompt_v2, query, 2))
    
    # Выводим результаты
    print("\nРезультаты сравнения:")
    print("=" * 50)
    for result in results:
        if "error" not in result:
            print(f"\nВерсия {result['version']}:")
            print(f"Время выполнения: {result['execution_time']:.2f} сек")
            print(f"Использовано токенов: {result['tokens_used']}")
            print(f"Ответ: {result['answer'][:200]}...")
        else:
            print(f"\nОшибка в версии {result['version']}: {result['error']}")
    
    return results

if __name__ == "__main__":
    # Пример запроса для тестирования
    test_query = "Какие преимущества у Python перед другими языками программирования?"
    
    # Запускаем сравнение
    results = compare_prompts(test_query)
    
    # Сохраняем результаты в файл
    with open("prompt_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2) 