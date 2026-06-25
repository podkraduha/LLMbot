# Задание №3 - Мини-продукт с LLM-аналитикой

### Цель:
Создать веб-интерфейс, который решает аналитическую задачу с помощью ИИ-агента.
### Реализация: 
Веб-интерфейс: страница Streamlit, куда загружается файл и на его основе выводится 
результат анализа данных, ключевые метрики и графики. 

### Примечание: 
Проект реализует **агентный подход** к анализу данных - LLM пишет Python-код, выполняет его
через **интерпретатор PythonREPLTool**, анализирует резульштаты и формирует готовый отчтет с визуализацией.

При этом для анализа можно использовать **xlsx и csv** файлы.

Также реализована **защита от Prompt Injection**: есть жесткая инструкция в системном промте, которая
запрещает агенту использовать модули os, sys, subprocess, shutil, socket,
выполнять команды open(), eval(), exec(), __import__(), получать доступ к файловой системе или сети,
менять свою роль по запросу пользователя. Еще перед выполнение код проверяется через регулярные выражения
на наличие опасных паттернов, и пользовательский ввод передается как контекст задачи, а не как системная
инструкция, что снижает риск использования агента не для анализа данных.

Для веб-интерфеса используется - Streamlit 1.35.0

Для LLM-фреймворка используется - LangChain 0.2.5

Для интерпретатора кода используется - PythonREPLToll

LLM API - OpenRouter https://openrouter.ai/workspaces/default/keys

Анализ данных - Pandas 2.2.2, NumPy

Визуализация - Matplotlib 3.9.0

Контейнеризация - Docker (docker compose)

Сервер c российского облачного провайдера Beget - http://85.198.101.89:8501/


Установка и запуск c 0:
1. git clone https://github.com/podkraduha/LLMbot.git
2. ssh root@85.198.101.89
3. MySecretPassword123!
4. cd LLMbot
5. docker compose up -d --build
6. http://85.198.101.89:8501/
7. docker compose down - после завершения работы

Использование: 
1. Ввести API-ключ от OpenRouter
2. Загрузить csv или xlsx файл через drag-and-drop или кнопку "Browse files"
3. Описать контекст задачи
4. Нажать кнопку "Запустить анализ"
5. Получить отчет с анализом и графиками

Пример работы: 
Входные данные: датасет games.csv с данными про шахматные партии с lichess
Промт: Найди тренды и аномалии
<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/da58ad0e-a683-450a-861f-19a3cf54ecbb" />

Выполнение работы: агент загружает DataFrame и изучает его структуру, пишет код для анализа, 
выполняет код через интерпретатоор, анализирует результат и формирует отчет.
<img width="1432" height="608" alt="image" src="https://github.com/user-attachments/assets/87da7632-e21f-4f56-9636-0b641c3a1803" />

Результат: отчет, графики 
<img width="1449" height="844" alt="image" src="https://github.com/user-attachments/assets/9fbbdf7d-57c3-4893-a8f3-20fb8d944bfd" />
<img width="1064" height="738" alt="image" src="https://github.com/user-attachments/assets/fe334653-47fe-43d4-b9ed-809dcbfcf33e" />
<img width="895" height="766" alt="image" src="https://github.com/user-attachments/assets/5dd5d821-4903-4b7c-b323-31e62f850b3f" />
<img width="775" height="396" alt="image" src="https://github.com/user-attachments/assets/006a6c12-287b-4af9-b870-9610ec727f27" />


