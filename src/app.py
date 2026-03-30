from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from dotenv import load_dotenv
import os

from fastapi import FastAPI, Form

from pydantic import BaseModel, Field

from src.sentiment_detector.sentiment_detector import SentimentDetector

load_dotenv('.env', verbose=True)

app = FastAPI()

sentiment_detector = SentimentDetector(os.environ.get("SENTIMENT_DETECTION_MODEL"))

class State(TypedDict):
    original_review: str
    fixed_review: str
    sentiment: str

llm = ChatOpenAI(
    base_url=os.getenv('BASEURL'),
    model='qwen-3-32b',
    temperature=0.0,
    api_key=os.getenv('APIKEY'))

memory = MemorySaver()


def fix_review_call(state: State):
    print('Review fix call')
    print(state['original_review'])
    system_message = SystemMessage(content="""
    Ты редактор текстов в издательстве.
    На вход ты получаешь текст. Твоя задача его исправить, ничего не удаляя, и не добавляя.
    
    Ты должен:
    - Исправить синтаксис;
    - Исправить пунктуацию;
    - Исправить орфографические ошибки;
    - Исправить опечатки;
    - Удалить эмодзи;
    - Убрать лишние пробелы;
    - Удали xml/html тэги.
    """)

    user_message = HumanMessage(content=f"""Исправить следующий отзыв о товаре согласно правилам:
    <original_review>{state['original_review']}</original_review>""")
    response = llm.invoke([system_message, user_message])
    print(response.content)

    return {
        'fixed_review': response.content,
    }

def sentiment_detection_call(state: State):
    print('Sentiment detection call')
    sentiment = sentiment_detector.predict_sentiment([state['fixed_review']])[0]['label']
    return {
        'sentiment': sentiment,
    }

class GoodPointsOfReview(BaseModel):
    SpeedOfDelivery: bool = Field(description='Пользователю нравится скорость доставки')
    Price: bool = Field(description='Пользователю нравится цена')
    Quality: bool = Field(description='Пользователю нравится качество товара')
    GoodLooking: bool = Field(description='Пользователю нравится дизайн и внешний вид товара')
    FitDescription: bool = Field(description='Пользователю нравится то, что товар соответствует описанию товара')

def good_points_detection_call(state: State):
    print('Good points detection call')
    system_message = SystemMessage(
        f'''Ты должен найти в отзыве то, что нравится пользователю:
        - Скорость доставки
        - Качество товара
        - Цена товара
        - Дизайн и внешний вид товара
        - Функциональность товара
        - Соответствие товара описанию

         Игнорируй то, что расстраивает пользователя.''')

    user_message = HumanMessage(f"""Найди то, что нравится пользователю в этом отзыве.

                                <review>{state['fixed_review']}</review""")
    response = llm.with_structured_output(GoodPointsOfReview).invoke([system_message, user_message])
    print(response)

class BadPointsOfReview(BaseModel):
    SpeedOfDelivery: bool = Field(description='Пользователя расстраивает скорость доставки')
    Price: bool = Field(description='Пользователя расстраивает цена')
    Quality: bool = Field(description='Пользователя расстраивает качество товара')
    GoodLooking: bool = Field(description='Пользователя расстраивает дизайн и внешний вид товара')
    FitDescription: bool = Field(description='Пользователя расстраивает то, что товар отличается от описания товара')

def bad_points_detection_call(state: State):
    print('Bad points detection call')
    system_message = SystemMessage(
        f'''Ты должен найти в отзыве то, что расстраивает пользователя:
        - Скорость доставки
        - Качество товара
        - Цена товара
        - Дизайн и внешний вид товара
        - Функциональность товара
        - Соответствие товара описанию

         Игнорируй то, что нравится пользователю.''')

    user_message = HumanMessage(f"""Найди то, что расстраивает пользователя в этом отзыве.

                                <review>{state['fixed_review']}</review""")
    response = llm.with_structured_output(BadPointsOfReview).invoke([system_message, user_message])
    print(response)

workflow = StateGraph(State)
workflow.add_node('fix_review', fix_review_call)
workflow.add_node('sentiment_detection', sentiment_detection_call)
workflow.add_node('good_points_detection', good_points_detection_call)
workflow.add_node('bad_points_detection', bad_points_detection_call)


workflow.add_edge(START, "fix_review")
workflow.add_edge('fix_review', 'sentiment_detection')
workflow.add_edge('sentiment_detection', 'good_points_detection')
workflow.add_edge('good_points_detection', 'bad_points_detection')
workflow.add_edge('bad_points_detection', END)


graph = workflow.compile(checkpointer=memory)

original_review = '''
похоже на подростковое фэнтези,может так и      задумывалось. взрослому скучновато, простовато, герой как-будто один и тот же, снова бессмертный, 
снова на нем сошелся "свет клином", да еще и неандертальцы...как эхо романов про Северина Морозова, эхайна. (которые, кстати, написаны отменно). 
ну и фирменное "давайте поделим на тысячу книг и много денежек заработаем", скоро будут каждую главу за деньги продавать. 
несмотря на вышесказанное, вечерок скоротать можно.
'''

original_review = '''
Достоинства:Внешний вид, светит ярко даже от двух ламп,соответствует описанию товара.Недостатки:Брак. Горят только два плафона. Лампы пробовали переставлять - эффект тот же. Электрик сказал, что еще и контакты не в порядке, все зачищал, переделывал что-то.Комментарий:Заморачиваться с возвратом за такие деньги не буду, освещение не основное. Пока так, но досадно.
'''

original_review = '''
отличный, всё супер, спасибо продавцу за оперативную отправку, монитор пришёл раньше обещанной даты, битых пикселей нет, параметры в соответствии с заявленными  
'''

@app.get('/')
def root():
    return {'status': 'ok'}

@app.post('/analyze_review')
def analyze_review(review: str = Form(..., description="Review from marketplace")):
    config: RunnableConfig = {
        'configurable': {'thread_id': 1},
        'callbacks': [LangfuseCallbackHandler()],
    }

    graph.invoke(State({'original_review': review}), config=config)

    state = graph.get_state(config)
    print(state)


def main():
    config: RunnableConfig = {
        'configurable': {'thread_id': 1},
        'callbacks': [LangfuseCallbackHandler()],
    }

    graph.invoke(State({'original_review': original_review}), config=config)

    state = graph.get_state(config)
    print(state)

if __name__ == "__main__":
    main()