from typing import TypedDict, NotRequired

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from dotenv import load_dotenv
import os
import uuid

from fastapi import FastAPI, Form

from pydantic import BaseModel, Field

from src.sentiment_detector.sentiment_detector import SentimentDetector

load_dotenv('.env')

app = FastAPI()

sentiment_detector = SentimentDetector(os.environ.get("SENTIMENT_DETECTION_MODEL"))

class State(TypedDict):
    original_review: str
    fixed_review: NotRequired[str]
    sentiment: NotRequired[str]

    good_speed_of_delivery: NotRequired[bool]
    good_price: NotRequired[bool]
    good_quality: NotRequired[bool]
    good_good_looking: NotRequired[bool]
    good_fit_description: NotRequired[bool]
    good_functionality: NotRequired[bool]

    bad_speed_of_delivery: NotRequired[bool]
    bad_price: NotRequired[bool]
    bad_quality: NotRequired[bool]
    bad_good_looking: NotRequired[bool]
    bad_fit_description: NotRequired[bool]
    bad_functionality: NotRequired[bool]

llm = ChatOpenAI(
    base_url=os.getenv('BASEURL'),
    model='qwen-3-32b',
    temperature=0.0,
    api_key=os.getenv('APIKEY'))

memory = MemorySaver()


def fix_review_call(state: State):
    system_message = SystemMessage(content="""
    Ты редактор текстов в издательстве.
    На вход ты получаешь текст. Твоя задача его исправить, не удаляй, и не добавляй смысловые фрагменты.
    
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

    return {
        'fixed_review': response.content,
    }

def sentiment_detection_call(state: State):
    fixed_review = state.get("fixed_review")
    if not fixed_review:
        raise ValueError("fixed_review missing")

    sentiment = sentiment_detector.predict_sentiment([fixed_review])[0]['label']
    return {
        'sentiment': sentiment,
    }

class GoodPointsOfReview(BaseModel):
    speed_of_delivery: bool = Field(description='Пользователю нравится скорость доставки. Если в тексте нет явного упоминания — ставь false')
    price: bool = Field(description='Пользователю нравится цена. Если в тексте нет явного упоминания — ставь false')
    quality: bool = Field(description='Пользователю нравится качество товара. Если в тексте нет явного упоминания — ставь false')
    good_looking: bool = Field(description='Пользователю нравится дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    fit_description: bool = Field(description='Пользователю нравится то, что товар соответствует описанию товара. Если в тексте нет явного упоминания — ставь false')
    functionality: bool = Field(description='Пользователю нравится функциональность товара. Если в тексте нет явного упоминания — ставь false')

def good_points_detection_call(state: State):
    system_message = SystemMessage(
        '''Ты должен найти в отзыве то, что нравится пользователю:
        - Скорость доставки
        - Качество товара
        - Цена товара
        - Дизайн и внешний вид товара
        - Функциональность товара
        - Соответствие товара описанию

         Игнорируй то, что расстраивает пользователя. Ставь true только при явном упоминании, иначе false.''')

    fixed_review = state.get("fixed_review")
    if not fixed_review:
        raise ValueError("fixed_review missing")

    user_message = HumanMessage(f"""Найди то, что нравится пользователю в этом отзыве.

                                <review>{fixed_review}</review>""")
    response = llm.with_structured_output(GoodPointsOfReview).invoke([system_message, user_message])

    return {
        'good_speed_of_delivery': response.speed_of_delivery,
        'good_price': response.price,
        'good_quality': response.quality,
        'good_good_looking': response.good_looking,
        'good_fit_description': response.fit_description,
        'good_functionality': response.functionality,
    }

class BadPointsOfReview(BaseModel):
    speed_of_delivery: bool = Field(description='Пользователя расстраивает скорость доставки. Если в тексте нет явного упоминания — ставь false')
    price: bool = Field(description='Пользователя расстраивает цена. Если в тексте нет явного упоминания — ставь false')
    quality: bool = Field(description='Пользователя расстраивает качество товара. Если в тексте нет явного упоминания — ставь false')
    good_looking: bool = Field(description='Пользователя расстраивает дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    fit_description: bool = Field(description='Пользователя расстраивает то, что товар отличается от описания товара. Если в тексте нет явного упоминания — ставь false')
    functionality: bool = Field(description='Пользователя расстраивает функциональность товара. Если в тексте нет явного упоминания — ставь false')

def bad_points_detection_call(state: State):
    system_message = SystemMessage(
        '''Ты должен найти в отзыве то, что расстраивает пользователя:
        - Скорость доставки
        - Качество товара
        - Цена товара
        - Дизайн и внешний вид товара
        - Функциональность товара
        - Соответствие товара описанию

         Игнорируй то, что нравится пользователю. Ставь true только при явном упоминании, иначе false.''')

    fixed_review = state.get("fixed_review")
    if not fixed_review:
        raise ValueError("fixed_review missing")

    user_message = HumanMessage(f"""Найди то, что расстраивает пользователя в этом отзыве.

                                <review>{fixed_review}</review>""")
    response = llm.with_structured_output(BadPointsOfReview).invoke([system_message, user_message])

    return {
        'bad_speed_of_delivery': response.speed_of_delivery,
        'bad_price': response.price,
        'bad_quality': response.quality,
        'bad_good_looking': response.good_looking,
        'bad_fit_description': response.fit_description,
        'bad_functionality': response.functionality,
    }

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
    thread_id = str(uuid.uuid4())

    config: RunnableConfig = {
        'configurable': {'thread_id': thread_id},
        'callbacks': [LangfuseCallbackHandler()],
    }

    try:
        result = graph.invoke({'original_review': review}, config=config)

        return {
            'original_review': result.get('original_review'),
            'fixed_review': result.get('fixed_review'),
            'sentiment': result.get('sentiment'),
            'thread_id': thread_id,
            'good_points': {
                'speed_of_delivery': result.get('good_speed_of_delivery'),
                'price': result.get('good_price'),
                'quality': result.get('good_quality'),
                'good_looking': result.get('good_good_looking'),
                'fit_description': result.get('good_fit_description'),
                'functionality': result.get('good_functionality'),
            },
            'bad_points': {
                'speed_of_delivery': result.get('bad_speed_of_delivery'),
                'price': result.get('bad_price'),
                'quality': result.get('bad_quality'),
                'good_looking': result.get('bad_good_looking'),
                'fit_description': result.get('bad_fit_description'),
                'functionality': result.get('bad_functionality'),
            }
        }
    finally:
        memory.delete_thread(thread_id)


def main():
    thread_id = str(uuid.uuid4())

    config: RunnableConfig = {
        'configurable': {'thread_id': thread_id},
        'callbacks': [LangfuseCallbackHandler()],
    }

    try:
        graph.invoke({'original_review': original_review}, config=config)

        state = graph.get_state(config)
        print(state)
    finally:
        memory.delete_thread(thread_id)

if __name__ == "__main__":
    main()