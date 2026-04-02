from typing import TypedDict, NotRequired

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

import os
import uuid


from pydantic import BaseModel, Field

from src.sentiment_detector.sentiment_detector import SentimentDetector

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

class PointsOfReview(BaseModel):
    good_speed_of_delivery: bool = Field(
        description='Пользователю нравится скорость доставки. Если в тексте нет явного упоминания — ставь false')
    good_price: bool = Field(
        description='Пользователю нравится цена. Если в тексте нет явного упоминания — ставь false')
    good_quality: bool = Field(
        description='Пользователю нравится качество товара. Если в тексте нет явного упоминания — ставь false')
    good_good_looking: bool = Field(
        description='Пользователю нравится дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    good_fit_description: bool = Field(
        description='Пользователю нравится то, что товар соответствует описанию товара. Если в тексте нет явного упоминания — ставь false')
    good_functionality: bool = Field(
        description='Пользователю нравится функциональность товара. Если в тексте нет явного упоминания — ставь false')

    bad_speed_of_delivery: bool = Field(
        description='Пользователя расстраивает скорость доставки. Если в тексте нет явного упоминания — ставь false')
    bad_price: bool = Field(
        description='Пользователя расстраивает цена. Если в тексте нет явного упоминания — ставь false')
    bad_quality: bool = Field(
        description='Пользователя расстраивает качество товара. Если в тексте нет явного упоминания — ставь false')
    bad_good_looking: bool = Field(
        description='Пользователя расстраивает дизайн и внешний вид товара. Если в тексте нет явного упоминания — ставь false')
    bad_fit_description: bool = Field(
        description='Пользователя расстраивает то, что товар отличается от описания товара. Если в тексте нет явного упоминания — ставь false')
    bad_functionality: bool = Field(
        description='Пользователя расстраивает функциональность товара. Если в тексте нет явного упоминания — ставь false')

class ReviewAnalyzer():
    def __init__(self, sentiment_detection_model: str):
        self._sentiment_detector = SentimentDetector(sentiment_detection_model)

        self._llm = ChatOpenAI(
            base_url=os.getenv('BASEURL'),
            model='qwen-3-32b',
            temperature=0.0,
            api_key=os.getenv('APIKEY'))

        self._memory = MemorySaver()

        self._workflow = StateGraph(State)
        self._workflow.add_node('fix_review', self.fix_review_call)
        self._workflow.add_node('sentiment_detection', self.sentiment_detection_call)
        self._workflow.add_node('points_detection', self.points_detection_call)

        self._workflow.add_edge(START, "fix_review")
        self._workflow.add_edge('fix_review', 'sentiment_detection')
        self._workflow.add_edge('sentiment_detection', 'points_detection')
        self._workflow.add_edge('points_detection', END)

        self._graph = self._workflow.compile(checkpointer=self._memory)


    def analyze(self, review: str):
        thread_id = str(uuid.uuid4())

        try:
            config: RunnableConfig = {
                'configurable': {'thread_id': thread_id},
                'callbacks': [LangfuseCallbackHandler()],
            }

            result = self._graph.invoke({'original_review': review}, config=config)
            return result
        finally:
            self._memory.delete_thread(thread_id)



    def fix_review_call(self, state: State):
        if os.getenv("SKIP_REVIEW_FIX") == "true":
            return {
                'fixed_review': state['original_review'],
            }

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
        response = self._llm.invoke([system_message, user_message])

        return {
            'fixed_review': response.content,
        }


    def sentiment_detection_call(self, state: State):
        fixed_review = state.get("fixed_review")
        if not fixed_review:
            raise ValueError("fixed_review missing")

        sentiment = self._sentiment_detector.predict_sentiment([fixed_review])[0]['label']
        return {
            'sentiment': sentiment,
        }


    def points_detection_call(self, state: State):
        system_message = SystemMessage(
            '''Ты должен найти в отзыве то, что нравится автора отзыва и расстраивает его:
            - Скорость доставки
            - Качество товара
            - Цена товара
            - Дизайн и внешний вид товара
            - Функциональность товара
            - Соответствие товара описанию
    
             Учитывай, что каждый из этих пунктов может быть true либо в среди положительных качеств, либо среди отрицательных.
             Но может быть false в обоих случаях.
    
             Ставь true только при явном упоминании, иначе false.''')

        fixed_review = state.get("fixed_review")
        if not fixed_review:
            raise ValueError("fixed_review missing")

        user_message = HumanMessage(f"""Найди то, что нравится пользователю в этом отзыве.
    
                                    <review>{fixed_review}</review>""")
        response = self._llm.with_structured_output(PointsOfReview).invoke([system_message, user_message])

        return {
            'good_speed_of_delivery': response.good_speed_of_delivery,
            'good_price': response.good_price,
            'good_quality': response.good_quality,
            'good_good_looking': response.good_good_looking,
            'good_fit_description': response.good_fit_description,
            'good_functionality': response.good_functionality,

            'bad_speed_of_delivery': response.bad_speed_of_delivery,
            'bad_price': response.bad_price,
            'bad_quality': response.bad_quality,
            'bad_good_looking': response.bad_good_looking,
            'bad_fit_description': response.bad_fit_description,
            'bad_functionality': response.bad_functionality,
        }