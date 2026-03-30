from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from dotenv import load_dotenv
import os

from pydantic import BaseModel, Field

from src.sentiment_detector.sentiment_detector import SentimentDetector

load_dotenv('../.env', verbose=True)

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


workflow = StateGraph(State)
workflow.add_node('fix_review', fix_review_call)
workflow.add_node('sentiment_detection', sentiment_detection_call)

workflow.add_edge(START, "fix_review")
workflow.add_edge('fix_review', 'sentiment_detection')
workflow.add_edge('sentiment_detection', END)


app = workflow.compile(checkpointer=memory)

original_review = '''
похоже на подростковое фэнтези,может так и      задумывалось. взрослому скучновато, простовато, герой как-будто один и тот же, снова бессмертный, 
снова на нем сошелся "свет клином", да еще и неандертальцы...как эхо романов про Северина Морозова, эхайна. (которые, кстати, написаны отменно). 
ну и фирменное "давайте поделим на тысячу книг и много денежек заработаем", скоро будут каждую главу за деньги продавать. 
несмотря на вышесказанное, вечерок скоротать можно.
'''

def main():
    config: RunnableConfig = {
        'configurable': {'thread_id': 1},
        'callbacks': [LangfuseCallbackHandler()],
    }

    app.invoke(State({'original_review': original_review}), config=config)

    state = app.get_state(config)
    print(state)

if __name__ == "__main__":
    main()