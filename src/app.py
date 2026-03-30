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

load_dotenv('../.env', verbose=True)

class State(TypedDict):
    original_review: str
    fixed_review: str

llm = ChatOpenAI(
    base_url=os.getenv('BASEURL'),
    model='qwen-3-32b',
    temperature=0.0,
    api_key=os.getenv('APIKEY'))

memory = MemorySaver()


def fix_review_call(state: State):
    print('Review fix call')
    print(state['original_review'])

workflow = StateGraph(State)
workflow.add_node('fix_review', fix_review_call)

workflow.add_edge(START, "fix_review")
workflow.add_edge('fix_review', END)

app = workflow.compile(checkpointer=memory)

original_review = '''
похоже на подростковое фэнтези, может так и задумывалось. взрослому скучновато, простовато, герой как-будто один и тот же, снова бессмертный, 
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