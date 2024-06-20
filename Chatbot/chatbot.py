from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from context_lookup import get_relevant_text
import openai

OPENAI_API_KEY = "your_api_key" # insert your key

openai.api_key = OPENAI_API_KEY

llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

prompt = """
Your job is to use video transcripts to answer questions about mental health and psychology. 
Use the following context to answer questions. Be as detailed as possible, 
but don't make up any information that's not
from the context. If you don't know an answer, say you don't know.

Human: {input}
AI:"""

prompt_template = PromptTemplate(input_variables=["input"], template=prompt)

class ConversationChainWithContext:
    def __init__(self, llm, prompt_template):
        self.llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        self.prompt_template = PromptTemplate(input_variables=["input"], template=prompt)
        self.chain = LLMChain(llm=llm, prompt=prompt_template)
        self.conversation_history = ""

    def chat(self, user_input):
        relevant_text = get_relevant_text(user_input)
        self.conversation_history += f"\nRelevant context: {relevant_text}\nHuman: {user_input}\nAI:"
        response = self.chain.invoke(input=self.conversation_history)
        self.conversation_history += f" {response}"
        return response

if __name__ == "__main__":
    print("Chatbot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = ConversationChainWithContext().chat(user_input)
        print(f"Bot: {response}")