from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage 
model = ChatMistralAI(
    model="mistral-small-latest", 
            temperature=0.9
)
print("choose your AI mode")
print("press 1 for angry mode ")
print("press 2 for funny mode ")
print("press 3 for sad mode ")
choice= int(input("enter your choice: "))
if choice == 1:
    mode= "you are an angry AI agent. you resopond aggressively and impatiently"
elif choice == 2:
    mode= "you are a funny AI agent. you respond with humor and jokes"
elif choice == 3:
    mode= "you are a sad AI agent. you respond with empathy and understanding"
messages = [SystemMessage(content=mode)]
print("------------WELCOME I'M CHATBOT TYPE 0 TO EXIT APPLICATIONS----------------")
while True:
    
    prompt = input("you: ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot:", response.content)