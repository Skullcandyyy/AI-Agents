from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-chat-v1.0",
    
    task="text-generation",
    pipeline_kwargs= dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
)
chat_model=ChatHuggingFace(llm=llm)
result=chat_model.invoke("What is the capital of India?")
print(result.content)