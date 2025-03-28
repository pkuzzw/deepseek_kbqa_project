from openai import OpenAI
client = OpenAI(api_key="sk-ooscmypgomjmlrzejjhcidzhzegvmirvvweonaoacfmpcrrc", 
                base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
        # {
        #     "role": "system",
        #     "content": "You are an AI assistant who knows everything.",
        # },
        # {
        #     "role": "user",
        #     "content": "Tell me, When did apple release iphone4?"
        # },
        {"role": "system", "content": "你是一个知识问答助手，根据提供的背景信息回答问题，回答范围限制在背景信息提供的内容中。"},
        {"role": "user", "content": f"背景信息：苹果是一种常见的水果，富含维生素 C。"},
        {"role": "user", "content": "苹果有什么营养成分？"}
    ],
)

message = response.choices[0].message.content

print(f"Assistant: {message}")