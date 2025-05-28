import streamlit as st
import time
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from openai import OpenAI


def get_ai_response(user_prompt):
    model = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    chain = ConversationChain(llm=model, memory=st.session_state['memory'])
    return chain.invoke({'input': user_prompt})['response']


def generate_image(prompt, image_model):
    client = OpenAI(
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    model = "dall-e-3" if image_model == "DALL-E 3（高质量，高成本）" else "dall-e-2"
    size = "1024x1024" if model == "dall-e-3" else "512x512"

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        return f"图片生成失败：{str(e)}"


# 使用 GPT 模型进行拼写检查（替代 spellchecker 库）
def auto_correct_with_gpt(user_input):
    model = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    prompt = f"请修正以下文本的拼写错误：{user_input}"
    corrected = model.invoke({"input": prompt})['response']
    return user_input if corrected == "" else corrected, corrected


st.title('我的ChatGPT')

with st.sidebar:
    api_key = st.text_input('请输入你的Key：', type='password')
    st.session_state['API_KEY'] = api_key
    image_model = st.radio(
        "选择图片生成模型",
        ("DALL-E 3（高质量，高成本）", "DALL-E 2（低质量，低成本）"),
        index=1
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'ai', 'content': '你好主人，我是你的AI助手，我叫小美'}]
    st.session_state['memory'] = ConversationBufferMemory(return_message=True)

for message in st.session_state['messages']:
    role, content = message['role'], message['content']
    if role == 'ai' and content.startswith('![Image]('):
        st.chat_message(role).markdown(content)
    else:
        st.chat_message(role).write(content)

user_input = st.chat_input()
if user_input:
    if not api_key:
        st.info('请输入自己专属的Key！！！')
        st.stop()

    # 使用 GPT 进行拼写检查
    original_input = user_input
    corrected_input, suggestion = auto_correct_with_gpt(user_input)

    if suggestion != original_input:
        st.info(f"你可能想输入：**{suggestion}**")
        user_input = corrected_input

    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role': 'human', 'content': user_input})

    start_time = time.time()
    with st.spinner('AI正在思考，请等待……'):
        if user_input.lower().startswith('/image'):
            image_prompt = user_input[len('/image'):].strip()
            if not image_prompt:
                resp_from_ai = "请提供图片描述，例如：/image 一只微笑的猫咪"
            else:
                image_url = generate_image(image_prompt, image_model)
                resp_from_ai = f"![Image]({image_url})" if "失败" not in image_url else image_url
        else:
            resp_from_ai = get_ai_response(user_input)

    st.chat_message('ai').markdown(resp_from_ai)
    st.session_state['messages'].append({'role': 'ai', 'content': resp_from_ai})

    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        st.warning(
            f"⚠️ 本次响应耗时 {elapsed_time:.2f} 秒，建议：\n"
            "• 检查网络连接\n"
            "• 选择更低成本的图片模型\n"
            "• 简化提问内容"
        )