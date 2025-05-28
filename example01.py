import streamlit as st
import time
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from openai import OpenAI
from spellchecker import SpellChecker

spell = SpellChecker()


def get_ai_response(user_prompt):
    model = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    chain = ConversationChain(llm=model, memory=st.session_state['memory'])
    return chain.invoke({'input': user_prompt})['response']


st.title('我的ChatGPT')

with st.sidebar:
    api_key = st.text_input('请输入你的Key：', type='password')
    st.session_state['API_KEY'] = api_key

    # 新增模型选择开关
    image_model = st.radio(
        "选择图片生成模型",
        ("DALL-E 3（高质量，高成本）", "DALL-E 2（低质量，低成本）"),
        index=1  # 默认选择 DALL-E 2
    )


def generate_image(prompt):
    client = OpenAI(
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )

    # 根据用户选择动态切换模型和尺寸
    if image_model == "DALL-E 3（高质量，高成本）":
        model = "dall-e-3"
        size = "1024x1024"
    else:
        model = "dall-e-2"
        size = "512x512"

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )
    return response.data[0].url


def auto_correct(user_input):
    words = user_input.split()
    misspelled = spell.unknown(words)
    if misspelled:
        corrected_words = [spell.correction(word) for word in words]
        corrected_input = ' '.join(corrected_words)
        return f"你可能想输入：**{corrected_input}**", corrected_input
    else:
        return None, user_input


st.title('我的ChatGPT')

with st.sidebar:
    api_key = st.text_input('请输入你的Key：', type='password')
    st.session_state['API_KEY'] = api_key

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

    correction_msg, corrected_input = auto_correct(user_input)
    if correction_msg:
        st.info(correction_msg)
        user_input = corrected_input

    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role': 'human', 'content': user_input})

    # 性能监控开始时间
    start_time = time.time()

    with st.spinner('AI正在思考，请等待……'):
        if user_input.lower().startswith('/image'):
            image_prompt = user_input[len('/image'):].strip()
            if not image_prompt:
                resp_from_ai = "请提供图片描述，例如：/image 一只微笑的猫咪"
            else:
                try:
                    image_url = generate_image(image_prompt)
                    resp_from_ai = f"![Image]({image_url})"
                except Exception as e:
                    resp_from_ai = f"图片生成失败：{str(e)}"
        else:
            resp_from_ai = get_ai_response(user_input)

    # 计算响应时间
    elapsed_time = time.time() - start_time

    # 性能优化提示逻辑
    if elapsed_time > 10:  # 超过10秒视为响应缓慢
        st.warning(
            f"⚠️ 本次响应耗时 {elapsed_time:.2f} 秒，可能由于以下原因：\n"
            "1. 网络连接不稳定\n"
            "2. 模型计算复杂度高（如生成图片）\n"
            "3. 同时运行程序过多\n\n"
            "建议：\n"
            "• 检查网络连接\n"
            "• 降低图片生成尺寸（如使用 /image 小尺寸描述）\n"
            "• 关闭其他占用资源的程序"
        )

    st.session_state['history'] = resp_from_ai
    st.chat_message('ai').markdown(resp_from_ai)
    st.session_state['messages'].append({'role': 'ai', 'content': resp_from_ai})