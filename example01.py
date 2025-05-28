import streamlit as st
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


def generate_image(prompt):
    client = OpenAI(
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


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
    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role': 'human', 'content': user_input})

    with st.spinner('AI正在思考，请等待……'):
        if user_input.lower().startswith('/image'):
            # 提取图片描述文本
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

        st.session_state['history'] = resp_from_ai
        st.chat_message('ai').markdown(resp_from_ai)
        st.session_state['messages'].append({'role': 'ai', 'content': resp_from_ai})