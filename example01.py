import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 配置常量 - 可根据实际积分策略调整
DALLE_MODELS = {
    "dall-e-3": {"size_options": ["1024x1024", "1024x1792", "1792x1024"], "cost_factor": 4},
    "dall-e-2": {"size_options": ["256x256", "512x512", "1024x1024"], "cost_factor": 1}
}


def get_ai_response(user_prompt):
    model = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )
    chain = ConversationChain(llm=model, memory=st.session_state['memory'])
    return chain.invoke({'input': user_prompt})['response']


# 优化后的图片生成函数 - 降低成本核心逻辑
def generate_image(prompt):
    # 检查缓存
    if 'image_cache' in st.session_state and prompt in st.session_state['image_cache']:
        st.write("ℹ️ 使用缓存图片以节省积分")
        return st.session_state['image_cache'][prompt]

    client = OpenAI(
        api_key=st.session_state['API_KEY'],
        base_url='https://twapi.openai-hk.com/v1'
    )

    # 根据提示复杂度自动选择模型和尺寸
    model_choice, size_choice = select_optimal_config(prompt)

    try:
        response = client.images.generate(
            model=model_choice,
            prompt=prompt,
            size=size_choice,
            quality="standard",  # 避免使用高成本的hd质量
            n=1,  # 只生成一张图片
        )
        url = response.data[0].url

        # 保存到缓存
        if 'image_cache' not in st.session_state:
            st.session_state['image_cache'] = {}
        st.session_state['image_cache'][prompt] = url

        cost_info = f"💡 成本优化：使用{model_choice}@{size_choice}"
        st.write(cost_info)

        return url

    except Exception as e:
        # 出错时尝试降级配置
        if model_choice == "dall-e-3":
            st.write(f"⚠️ DALL-E 3 失败，尝试降级到 DALL-E 2: {str(e)}")
            return generate_image_with_fallback(prompt, client)
        else:
            raise e


# 根据提示复杂度选择最优配置
def select_optimal_config(prompt):
    # 简单启发式：基于词数和特殊关键词
    words = prompt.lower().split()
    complexity_score = len(words) + ('detailed' in words) * 5 + ('high quality' in words) * 10

    if complexity_score > 20:
        # 复杂提示使用DALL-E 3，但降低分辨率
        return "dall-e-3", "1024x1024"
    else:
        # 简单提示使用DALL-E 2 + 小尺寸
        return "dall-e-2", "256x256"


# 降级策略
def generate_image_with_fallback(prompt, client):
    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="512x512",  # 中等尺寸作为降级选择
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        # 再次降级到最小尺寸
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="256x256",
            quality="standard",
            n=1,
        )
        return response.data[0].url


# 其他代码保持不变...

# 主应用逻辑
st.title('我的ChatGPT')

with st.sidebar:
    api_key = st.text_input('请输入你的Key：', type='password')
    st.session_state['API_KEY'] = api_key

    # 添加成本控制选项
    st.subheader("成本控制")
    use_cache = st.checkbox("启用图片缓存", value=True)
    prefer_low_cost = st.checkbox("优先使用低成本模型", value=True)

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