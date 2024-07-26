import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(
        vector_store_dir,
        OpenAIEmbeddings(
            api_key="sk-fyCJARHrOoAyXf7X2222F820Fb04405e8c89991182BaE5Cf",
            # 代理地址，填写商家中转站或自建OpenAI代理
            base_url="https://api.xiaoai.plus/v1",
        ),
        allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        api_key="sk-tdqfro61NniG2LZd5394E5F28c51419d8b4dE7Db5a6105C4",
        openai_api_base="https://api.xiaoai.plus/v1",
    )

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8},
        ),
    )
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    elif enable_chat:
        return "这个问题我要问问领导"
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="笔记本销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
