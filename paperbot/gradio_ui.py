import gradio as gr
import os
import time
from paperbot.bot import PaperBot
from paperbot.datatype import UserAction


paper_bot = PaperBot()


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    user_action = paper_bot.interpret_action(history[-1][0])
    response = ""
    history[-1][1] = ""
    if user_action == None:
        response = "Sorry, I could not understand your request."
    elif user_action == UserAction.SUMMARY_PAPER:
        response = "Summary paper!"
    elif user_action == UserAction.ANSWER_QUESTION_FROM_PAPER:
        response = "Try answering question"
    else:
        response = "Sorry, I could not understand your request."
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


# def bot(history):
#     print(history[-1][0])
#     history[-1][1] = "fuga"
# user_action = paper_bot.interpret_action(history[-1][0])
# if user_action == None:
#     history[-1][1] = "Sorry, I could not understand your request."
# elif user_action == UserAction.SUMMARY_PAPER:
#     history[-1][1] = "Summary paper!"
# elif user_action == UserAction.ANSWER_QUESTION_FROM_PAPER:
#     history[-1][1] = "Try answering question"
# else:
#     history[-1][1] = "Sorry, I could not understand your request."

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.jpeg"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Please upload pdf, and enter question about pdf.",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["pdf"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
