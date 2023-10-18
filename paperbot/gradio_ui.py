import gradio as gr
import os
import time
from paperbot.bot import PaperBot
from paperbot.datatype import UserAction


class GradioUi:
    def __init__(self):
        self.paper_bot = PaperBot()
        self.sentences = []

    def add_text(self, history, text):
        history = history + [(text, None)]
        return history, gr.Textbox(value="", interactive=False)

    def add_file(self, history, file):
        history = history + [("要約して", "pdfを読み込んでいるよ")]
        self.sentences = self.paper_bot.load_pdf(file.name)
        return history

    def bot(self, history):
        user_action = self.paper_bot.interpret_action(history[-1][0])
        response = ""
        history[-1][1] = ""
        if user_action == None:
            # response = "Sorry, I could not understand your request."
            response = "ごめんね、何をすればいいかわからなかったよ"
        elif user_action == UserAction.SUMMARY_PAPER:
            response = self.paper_bot.summary(
                self.paper_bot.load_string(
                    self.paper_bot.summary_by_sumy(self.sentences)
                ),
                self.paper_bot.detect_language(history[-1][0]),
            )
        elif user_action == UserAction.ANSWER_QUESTION_FROM_PAPER:
            response = self.paper_bot.answer(self.sentences, history[-1][0])
        else:
            response = "Sorry, I could not understand your request."
        for character in response:
            history[-1][1] += character
            time.sleep(0.01)
            yield history


ui = GradioUi()


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.jpeg"))),
        height=600,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Please upload pdf, and enter question about pdf.",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["pdf"])

    txt_msg = txt.submit(ui.add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        ui.bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(ui.add_file, [chatbot, btn], [chatbot], queue=False).then(
        ui.bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
