import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLineEdit, QLabel, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from run_pplm_re import run_pplm_example

class TextGenerationThread(QThread):
    text_generated = pyqtSignal(str)
    generation_finished = pyqtSignal()

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            generated_text = ""
            generated_texts = run_pplm_example(**self.params)
            for i, text in enumerate(generated_texts):
                generated_text += f"= Perturbed generated text {i+1} =\n{text}\n\n"
            self.text_generated.emit(generated_text)
        except Exception as e:
            self.text_generated.emit("Error during text generation: " + str(e))
        finally:
            self.generation_finished.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'PPLM Text Generation'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.parameter_inputs = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        layout = QVBoxLayout()

        self.input_text = QTextEdit(self)
        self.input_text.setPlaceholderText("Enter your text here...")
        layout.addWidget(self.input_text)

        self.add_parameter_input(layout, "Bag of words:", "military")
        self.add_parameter_input(layout, "Discrim:", "sentiment")
        self.add_parameter_input(layout, "Class label:", "very_negative")
        self.add_parameter_input(layout, "Length:", "50")
        self.add_parameter_input(layout, "Step size:", "0.03")
        self.add_parameter_input(layout, "Num. samples:", "1")
        self.add_parameter_input(layout, "Num. iterations:", "3")
        self.add_parameter_input(layout, "Window length:", "5")
        self.add_parameter_input(layout, "Gamma:", "1.5")
        self.add_parameter_input(layout, "GM scale:", "0.95")
        self.add_parameter_input(layout, "KL scale:", "0.1")

        self.generate_button = QPushButton('Generate Text', self)
        self.generate_button.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_button)

        self.output_text = QTextEdit(self)
        self.output_text.setPlaceholderText("Generated text will appear here...")
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.show()

    def add_parameter_input(self, layout, label_text, default_value):
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        h_layout.addWidget(label)
        line_edit = QLineEdit(default_value)
        h_layout.addWidget(line_edit)
        layout.addLayout(h_layout)
        self.parameter_inputs[label_text.replace(".", "").replace(":", "").replace(" ", "_").lower()] = line_edit

    def on_generate(self):
        self.generate_button.setEnabled(False)
        QMessageBox.information(self, "Text Generation", "Generating text. Please wait...")
        params = {
            'cond_text': str(self.input_text.toPlainText()),
            'bag_of_words': self.parameter_inputs['bag_of_words'].text(),
            'discrim': self.parameter_inputs['discrim'].text(),
            'class_label': self.parameter_inputs['class_label'].text(),
            'length': int(self.parameter_inputs['length'].text()),
            'stepsize': float(self.parameter_inputs['step_size'].text()),
            'num_samples': int(self.parameter_inputs['num_samples'].text()),
            'num_iterations': int(self.parameter_inputs['num_iterations'].text()),
            'window_length': int(self.parameter_inputs['window_length'].text()),
            'gamma': float(self.parameter_inputs['gamma'].text()),
            'gm_scale': float(self.parameter_inputs['gm_scale'].text()),
            'kl_scale': float(self.parameter_inputs['kl_scale'].text()),
            'sample': True,
            'colorama': True,
            'verbosity': 'quiet'
        }

        self.text_generation_thread = TextGenerationThread(params)
        self.text_generation_thread.text_generated.connect(self.update_output_text)
        self.text_generation_thread.generation_finished.connect(self.on_generation_finished)
        self.text_generation_thread.start()

    def update_output_text(self, text):
        self.output_text.setPlainText(text)

    def on_generation_finished(self):
        QMessageBox.information(self, "Text Generation", "Text generation finished.")
        self.generate_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
