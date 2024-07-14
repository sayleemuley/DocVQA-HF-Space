from io import BytesIO
from PIL import Image
import gradio as gr
import re
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import AutoProcessor, PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoModelForVision2Seq
from huggingface_hub import InferenceClient
import base64

device = "cuda" if torch.cuda.is_available() else "cpu"

model_choices = [
    "idefics2",
    "paligemma",
    "donut"
]



def load_donut_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    model.to(device)
    return model, processor

def load_paligemma_docvqa():
    # model_id = "google/paligemma-3b-ft-docvqa-896"
    model_id = "google/paligemma-3b-mix-448"
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    return model, processor

def load_idefics_docvqa():
    model_id = "HuggingFaceM4/idefics2-8b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    model.to(device)
    return model, processor

def load_models():
    # load donut
    donut_model, donut_processor = load_donut_model()
    print("donut downloaded")
    # #load paligemma
    pg_model, pg_processor = load_paligemma_docvqa()
    print("paligemma downloaded")
    
    return {"donut":[donut_model, donut_processor],
            "paligemma": [pg_model, pg_processor]
            }

loaded_models = load_models()
print("models loaded")

def base64_encoded_image(image_array):
    im = Image.fromarray(image_array)
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('ascii')
    return image_base64


def inference_calling_idefics(image_array, question):
    model_id = "HuggingFaceM4/idefics2-8b"
    client = InferenceClient(model=model_id)
    image_base64 = base64_encoded_image(image_array)
    image_info = f"data:image/png;base64,{image_base64}"
    prompt = f"![]({image_info}){question}\n\n"
    response = client.text_generation(prompt)
    return response


def process_document_donut(image_array, question):
    model, processor = loaded_models.get("donut")
    
    # prepare encoder inputs
    pixel_values = processor(image_array, return_tensors="pt").pixel_values
    
    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
          
    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    op = processor.token2json(sequence)
    op = op.get("answer", str(op))
    
    return op

def process_document_pg(image_array, question):
    print("qustion :", question)
    print("called loaded model")
    model, processor = loaded_models.get("paligemma")

    print("converting inputs")
    inputs = processor(images=image_array, text=question, return_tensors="pt").to(device)
    print("get predictions")
    predictions = model.generate(**inputs, max_new_tokens=100)
    print("returning decoding")
    return processor.decode(predictions[0], skip_special_tokens=True)[len(question):].lstrip("\n")

def process_document_idf(image_array, question):
    model, processor = loaded_models.get("idefics")

    inputs = processor(images=image_array, text=question, return_tensors="pt") #.to(device)
    predictions = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(predictions[0], skip_special_tokens=True)[len(question):].lstrip("\n")
    

def generate_answer_donut(image_array, question):
    try:
        print("processing document - donut")
        answer = process_document_donut(image_array, question)
        print(answer)
        return answer
    except Exception as e:
        print(e)
        gr.Warning("There is some issue, please try again later.")
        return "sorry :("

def generate_answer_idefics(image_array, question):
    try:
        print("processing document - idf2")
        # answer = process_document_idf(image_array, question)
        answer = inference_calling_idefics(image_array, question)
        print(answer)
        return answer
    except Exception as e:
        print(e)
        gr.Warning("There is some issue, please try again later.")
        return "sorry :("

def generate_answer_paligemma(image_array, question):
    try:
        print("processing document - pg")
        answer = process_document_pg(image_array, question)
        print(answer)
        return answer
    except Exception as e:
        print(e)
        gr.Warning("There is some issue, please try again later.")
        return "sorry :("

def generate_answers(image_path, question, selected_model=model_choices[0]):
    print("selected model: ", selected_model)
    try:
        if selected_model == "donut":
            print("generate answers donut")
            answer = generate_answer_donut(image_path, question)
        elif selected_model == "paligemma":
            print("generate answers pg")
            answer = generate_answer_paligemma(image_path, question) 
        else:
            print("generate answers idf2")
            answer = generate_answer_idefics(image_path, question)
    
        return [answer] #[donut_answer, pg_answer, idf_answer]
    except Exception as e:
        print(e)
        gr.Warning("There is some issue, please try again later.")
        return ["sorry :("]


def greet(name, shame, game):
    return "Hello " + shame + "!!"

INTRO_TEXT = """## VQA demo\n\n
VQA task models comparison 
This space is to compare multiple models on visual document question answering. \n\n
**Note: As the app is running on CPU currently, you might get error if you run multiple models back to back. Please reload the app to get the output.
"""

with gr.Blocks(css="style.css") as demo:
  gr.Markdown(INTRO_TEXT)
#   with gr.Tab("Text Generation"):
  with gr.Column():
    image = gr.Image(label="Input Image")
    question = gr.Text(label="Question")
    selected_model = gr.Radio(model_choices, label="Model", info="Select the model you want to run")

    outputs_answer = gr.Text(label="Answer generated by the selected model")
    run_button = gr.Button()

    inputs = [
        image,
        question,
        selected_model
        ]
    outputs = [
        outputs_answer
    ]
    run_button.click(
        fn=generate_answers,
        inputs=inputs,
        outputs=outputs,
    )
    
    examples = [["images/sample_vendor_contract.png", "Who is agreement between?"],
               ["images/apple-10k-form.png", "What are EMEA revenues in 2017?"], 
               ["images/infographic.png", "What is workforce in UPS?"],
               ]
    gr.Examples(
        examples=examples,
        inputs=inputs,
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)