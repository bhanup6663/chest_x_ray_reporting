import json
import os
from PIL import Image, ImageDraw
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Function to preprocess bounding box data
def preprocess_box(box, image_width, image_height):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    position_x = x_min / image_width
    position_y = y_min / image_height
    box_center_x = (x_min + x_max) / 2
    box_center_y = (y_min + y_max) / 2
    
    return {
        "width": width,
        "height": height,
        "position_x": position_x,
        "position_y": position_y,
        "center_x": box_center_x,
        "center_y": box_center_y
    }

# Function to generate a prompt
def generate_prompt(image_id, box_data, image_width, image_height):
    prompt = f"Analyze the X-ray image with ID: {image_id}. The image contains the following findings:\n"
    for entry in box_data:
        box_info = preprocess_box(entry['box'], image_width, image_height)
        prompt += (
            f"- {entry['class_label']} with a bounding box of width {box_info['width']} pixels, "
            f"height {box_info['height']} pixels, located at approximately "
            f"({box_info['position_x']:.2f}, {box_info['position_y']:.2f}) relative to the image size. "
            f"The center of the box is at ({box_info['center_x']}, {box_info['center_y']}).\n"
        )
    return prompt

# Function to resize image and add bounding boxes
def process_image(image_path, box_data):
    # Load and resize image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    for entry in box_data:
        box = entry['box']
        draw.rectangle(box, outline="red", width=3)
    
    return image

# Function to generate report using Langchain and an LLM
def generate_report_with_langchain(prompt):
    # Set up the LLM using langchain with HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
    model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
    pipe = torch.nn.Sequential(tokenizer, model)
    
    # Create the LLMChain
    llm = HuggingFacePipeline(pipeline=pipe)
    template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    chain = LLMChain(llm=llm, prompt=template)
    
    # Generate the report
    report = chain.run({"prompt": prompt})
    return report
