from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import re

# Define the directory where models will be saved
MODEL_DIR = './saved_model'


# Load the processor and model
def load_model_and_processor():
    load_processor = AutoProcessor.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    load_model = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    # Save them locally
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"Model and processor saved to {MODEL_DIR}")

    return load_processor, load_model


# Generate a reward value (single float) from the model output
def generate_reward_from_image(image_url, processor, model):
    # Process the image and input text
    inputs = processor.process(
        images=[Image.open(requests.get(image_url, stream=True).raw)],
        text=f"""
            You are a reinforcement learning expert assisting with training an agent in the MetaWorld Reacher environment.
            Your task is to assign a single reward value based on the input observation, represented as an image.
            The image depicts the robotic arm and the target position in the 3D workspace.
            Based on the image, you need to assign a reward based on how close the robot's end-effector (arm tip) is to the target,
            how well the arm's joints align with the target, and how efficiently the robot is moving toward or has reached the target.
            A higher reward is given for proximity to the target, with a perfect reward if the end-effector is close to the target.
            A lower reward is assigned if the end-effector is far away or if the robot is moving inefficiently.
            You can optionally penalize excessive movements or deviations from the optimal path.
            The output should be a single numeric value representing the reward, with no extra explanation or text.
        """
    )

    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output with specified configuration
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=10, stop_strings=["<|endoftext|>", " "], temperature=0.1),
        tokenizer=processor.tokenizer
    )

    # Decode the generated tokens
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("generated_text: ", generated_text)

    # Extract the first float from the generated text
    reward = extract_float_from_text(generated_text)
    return reward


# Extract the first float from a given text
def extract_float_from_text(text):
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)  # Regular expression for floats
    if match:
        return float(match.group(0))
    return None


# Example usage
image_url = "https://picsum.photos/id/237/536/354"
processor, model = load_model_and_processor()
generated_reward = generate_reward_from_image(image_url, processor, model)
print("Generated reward:", generated_reward)
