from flask import Flask, render_template, request
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os
import uuid
from datetime import datetime

app = Flask(__name__)

# ==============================
# CONFIGURATION
# ==============================

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

# ==============================
# LOAD AI MODEL
# ==============================

print("========================================")
print("Loading AI model...")
print("========================================")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

print("AI model loaded successfully!")
print("========================================")


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_image(image_path):

    print()
    print("Starting prediction...")
    print("Image:", image_path)

    image = Image.open(image_path).convert("RGB")

    prompt = """
Look carefully at this image.

Identify the MAIN SUBJECT or OBJECT.

Give ONLY a short and simple name.

Examples:

Shark
Dog
Car
Bus
Smartphone
Building
Mountain
Ocean
Pizza
Tree
Person
Statue

If the image contains a religious idol or statue,
identify the specific subject if you can.

Do not describe the image.
Do not give a sentence.
Do not give an explanation.

Return ONLY the main subject name.

If you genuinely cannot identify the main subject,
return:

Unknown
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Create prompt
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Process image
    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt"
    )

    # AI inference
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    # Remove prompt tokens
    generated = outputs[
        :,
        inputs["input_ids"].shape[-1]:
    ]

    # Decode result
    result = processor.batch_decode(
        generated,
        skip_special_tokens=True
    )[0].strip()

    # Clean result
    result = result.replace("\n", " ").strip()

    result = result.replace(
        "Prediction:",
        ""
    ).strip()

    result = result.rstrip(
        ".!?,:;"
    ).strip()

    if not result:
        result = "Unknown"

    print("Prediction:", result)

    return result


# ==============================
# FILE VALIDATION
# ==============================

def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ==============================
# HOME ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    image_path = None
    error = None
    image_name = None
    prediction_time = None

    if request.method == "POST":

        # Check file field
        if "image" not in request.files:

            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                image_name=image_name,
                prediction_time=prediction_time,
                error=error
            )

        file = request.files["image"]

        # Empty filename
        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                image_name=image_name,
                prediction_time=prediction_time,
                error=error
            )

        # Validate extension
        if not allowed_file(file.filename):

            error = (
                "Invalid file type. "
                "Please upload JPG, JPEG, PNG or WEBP."
            )

            return render_template(
                "index.html",
                prediction=prediction,
                image_path=image_path,
                image_name=image_name,
                prediction_time=prediction_time,
                error=error
            )

        # Generate unique filename
        extension = os.path.splitext(
            file.filename
        )[1].lower()

        filename = (
            str(uuid.uuid4()) + extension
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Save image
        file.save(filepath)

        print("Image saved:", filepath)

        try:

            # Predict
            prediction = predict_image(
                filepath
            )

            # Browser path
            image_path = "/" + filepath.replace(
                "\\",
                "/"
            )

            image_name = file.filename

            prediction_time = datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )

        except Exception as e:

            print("ERROR:", e)

            error = (
                "Something went wrong while "
                "analyzing the image."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        image_name=image_name,
        prediction_time=prediction_time,
        error=error
    )


# ==============================
# START FLASK
# ==============================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=5000
    )