from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import cv2

# Load model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters for the caption generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_frame(frame):
    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Process the image and generate the caption
    pixel_values = feature_extractor(images=[pil_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Initialize the video capture
cap = cv2.VideoCapture("/dev/video0")  # 0 is usually the default camera
num = 0
current_caption = ""  # Variable to store the last caption

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if num % 60 == 0:
        # Predict the caption for the current frame
        current_caption = predict_frame(frame)
        print(current_caption)
    
    # Display the current caption on the frame
    cv2.putText(frame, current_caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame with the caption
    cv2.imshow('Live Captioning', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    num += 1

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
