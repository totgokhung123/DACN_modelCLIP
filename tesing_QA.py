from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# Load model và processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
).cuda()
model.eval()

# Load frame ảnh
image = Image.open("2906.jpg")  # Frame từ truy vấn CLIP

# Câu hỏi
question = "How many people are in the image?"

# Xử lý input
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)

# Trả lời
generated_ids = model.generate(**inputs)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Answer:", answer)