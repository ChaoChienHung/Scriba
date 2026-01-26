from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

image = Image.open("handwritten_form.jpg")

input_ids = processor(image, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
text = processor.decode(outputs[0], skip_special_tokens=True)

print(text)
