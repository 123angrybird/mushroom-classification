
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

# Initiate the model name and image
#  
# image = files.upload().popitem()[1]
# image = Image.open(BytesIO(image)).convert('RGB') 
# model_name = "microsoft/resnet-50"
model_name = "./fine_tuned_model"
image = Image.open('a.jpg').convert('RGB')

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# get the item with the highest probability then convert to label
predicted_label = logits.argmax(-1).item()

# print(model.config.id2label[predicted_label])
def covert_tensor_to_list(logits,labels):
    mushrooms = []
    for i in range(0,len(logits[0])):
        mushrooms.append({'value': round(logits[0][i].item(),6), 'label': labels[i]})
    return mushrooms

def key(mushrooms):
    return mushrooms['value']

prediction = covert_tensor_to_list(F.softmax(logits,dim=1), model.config.id2label)
prediction.sort(key=key, reverse=True)
for i in range(0,len(prediction)):
    print(prediction[i]['label'], "\r\t\t- ", prediction[i]['value'])
print("\n")


# Save the model and processor
# model.save_pretrained("saved_model")
# image_processor.save_pretrained("saved_model")