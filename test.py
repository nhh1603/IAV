# import statements for python, torch and companion libraries and your own modules
# TIP: use the python standard json module to write python dictionaries as JSON files
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import sigmoid
from dataset import COCOTestImageDataset
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


# global variables defining inference hyper-parameters among other things 
# DON'T forget the multi-task classification probability threshold
minibatch_size = 64
threshold = 0.5
# mean = np.array([0.5, 0.5, 0.5])
# std = np.array([0.25, 0.25, 0.25])

# data, trained model and output directories/filenames initialization
test_dir = "./ms-coco/images/test-resized"
trained_model = "./trained_models/best_model_resnet18_lastblock.pth"
output_json = "./predictions/predictions_resnet18_lastblock.json"

# device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# instantiation of transforms, dataset and data loader
transform = transforms.Compose([transforms.ToTensor(),
                                ResNet18_Weights.DEFAULT.transforms(antialias=True)])

test_dataset = COCOTestImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)

# load network model from saved file
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 80)
model.load_state_dict(torch.load(trained_model))
model.to(device)
model.eval()

# initialize output dictionary
output_dict = {}

# prediction loop over test_loader
#    get mini-batch
#    compute network output
#    threshold network output
#    update dictionary entries write corresponding class indices
with torch.no_grad():
    for data in test_loader:
        images, names = data[0].to(device), data[1]

        outputs = model(images)
        outputs = sigmoid(outputs)

        predictions = (outputs > threshold).int()

        for name, prediction in zip(names, predictions):
            predicted_indices = torch.nonzero(prediction).squeeze().tolist()
            if isinstance(predicted_indices, int):  # Handle single-class case
                predicted_indices = [predicted_indices]

            output_dict[name] = predicted_indices

# write JSON file
with open(output_json, "w") as f:
    json.dump(output_dict, f, indent=4)