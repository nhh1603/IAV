# import statements for python, torch and companion libraries and your own modules
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from dataset import COCOTrainImageDataset, COCOTestImageDataset
from loop import train_loop, validation_loop


# global variables defining training hyper-parameters among other things
learning_rate=0.001
epochs = 5
minibatch_size = 64
num_workers = 4

# device initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

# data directories initialization
train_dir = "./ms-coco/images/train-resized"
test_dir = "./ms-coco/images/test-resized"
labels_dir = "./ms-coco/labels/train"

# instantiation of transforms, datasets and data loaders
# TIP : use torch.utils.data.random_split to split the training set into train and validation subsets
transform = transforms.Compose([transforms.ToTensor(),
                                ResNet18_Weights.DEFAULT.transforms(antialias=True)])

full_dataset = COCOTrainImageDataset(train_dir, labels_dir, transform=transform)
test_dataset = COCOTestImageDataset(test_dir, transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=minibatch_size,
                                           shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=minibatch_size,
                                         shuffle=False)

# class definitions
classes = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",       
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", 
           "hair drier", "toothbrush")
num_classes = len(classes)

# instantiation and preparation of network model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# instantiation of loss criterion
criterion = nn.BCEWithLogitsLoss()

# instantiation of optimizer, registration of network parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# definition of current best model path
# initialization of model selection metric
best_f1 = 0

# creation of tensorboard SummaryWriter (optional)

# epochs loop:
#   train
#   validate on train set
#   validate on validation set
#   update graphs (optional)
#   is new model better than current model ?
#       save it, update current best metric
if __name__ == '__main__':  
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Training phase
        train_results = train_loop(train_loader, model, criterion, optimizer, device)
        print("Train finished")
        
        # Validation phase on validation set
        val_results = validation_loop(val_loader, model, criterion, num_classes, device, one_hot=True)
        print(f"Validation set results: {val_results}")
        # Update graphs (if you have a graphing system in place, you can add code here to log or update graphs)
        # Example: update your training/validation loss and F1 score in a graphing tool like TensorBoard or Matplotlib
        # Check if the current model is the best based on validation F1
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            torch.save(model.state_dict(), "best_model.pth")  # Save the model if it's the best so far
            print("New best model saved with F1:", best_f1)
        # Logging metrics
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_val_results['loss']:.4f}, "
        #     f"Train F1: {train_val_results['f1']:.4f}, "
        #     f"Val Loss: {val_results['loss']:.4f}, Val F1: {val_results['f1']:.4f}")
    print("Training complete. Best F1 on validation set:", best_f1)
        
    
# close tensorboard SummaryWriter if created (optional)