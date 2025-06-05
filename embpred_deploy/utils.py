import torch
from .models import models 

mapping = {0: "t1", 1: "tPN", 2: "tPNf", 3: "t2", 4: "t3", 5: "t4", 6: "t5", 7: "t6",
    8: "t7",
    9: "t8",
    10: "tM",
    11: "tB",
    12: "tEB"
}

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(model_path, device, num_classes, model_class=None, class_args=None):
    """
    Load a model from a saved checkpoint.

    Parameters:
    - model_path: The path to the checkpoint file.
    - device: The device to load the model onto (e.g., 'cpu' or 'cuda').
    - num_classes: The number of classes in the dataset.
    - model_class: The class of the model to instantiate.
    - class_args: Optional dictionary of additional arguments for the model constructor.

    Returns:
    - model: The loaded model with the state dictionary applied.
    - epoch: The epoch at which the checkpoint was saved.
    - best_val_auc: The best validation AUC at the time the checkpoint was saved.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Prepare additional class arguments if not provided
    if class_args is None:
        class_args = {}
    
    # Instantiate the model with both num_classes and additional class_args
    model = model_class(num_classes=num_classes, **class_args).to(device)
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieve additional information
    epoch = checkpoint['epoch']
    best_val_auc = checkpoint['best_val_auc']
    
    print(f"Model loaded from epoch {epoch} with best validation AUC: {best_val_auc:.4f}")
    
    return model, epoch, best_val_auc


def instantiate_model(class_name: str, *args, **kwargs):
    """
    Dynamically instantiate a model class from models.py given the class name.
    
    Parameters:
        class_name (str): The name of the model class to instantiate.
        *args, **kwargs: Arguments to pass to the model's constructor.
        
    Returns:
        An instance of the specified model class.
    
    Raises:
        ValueError: If the class does not exist in models.py.
    """
    try:
        model_class = getattr(models, class_name)
        return model_class
    except AttributeError:
        raise ValueError(f"Model class '{class_name}' not found in models.py")
        