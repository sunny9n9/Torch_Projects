from .imports import torch

__all__ = ['IntersectionOverUnion', 'DiceLoss']

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def forward(self, label, pred):
        # dice score

        # Handle the deep supervision list For UNET++, as we return list of outuput(multiple output from different layers) 
        if isinstance(pred, list):
            # currently taking average of all outputs, may use a weighted average for more tuning
            return sum(self.forward(p, label) for p in pred) / len(pred)
        
        pred_probab = torch.sigmoid(pred)
        pred_intersection = torch.mul(label, pred_probab)
        total_pred_intersection = torch.sum(pred_intersection)
        total_label_acutal_and_predicted = torch.sum(pred_probab) + torch.sum(label)

        dice_score = (2*total_pred_intersection)/(total_label_acutal_and_predicted + self.epsilon)
        dice_loss = 1 - dice_score

        return dice_loss
    
class IntersectionOverUnion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.epilon = 1e-6
        
    def forward(self, actual, prediction):
        intersection = torch.sigmoid(prediction) * actual
        total_overlap = torch.sigmoid(prediction) + actual
        union = total_overlap - intersection
        IoU =  intersection / (union + self.epilon)
        return 1 - IoU

class Mixed_Dice_Sigmoid(torch.nn.Module):
    def __init__(self, dice_weight = 0.5):
        super().__init__()
        self.epsilon = 1e-6
        self.dice_weight = 0.5

    def forward(self, actual, predicted_all):
        accumulated_loss = 0 
        actual_flat = actual.view(actual.size(0), -1)
        def _dice_score(actual_flat, pred_flat):
            intersection = torch.mul(pred_flat, actual_flat)
            intersection_int = torch.sum(intersection)
            total_sum = torch.sum(pred_flat) + torch.sum(actual_flat)

            score = 2*intersection_int / (total_sum + self.epsilon)
            return score
        
        for predicted in predicted_all:
            pred_probab = torch.sigmoid(predicted)
            pred_flat = pred_probab.view(pred_probab.size(0), -1) 
            dice_loss = 1 - _dice_score(actual_flat, pred_flat)
            bce = torch.nn.functional.binary_cross_entropy(pred_flat, actual_flat)
            accumulated_loss += (((1 - self.dice_weight) * bce) + (self.dice_weight * dice_loss))
        return 1/len(predicted_all) * accumulated_loss
    
    # well this is a problem, torch convention is prediction first then actual, and i did actual first then prediction
    # everywhere, i will have to "refactor" them all, vs code does not support this currently
class MixedLoss(torch.nn.Module):
    def __init__(self, *losses, weights : list):
        super().__init__()
        self.losses = {f"loss_{i}" for i, loss in enumerate(losses)}
        self.weights = {f"weight_{i}" for i, weight in enumerate(weights)}
        # i should normalize weights so that Sum = 1, or maybe not, it should work either way i guess

    def forward(self, actual, prediction):
        loss = 0
        for loss, weight in zip(self.losses, self.weights):
            loss += (weight * loss(actual, prediction))

            return loss
        