from .imports import torch

__all__ = ['IntersectionOverUnion', 'DiceLoss', 'MixedLoss']

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6

    def forward(self, label, pred):
        if isinstance(pred, list):
            return sum(self.forward(label, p) for p in pred) / len(pred)

        pred_prob = torch.sigmoid(pred)

        B = pred_prob.shape[0]
        intersection = (label * pred_prob).view(B, -1).sum(dim=1)
        denom = (pred_prob + label).view(B, -1).sum(dim=1)
        dice_score = (2 * intersection) / (denom + self.epsilon) 
        return (1 - dice_score).mean()           
    
class IntersectionOverUnion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6
        
    def forward(self, actual, prediction):
        pred_prob = torch.sigmoid(prediction)
        
        B = pred_prob.shape[0]
        intersection = (pred_prob * actual).view(B, -1).sum(dim=1) 
        union = (pred_prob + actual).view(B, -1).sum(dim=1) - intersection
        iou = intersection / (union + self.epsilon)
        return (1 - iou).mean()        

class Mixed_Dice_Sigmoid(torch.nn.Module):
    def __init__(self, dice_weight = 0.5):
        super().__init__()
        self.epsilon = 1e-6
        self.dice_weight = 0.5

    def forward(self, actual, predicted_all):
        if not isinstance(predicted_all, list):
            predicted_all = [predicted_all]
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
    # ===================== NOT IMPLIMENTED ===========================
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
        