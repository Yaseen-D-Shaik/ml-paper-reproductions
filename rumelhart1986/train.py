"""Training loop for the Rumelhart 1986 reproduction."""

# Implement training logic here.
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data.family_tree import PEOPLE, RELATIONSHIPS, FAMILY_DATA
from model.network import TreeNet, encode


def compute_masked_loss(output, target):
    error = output - target
    mask = ~((target == 1.0) & (output > 0.8)) & \
           ~((target == 0.0) & (output < 0.2))
    
    mask_error= error * mask.detach()
    return 0.5 * torch.sum(mask_error ** 2)


def dataset():
    
    test_triples = [
        (PEOPLE.index("Colin"), RELATIONSHIPS.index("uncle"), PEOPLE.index("Arthur")),
        (PEOPLE.index("Alfonse"), RELATIONSHIPS.index("uncle"), PEOPLE.index("Emilio")),
        (PEOPLE.index("Penelope"), RELATIONSHIPS.index("mother"), PEOPLE.index("Victoria")),
        (PEOPLE.index("Charlotte"), RELATIONSHIPS.index("aunt"), PEOPLE.index("Christine")),
    ]
    test_set = set(test_triples)
    
    for (p_ele, r_ele), p2_list in FAMILY_DATA.items():
        
        all_triples= []
        p_indx= PEOPLE.index(p_ele)
        r_indx= RELATIONSHIPS.index(r_ele)
        
        for p2_ele in p2_list:
            p2_indx= PEOPLE.index(p2_ele)
            all_triples.append((p_indx, r_indx, p2_indx))
            
    train_triples = [t for t in all_triples if t not in test_set]
    return train_triples, test_triples
        

def evaluate(model, test_triples):
    model.eval()
    print("\n" + "="*55)
    print("      HELD-OUT TEST TRIPLES EVALUATION")
    print("="*55)

    correct_count = 0
    with torch.no_grad():
        for p1_idx, rel_idx, p2_idx in test_triples:
            p1_tensor, r_tensor = encode(p1_idx, rel_idx)
            output = model(p1_tensor, r_tensor)

            target_activation = output[0, p2_idx].item()
            is_correct = target_activation > 0.8
            if is_correct:
                correct_count += 1

            p1_name = PEOPLE[p1_idx]
            rel_name = RELATIONSHIPS[rel_idx]
            p2_name = PEOPLE[p2_idx]

            status = "PASSED" if is_correct else "FAILED"
            print(f"Query: ({p1_name}, {rel_name}) -> Target: {p2_name}")
            print(f"  Activation Score: {target_activation:.4f} | Status: [{status}]\n")

    print(f"Test Accuracy: {correct_count}/{len(test_triples)} ({correct_count/len(test_triples)*100:.1f}%)")
    print("="*55)
    

def main():
    
    torch.manual_seed(42)
    random.seed(42)
    
    train_triples, test_triples= dataset()
    print(f"Dataset Loaded | Train Triples: {len(train_triples)} | Test Triples: {len(test_triples)}")
    
    optim.SGD()
    