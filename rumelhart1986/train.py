"""Training loop for the Rumelhart 1986 reproduction."""

# Implement training logic here.
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
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
    
    all_triples = []
    for (p_ele, r_ele), p2_list in TRIPLES.items():
        p_indx = PEOPLE.index(p_ele)
        r_indx = RELATIONSHIPS.index(r_ele)
        
        for p2_ele in p2_list:
            p2_indx = PEOPLE.index(p2_ele)
            all_triples.append((p_indx, r_indx, p2_indx))
    
    print(f"DEBUG: all_triples count = {len(all_triples)}")
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
    
    model= TreeNet()
    optimizer= optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
    
    print("\nStarting Training (1500 Sweeps)...")
    
    
    for sweep in range(1, 1502):
        if sweep == 21:
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
                param_group['momentum']= 0.9
        
        optimizer.zero_grad()
        total_sweep_loss = 0.0
        
        for p_indx, r_indx, p2_indx in train_triples:
            
            person1, relationship= encode(p_indx, r_indx)
            
            target= torch.zeros(1, 24)
            target[0, p2_indx]= 1.0
            
            output= model(person1, relationship)
            loss= compute_masked_loss(output, target)
            
            loss.backward()
            total_sweep_loss += loss.item()
            
        optimizer.step()
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'b_' not in name:
                    param.mul_(0.998)
                
        if sweep == 1 or sweep % 100 == 0:
            print(f"Sweep {sweep:4d}/1500 | Loss: {total_sweep_loss:.6f}")
            
    evaluate(model, test_triples)
    


if __name__ == "__main__":
    main()