import argparse
import os
import sys
import numpy as np
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# For each sample in test set, this computes the gradient of the loss w.r.t the input data (data_grad), creates a perturbed image with fgsm_attack (perturbed_data)
# Then checks to see if the perturbed example is adversarial. To test the model accuracy, it saves and returns some successful adversarial examples to be visualized later
def FGSM_Test(model, device, test_loader, epsilon):

    correct = 0

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor to True to compute gradients with respect to input data
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        # If the initial prediction (before attack) is wrong, don't bother FGSM to turn it into adversarial
        if init_pred.item() != target.item():
            continue

        # Calculate the loss as Negative Log Likelihood Loss (NLLLoss)
        loss = F.nll_loss(output, target)

        # Zero all existing gradients and Calculate gradients of model in backward
        model.zero_grad()
        loss.backward()

        # Collect 'data gradients'. The data_grad is the gradient of the loss w.r.t the input image.
        data_grad = data.grad.data

        # FGSM attack function to create adversarial example (perturbed image)
        sign_data_grad  = data_grad.sign() # Collect the element-wise sign of the data gradient
        perturbed_data = data + epsilon * sign_data_grad # FGSM Definition
        perturbed_data = torch.clamp(perturbed_data, 0, 1) # Clipping perturbed_data within [input_min, input_max]

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Final prediction after applying attack
        final_pred = output.max(1, keepdim=True)[1] # To get the index of the maximum log-probability

        if final_pred.item() == target.item():
            correct += 1

    # Calculate final accuracy for the given epsilon after adversarial attack
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc

# To calculate the sparsity range of each layer for each input image
def calculate_range(current_sparsity_list, sparsity_range, class_index):
    
    for layer in range (len(current_sparsity_list)):

        if current_sparsity_list[layer] < sparsity_range[class_index][layer][0]:
            sparsity_range[class_index][layer][0] = current_sparsity_list[layer]

        if current_sparsity_list[layer] > sparsity_range[class_index][layer][1]:
            sparsity_range[class_index][layer][1] = current_sparsity_list[layer]

    return sparsity_range

# The output ranges from 0 to 1, where 0 means no similarity and 1 means identical tensors (all non-zero elements are the same)
def jaccard_similarity(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2).sum().item()
    union = torch.logical_or(tensor1, tensor2).sum().item()
    return intersection / union

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [min, max]
def clip_tensor(input_tensor, eps, batch_size, min_data, max_data):

    tensor_norm = torch.stack([torch.norm(input_tensor[i], p=2) for i in range(batch_size)])

    # If the L2 norm of generated adversarial is greater than the eps, we scale it down by a factor of (eps / tensor_norm)
    scaled_tensor = torch.stack([torch.where(tensor_norm[i] > eps, input_tensor[i] * (eps / tensor_norm[i]), input_tensor[i]) for i in range(batch_size)])

    clipped_tensor = torch.stack([torch.clamp(scaled_tensor[i], min_data[i], max_data[i]) for i in range(batch_size)])

    return clipped_tensor
    
# Generates sparsity attack for all images of original testset
def Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args):
    
    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_zeros_before = 0
    total_net_sizes_before = 0
    total_net_zeros_after = 0
    total_net_sizes_after = 0

    """
    # A value of 1 means the sparsity attack has polluted the last processed input data of the same class, so we need to leave the current input clean.
    # A value of 0 means the sparsity attack has left the last processed input data of the same class clean, so we need to store the polluted input.
    last_status_of_class  = [0] * num_classes
    """
    adversarial_data = []
    num_of_items_in_class = [0] * num_classes
    num_of_adver_in_class = [0] * num_classes
    
    for index, (data, target) in enumerate(tqdm(test_loader, desc='Attack Generation Progress')):

        data, target = data.to(device), target.to(device)
        c_min = 0
        c_max = 1
        momentum = 0.9
        i_max = args.imax
        eps = args.eps
        eps_iter = args.eps_iter
        coeff = torch.full((args.batch_size,), c_init, device=device)

        # To inject the Clean Image to model to compute the initial accuracy and sparsity rate
        output = model(data)
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_before += net_zeros
        total_net_sizes_before += net_sizes
        init_pred = output[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target).sum().item()

        min_data, _ = torch.min(data.view(args.batch_size, -1), dim=1)
        max_data, _ = torch.max(data.view(args.batch_size, -1), dim=1)
        
        x = data
        g = torch.zeros_like(x)
        x = x.clone().detach().requires_grad_(True) # To compute the gradient of loss function w.r.t input X
        
        optimizer = optim.Adam([x], lr=args.lr, amsgrad=True)
        """optimizer = optim.SGD([x], lr=args.lr, momentum=0.9)"""
        """mse_loss = nn.MSELoss()"""
        
        for i in range(i_max):
            
            optimizer.zero_grad()
            
            output = model(x)
            
            if (i>0):
                adv_pred = output[0].max(1, keepdim=False)[1]
                coeff = torch.stack([torch.where(init_pred[i] == adv_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
            
            # To compute cross-entropy loss independently for each image in the batch
            l_ce = torch.stack([F.cross_entropy(output[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])
            
            # To compute l_sparsity (using Tanh function) independently for each image in the batch
            l_sparsity = output[3]
            
            # To minimize the MSE between two images (L2-Norm Difference of two images might work better)
            """l_mse = torch.stack([mse_loss(x[j], data[j]) for j in range(args.batch_size)])"""
            
            """l_x = l_sparsity + (coeff * l_ce) + l_mse"""
            l_x = l_sparsity + (coeff * l_ce)
                    
            l_x.backward(torch.ones_like(l_x), retain_graph=True) # Using torch.ones_like(l_x) ensures that the gradients of each element in l_x is computed independently 

            g = (momentum * g) + x.grad.data
            
            with torch.no_grad():
                x = torch.stack([x[k] - (eps_iter * (g[k] / torch.norm(g[k], p=2))) for k in range(args.batch_size)])
                if args.constrained:
                    x = clip_tensor(x, eps, args.batch_size, min_data, max_data)
            
            x.requires_grad_(True)

        # To store the generated adversarial (x_adv) or benign data in a similar dataset with a pollution rate of 100% for each class
        x_adv = x.detach()
        if args.store_attack:
            for i in range(x_adv.size(0)):
                adversarial_data.append((x_adv[i], target[i], 1))
        for t in range(len(target)):
            num_of_items_in_class[target[t].item()] = num_of_items_in_class[target[t].item()] + 1
            num_of_adver_in_class[target[t].item()] = num_of_adver_in_class[target[t].item()] + 1
            
        """
        # To store the generated adversarial (x_adv) or benign data in a similar dataset with a pollution rate of 50% for each class
        if last_status_of_class[target.item()] == 0:
            if args.store_attack:
                adversarial_data.append((x_adv, target, 1))
            last_status_of_class[target.item()] = 1
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1
            num_of_adver_in_class[target.item()] = num_of_adver_in_class[target.item()] + 1
        else:
            if args.store_attack:
                adversarial_data.append((data, target, 0))
            last_status_of_class[target.item()] = 0
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1
        """

        # Compute the L2-Norm of difference between perturbed image (x_adv) and clean image (data)
        l2norm_diff = torch.norm((x_adv-data).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/((x_adv.view(args.batch_size, -1)).size(1))
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # To inject the Adversarial Image to model again to compute the final accuracy and sparsity rate
        output = model(x_adv)
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_after += net_zeros
        total_net_sizes_after += net_sizes
        final_pred = output[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target).sum().item()

        num_processed_images += args.batch_size

    # To Create a new dataset using the AdversarialDataset class
    adversarial_dataset = AdversarialDataset(adversarial_data)
    print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of adversarial in each class: {num_of_adver_in_class}")

    # Calculate overall accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    return adversarial_dataset, first_acc, final_acc, l2_norms, (total_net_zeros_before/total_net_sizes_before), (total_net_zeros_after/total_net_sizes_after)

def Sparsity_Attack_Profile (model, device, train_loader, num_classes, args):
    
    updated_maps  = [None] * num_classes
    diff_num_ones = [0] * num_classes
    prev_num_ones = [0] * num_classes
    curr_num_ones = [0] * num_classes
    num_layers = 5
    
    # To define sparsity-range for each class
    range_for_each_class = [[[float('inf'), float('-inf')] for _ in range(num_layers)] for _ in range(num_classes)]

    for index, (data, target) in enumerate(tqdm(train_loader, desc='Profiling Progress')):
        
        data, target = data.to(device), target.to(device)

        output = model(data)
        pred = output[0].max(1, keepdim=False)[1]
        
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers
                for i in range(num_layers):
                    current_sparsity_rate[i] = (output[1][i] / output[2][i]) if output[2][i] != 0 else 0
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[3]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[3])
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(f"\n{diff_num_ones}\n")
                    break

    return updated_maps, range_for_each_class, index

def Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_loader, num_classes, args):
    
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes

    # Initialization for sparsity-maps
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 5
    layer_inclusion_threshold = num_layers - 4
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target, adversarial) in enumerate(tqdm(test_loader, desc='Data Progress')): 

        for i in range (args.batch_size):

            single_image = data[i].unsqueeze(0).to(device)
            single_target = target[i].unsqueeze(0).to(device)

            output = model(single_image)
            
            pred = output[0].max(1, keepdim=False)[1]

            ############################################## sparsity-range #####################################################################

            current_sparsity_rate = [0.0] * num_layers

            for L in range(num_layers):
                current_sparsity_rate[L] = (output[1][L] / output[2][L]) if output[2][L] != 0 else 0
            
            in_range_status = [0] * num_layers
            for M in range(num_layers):
                if not offline_sparsity_ranges[pred.item()][M][0] <= current_sparsity_rate[M] <= offline_sparsity_ranges[pred.item()][M][1]:
                    in_range_status[M] = 1
            
            if sum(in_range_status) >= layer_inclusion_threshold:
                if adversarial[i].item() == 1:
                    correctly_predicted_adversarial_for_class_range[single_target.item()] += 1

            ############################################## sparsity-map #######################################################################
            
            sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[3])

            # if the following condition is True, we predict that the input is adversarial
            if sim_rate <= args.sim_threshold:
                if adversarial[i].item() == 1:
                    correctly_predicted_adversarial_for_class_map[single_target.item()] += 1

            # To check the real adversarial status for the same predicted class
            if adversarial[i].item() == 1:
                generated_adversarial_for_class[single_target.item()] += 1 

            num_of_items_in_class[single_target.item()] += 1

            ###################################################################################################################################
        
    for predicted, generated in zip(correctly_predicted_adversarial_for_class_map, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_map.append((predicted/generated)*100)
    
    for predicted, generated in zip(correctly_predicted_adversarial_for_class_range, generated_adversarial_for_class): # Range
        correctly_predicted_adversarial_ratio_range.append((predicted/generated)*100)
    
    correctly_predicted_adversarial_ratio_map = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_map]

    correctly_predicted_adversarial_ratio_range = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_range] # Range

    overall_accuracy_map = sum(correctly_predicted_adversarial_for_class_map)/sum(generated_adversarial_for_class)
    overall_accuracy_range = sum(correctly_predicted_adversarial_for_class_range)/sum(generated_adversarial_for_class)
    
    print(f"\nDistribution of data in each class: {num_of_items_in_class}\n")
    print(f"Correctly predicted adversarials for each class (map): {correctly_predicted_adversarial_for_class_map}\n")
    print(f"Correctly predicted adversarials for each class (rng): {correctly_predicted_adversarial_for_class_range}\n")
    print(f"Number of generated adversarials for each class: {generated_adversarial_for_class}\n")
    print(f"Percentage of correctly predicted adversarials for each class (map): {correctly_predicted_adversarial_ratio_map}\n")
    print(f"Percentage of correctly predicted adversarials for each class (rng): {correctly_predicted_adversarial_ratio_range}\n")
    print(f"Overall attack detection accuracy using sparsity-map method: {overall_accuracy_map*100:.2f}\n")
    print(f"Overall attack detection accuracy using sparsity-range method: {overall_accuracy_range*100:.2f}\n")
    
    return 

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--phase', default='train', help="train, FGSM, sparsity-attack, sparsity-profile, sparsity-detect, test")
    parser.add_argument('--method', default='sparsity-map', help="profiling can be performed based on sparsity-map or sparsity-range")
    parser.add_argument('--weights', default=None, help="The path to the pretrained weights. Should be specified when testing")
    parser.add_argument('--dataset', default=None, help="The path to the train and test datasets")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="similarity threshold")
    parser.add_argument('--img_start_index', default=0, type=int, help="The first index of the dataset")
    parser.add_argument('--img_end_index', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--constrained', action='store_true', help="To active clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To store the generated adversarials")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial rather than clean dataset in the test phase")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the architecture: cnvlutin or dadiannao")
    args = parser.parse_args()
    print(f"\n{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{device} assigned for processing!\n")

    # MNIST dataset and dataloader declaration
    if args.phase == 'train':
        train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset  = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'FGSM':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)
    elif args.phase == 'sparsity-attack':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.img_start_index, args.img_end_index)))
        test_loader  = DataLoader(test_dataset_sub, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'sparsity-profile':
        train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'sparsity-detect':
        test_dataset = torch.load(f"{args.dataset}/adversarial_dataset_constrained_False.pt", map_location=device)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'test':
        if args.adversarial:
            test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained_False.pt', map_location=device)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            test_dataset  = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
            test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Network Initialization
    if args.phase == 'train' or args.phase == 'FGSM':
        from models.model_train import LeNet5
        model = LeNet5().to(device)
    elif args.phase == 'sparsity-attack':
        from models.model_attack import LeNet5
        model = LeNet5(args).to(device)
    elif args.phase == 'sparsity-profile' or args.phase == 'sparsity-detect':
        from models.model_profile_detect import LeNet5
        model = LeNet5().to(device)
    elif args.phase == 'test':
        from models.model_test_power_delay import LeNet5
        model = LeNet5(args).to(device)
        
    print(f"{model}\n")
    
    if args.phase == 'train':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        loss_fn = CrossEntropyLoss()
        EPOCHS = args.epochs
        prev_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            for index, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs.float())
                loss = loss_fn(preds, labels.long())
                loss.backward()
                optimizer.step()
    
            model.eval()
            correct_num = 0
            sample_num = 0
            for index, (image, label) in enumerate(test_loader):
                image = image.to(device)
                label = label.to(device)
                preds = model(image.float().detach())
                preds =torch.argmax(preds, dim=-1)
                current_correct_num = preds == label
                correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                sample_num += current_correct_num.shape[0]
            
            acc = correct_num / sample_num
            print('Accuracy in Epoch %3d: %1.4f' % (epoch+1, acc*100), flush=True)
            
            if not os.path.isdir("weights"):
                os.mkdir("weights")
            torch.save(model.state_dict(), f"weights/mnist_{epoch+1}_{acc:.4f}.pkl")
            
            if np.abs(acc - prev_acc) < 1e-4:
                break
            prev_acc = acc
        
        print("Training finished!")

    elif args.phase == 'test':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        correct_num = 0
        sample_num = 0
        num_layers = 5
        
        total_inference_cycles = 0
        total_dynamic_energy_dataset = 0
        total_static_energy_dataset = 0
        detail_dynamic_energy_dataset = [0] * 10
        detail_static_energy_dataset = [0] * 10
        
        # Read from (NBin+Offset+SB) + Multiplication + Addition by AdderTree + Activation by ReLU + Write into NBout
        critical_path_delay = (0.201776e-9 + 0.201776e-9 + 0.861771e-9) + 3732e-12 + 16179e-12 + 124e-12 + (0.261523e-9)
            
        zeros_all_layers = [0] * num_layers
        sizes_all_layers = [0] * num_layers
        net_zeros  =  0
        net_sizes  =  0
    
        # Set the model in evaluation mode
        model.eval()
    
        # for index, (image, label, adversarial) in enumerate(tqdm(test_loader, desc='Data Progress')):
        for index, (image, label) in enumerate(tqdm(test_loader, desc='Data Progress')):
            
            image, label = image.to(device), label.to(device)
            
            output = model(image, args.power)
                   
            # Sparsity calculations
            for i in range(num_layers): zeros_all_layers[i] += output[1][i]
            for i in range(num_layers): sizes_all_layers[i] += output[2][i]
            net_zeros += sum(output[1])
            net_sizes += sum(output[2])
                
            # Power consumption profiling
            total_dynamic_energy_dataset += output[3]
            total_static_energy_dataset += output[4]
            detail_dynamic_energy_dataset = [a + b for a, b in zip(detail_dynamic_energy_dataset, output[5])]
            detail_static_energy_dataset = [a + b for a, b in zip(detail_static_energy_dataset, output[6])]
            total_inference_cycles += output[7]
                
            preds = torch.argmax(output[0], dim=-1)
            current_correct_num = preds == label
            correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            sample_num += current_correct_num.shape[0]

        # To print accuracy statistics
        acc = correct_num / sample_num
        print('\nAccuracy of Testing is (percent): %1.2f' % (acc*100), '\n')
            
        # To print sparsity rate statistics
        SR_L1  = (zeros_all_layers[0]/sizes_all_layers[0]) if sizes_all_layers[0] != 0 else 0
        SR_L2  = (zeros_all_layers[1]/sizes_all_layers[1]) if sizes_all_layers[1] != 0 else 0
        SR_L3  = (zeros_all_layers[2]/sizes_all_layers[2]) if sizes_all_layers[2] != 0 else 0
        SR_L4  = (zeros_all_layers[3]/sizes_all_layers[3]) if sizes_all_layers[3] != 0 else 0
        SR_L5  = (zeros_all_layers[4]/sizes_all_layers[4]) if sizes_all_layers[4] != 0 else 0
        SR_Net = (net_zeros/net_sizes) if net_sizes != 0 else 0
    
        print('Sparsity rate of L1 is: %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is: %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is: %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is: %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is: %1.5f'   % (SR_L5))
        print('Sparsity rate of Network: %1.5f' % (SR_Net))
            
        # To print energy consumption and latency statistics
        avg_inference_cycles = total_inference_cycles / sample_num
        avg_dynamic_energy_dataset = total_dynamic_energy_dataset / sample_num
        avg_static_energy_dataset = (total_static_energy_dataset / sample_num) * critical_path_delay
        total_energy = avg_dynamic_energy_dataset + avg_static_energy_dataset
        detail_static_energy_dataset = [x * critical_path_delay for x in detail_static_energy_dataset]
        eng_total_detail_dataset = [a + b for a, b in zip(detail_dynamic_energy_dataset, detail_static_energy_dataset)]
        avg_inference_latency = critical_path_delay * avg_inference_cycles
    
        if args.power:
            print('\n########## Latency Statistics ##########\n')
            print('Average Inference Latency: %1.9f (Sec)' % (avg_inference_latency))
            print('Average Inference Cycles : %1.2f' % (avg_inference_cycles))
            print('\n########## Energy Statistics ##########\n')
            print('Average Dynamic Energy of Dataset: %1.9f (J)' % (avg_dynamic_energy_dataset))
            print('Average Static Energy of Dataset: %1.9f (J)' % (avg_static_energy_dataset))
            print('Total Energy of Dataset: %1.9f (J)' % (total_energy),'\n')
            print('########## Dynamic Energy Breakdown ##########\n')
            print('Multiplier (J): %1.20f' % (detail_dynamic_energy_dataset[0]))
            print('AdderTree  (J): %1.20f' % (detail_dynamic_energy_dataset[1]))
            print('ReLu       (J): %1.20f' % (detail_dynamic_energy_dataset[2]))
            print('Encoder    (J): %1.20f' % (detail_dynamic_energy_dataset[3]))
            print('Dispatcher (J): %1.20f' % (detail_dynamic_energy_dataset[4]))
            print('NBin       (J): %1.20f' % (detail_dynamic_energy_dataset[5]))
            print('Offset     (J): %1.20f' % (detail_dynamic_energy_dataset[6]))
            print('SB         (J): %1.20f' % (detail_dynamic_energy_dataset[7]))
            print('NBout      (J): %1.20f' % (detail_dynamic_energy_dataset[8]))
            print('NM         (J): %1.20f' % (detail_dynamic_energy_dataset[9]))
        
            total_eng_dnmc = sum(detail_dynamic_energy_dataset)
            print('Total      (J): %1.20f' % (total_eng_dnmc),'\n')
        
            print('########## Static Energy Breakdown ###########\n')
            print('Multiplier (J): %1.20f' % (detail_static_energy_dataset[0]))
            print('AdderTree  (J): %1.20f' % (detail_static_energy_dataset[1]))
            print('ReLu       (J): %1.20f' % (detail_static_energy_dataset[2]))
            print('Encoder    (J): %1.20f' % (detail_static_energy_dataset[3]))
            print('Dispatcher (J): %1.20f' % (detail_static_energy_dataset[4]))
            print('NBin       (J): %1.20f' % (detail_static_energy_dataset[5]))
            print('Offset     (J): %1.20f' % (detail_static_energy_dataset[6]))
            print('SB         (J): %1.20f' % (detail_static_energy_dataset[7]))
            print('NBout      (J): %1.20f' % (detail_static_energy_dataset[8]))
            print('NM         (J): %1.20f' % (detail_static_energy_dataset[9]))
        
            total_eng_stat = sum(detail_static_energy_dataset)
            print('Total      (J): %1.20f' % (total_eng_stat),'\n')
        
            print('########## Total Energy Breakdown ############\n')
            print('Multiplier (J): %1.20f' % (eng_total_detail_dataset[0]))
            print('AdderTree  (J): %1.20f' % (eng_total_detail_dataset[1]))
            print('ReLu       (J): %1.20f' % (eng_total_detail_dataset[2]))
            print('Encoder    (J): %1.20f' % (eng_total_detail_dataset[3]))
            print('Dispatcher (J): %1.20f' % (eng_total_detail_dataset[4]))
            print('NBin       (J): %1.20f' % (eng_total_detail_dataset[5]))
            print('Offset     (J): %1.20f' % (eng_total_detail_dataset[6]))
            print('SB         (J): %1.20f' % (eng_total_detail_dataset[7]))
            print('NBout      (J): %1.20f' % (eng_total_detail_dataset[8]))
            print('NM         (J): %1.20f' % (eng_total_detail_dataset[9]))
        
            total_eng = sum(eng_total_detail_dataset)
            print('Total      (J): %1.20f' % (total_eng),'\n')
    
        sys.exit(0)
        
    elif args.phase == 'FGSM':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        # Epsilon parameter Initialization
        epsilons = [0, 0.007, 0.002, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        
        # Set the model in evaluation mode
        model.eval()
        accuracies = []

        # Run test for each epsilon
        for eps in epsilons:
            acc = FGSM_Test(model, device, test_loader, eps)
            accuracies.append(acc)

        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
        sys.exit()
    
    elif args.phase == 'sparsity-attack':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes  = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args)
        
        # Save the generated adversarial dataset to disk
        torch.save(adversarial_dataset, f"adversarial_data/adversarial_dataset_constrained_{args.constrained}.pt")

        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

    elif args.phase == 'sparsity-profile':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()
        
        num_classes  = len(train_dataset.classes)

        """The Sparsity_Attack_Profile function returns an array with the size of num_classes
           in which each element comprises a Sparsity-Map Tensor with values of 0 and 1"""
        sparsity_maps, sparsity_range, index = Sparsity_Attack_Profile(model, device, train_loader, num_classes, args) 
        
        # Store the Sparsity_Maps in an offline file
        print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}.\n")
        print(f"P1 = {((index+1)/len(train_loader)*100):.2f} % of trainset has been used for profiling.\n")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./profile_data/sparsity_maps.pth", pickle_protocol=4)
        
        if args.method == 'sparsity-range':
            # Save the Sparsity Range to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_range, "./profile_data/sparsity_ranges.pth", pickle_protocol=4)
            
        sys.exit(0)

    elif args.phase == 'sparsity-detect':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        # Load the sparsity-maps and sparsity-ranges from offline file (generated in profiling phase)
        offline_sparsity_maps = torch.load("profile_data/sparsity_maps.pth")
        offline_sparsity_ranges = torch.load("profile_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_loader, num_classes, args)
        
        sys.exit(0)
        

# Train:             python3 main.py --phase train --dataset mnist_dataset
# FGSM:              python3 main.py --phase FGSM --dataset mnist_dataset --weights weights/mnist_0.9882.pkl 
# Attack Generation: python3 main.py --phase sparsity-attack --eps 0.9 --eps_iter 0.9 --imax 200 --beta 20 --batch_size 5 --img_end_index 20 --constrained --store_attack --dataset mnist_dataset --weights weights/mnist_0.9882.pkl
# Attack Profiling:  python3 main.py --phase sparsity-profile --method sparsity-map/range --dataset mnist_dataset --weights weights/mnist_0.9882.pkl
# Attack Detection:  python3 main.py --phase sparsity-detect --sim_threshold 0.5 --dataset adversarial_data/mnist_dataset --weights weights/mnist_0.9882.pkl
# Test Power Delay:  python3 main.py --phase test --power --arch cnvlutin --batch_size 10 --adversarial --dataset mnist_dataset --weights weights/mnist_0.9882.pkl

# Compile C++ file:  "gcc -shared -o lib_power_functions.so -fPIC nested_loops.c"