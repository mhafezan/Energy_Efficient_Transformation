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
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Custom Dataset to store the transformed data
class Transformed_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [min, max]
def clip_tensor(input_tensor, eps, batch_size, min_data, max_data):

    tensor_norm = torch.stack([torch.norm(input_tensor[i], p=2) for i in range(batch_size)])

    # If the L2 norm of generated adversarial is greater than the eps, we scale it down by a factor of (eps / tensor_norm)
    scaled_tensor = torch.stack([torch.where(tensor_norm[i] > eps, input_tensor[i] * (eps / tensor_norm[i]), input_tensor[i]) for i in range(batch_size)])

    clipped_tensor = torch.stack([torch.clamp(scaled_tensor[i], min_data[i], max_data[i]) for i in range(batch_size)])

    return clipped_tensor
    
# Increases sparsity rate for all images of original test set
def sparsity_rate_increment(model, device, test_loader, num_classes, c_init, args):
    
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

    # To Create a new dataset using the Transformed Dataset class
    adversarial_dataset = Transformed_Dataset(adversarial_data)
    print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of adversarial in each class: {num_of_adver_in_class}")

    # Calculate overall accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    return adversarial_dataset, first_acc, final_acc, l2_norms, (total_net_zeros_before/total_net_sizes_before), (total_net_zeros_after/total_net_sizes_after)

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper-parameters
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--phase', default='train', help="train, test, sparsity-transform")
    parser.add_argument('--weights', default=None, help="Path to the pretrained weights")
    parser.add_argument('--dataset', default=None, help="Path to the train and test datasets")
    parser.add_argument('--imax', default=100, type=int, help="Optimization iterations of Sparsity-Transformer function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon coefficient for optimization iterations")
    parser.add_argument('--img_start_index', default=0, type=int, help="The first index of the dataset")
    parser.add_argument('--img_end_index', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--constrained', action='store_true', help="To active clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To store the generated adversarials")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial rather than clean dataset during test phase")
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
    elif args.phase == 'sparsity-transform':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.img_start_index, args.img_end_index)))
        test_loader  = DataLoader(test_dataset_sub, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'test':
        if args.adversarial:
            test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained_False.pt', map_location=device)
            test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            test_dataset  = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
            test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Network Initialization
    if args.phase == 'train':
        from models.model_train import LeNet5
        model = LeNet5().to(device)
    elif args.phase == 'sparsity-transform':
        from models.model_transform import LeNet5
        model = LeNet5(args).to(device)
    elif args.phase == 'test':
        from models.model_test_power_delay import LeNet5
        model = LeNet5(args).to(device)
        
    print(f"{model}\n")
    
    if args.phase == 'train':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        loss_fn = CrossEntropyLoss()
        EPOCHS = args.epochs
        prev_acc = 0
        
        # Lists to store per-epoch accuracies for plotting
        train_acc_list = []
        val_acc_list = []
        
        for epoch in range(EPOCHS):
            model.train()
            for index, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs.float())
                loss = loss_fn(preds, labels.long())
                loss.backward()
                optimizer.step()
            
            # Training accuracy (on train set) for each epoch
            model.eval()
            train_correct = 0
            train_samples = 0
            with torch.no_grad():
                for idx, (img, lbl) in enumerate(train_loader):
                    img, lbl = img.to(device), lbl.to(device)
                    out = model(img.float())
                    pred = torch.argmax(out, dim=-1)
                    eq = (pred == lbl)
                    train_correct += int(torch.sum(eq).cpu().numpy())
                    train_samples += eq.shape[0]
            train_acc = train_correct / train_samples if train_samples > 0 else 0.0
            train_acc_list.append(train_acc)
    
            # Validation accuracy (on test set)
            test_correct = 0
            test_samples = 0
            with torch.no_grad():
                for index, (image, label) in enumerate(test_loader):
                    image, label = image.to(device), label.to(device)
                    preds_test = model(image.float())
                    preds_test = torch.argmax(preds_test, dim=-1)
                    current_correct_num = preds_test == label
                    test_correct += int(torch.sum(current_correct_num.to('cpu')))
                    test_samples += current_correct_num.shape[0]

            test_acc = test_correct / test_samples if test_samples > 0 else 0.0
            val_acc_list.append(test_acc)
            print('Epoch %3d - Train Acc: %1.4f  Val Acc: %1.4f' % (epoch+1, train_acc*100, test_acc*100), flush=True)
            
            if not os.path.isdir("weights"):
                os.mkdir("weights")
            torch.save(model.state_dict(), f"weights/mnist_{epoch+1}_{test_acc:.4f}.pkl")
            
            if np.abs(test_acc - prev_acc) < 1e-4:
                break
            prev_acc = test_acc
        
        print("Training finished!")
        
        # Plot train and validation accuracy vs epochs
        try:
            epochs = list(range(1, len(train_acc_list) + 1))
            plt.figure()
            plt.plot(epochs, [a * 100 for a in train_acc_list], label='Train Accuracy')
            plt.plot(epochs, [a * 100 for a in val_acc_list], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Train versus Validation Accuracy')
            plt.legend()
            plt.grid(True)
            # Save figure
            if not os.path.isdir('figures'):
                os.mkdir('figures')
            plt.savefig('figures/train_vs_test.png', bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not plot accuracy curves: {e}")
        sys.exit(0)

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
    
    elif args.phase == 'sparsity-transform':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes  = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = sparsity_rate_increment(model, device, test_loader, num_classes, c_init, args)
        
        # Save the generated adversarial dataset to disk
        torch.save(adversarial_dataset, f"adversarial_data/adversarial_dataset_constrained_{args.constrained}.pt")

        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

# Train:            python3 main.py --phase train --dataset mnist_dataset
# Transformation:   python3 main.py --phase sparsity-transform --eps 0.9 --eps_iter 0.9 --imax 200 --beta 20 --batch_size 5 --img_end_index 20 --constrained --store_attack --dataset mnist_dataset --weights weights/mnist_0.9882.pkl
# Test Power Delay: python3 main.py --phase test --power --arch cnvlutin --batch_size 10 --adversarial --dataset mnist_dataset --weights weights/mnist_0.9882.pkl
# Compile C++ file: "gcc -shared -o lib_power_functions.so -fPIC nested_loops.c"