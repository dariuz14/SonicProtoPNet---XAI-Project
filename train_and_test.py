import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import push
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

def _preprocess_label(label):
    if torch.is_tensor(label) == False:
        target = torch.tensor([int(el) for el in label])
    else:
        target = label

    return target.cuda(), target 
        

def _train_or_test(model, data_loader, optimizer=None, class_specific=True, coefs=None, use_l1_mask=True, log=print):
    is_train = optimizer is not None
    sep_cost = 0
    clst_cost = 0
    total_cross_entropy = 0
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_loss_accum = 0
    
    # Loss function = cross entropy + lambda1 clustering cost + lambda2 separation cost
    criterion = nn.CrossEntropyLoss()

    for idx, (data, label) in enumerate(data_loader):

        data = data.cuda()
        target, label = _preprocess_label(label)
        gradient_enabled = torch.enable_grad() if is_train else torch.no_grad()

        with gradient_enabled: 
            output, min_distances = model(data)

            cross_entropy = criterion(output, target)

            if class_specific:
                # Max distance between prototypes and feature vectors of same class
                max_distances = (model.prototype_shape[1]* model.prototype_shape[2]* model.prototype_shape[3])

                # Calculate cluster cost
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_distances - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_distances - inverted_distances)

                # Calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_distances - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_distances - inverted_distances_to_nontarget_prototypes)
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1) 
            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)
            
            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            clst_cost += cluster_cost.item()
            sep_cost += separation_cost.item()

        if is_train:
            if class_specific:
                if coefs is not None:
                    total_loss = (coefs['crs_ent']*cross_entropy + coefs['clst'] * cluster_cost + coefs['sep'] * separation_cost + coefs['l1'] * l1)
                else:
                    total_loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:    
                    total_loss = (coefs['crs_ent']*cross_entropy + coefs['clst'] * cluster_cost + coefs['l1']*l1)
                else:
                    total_loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            total_loss.backward()
            total_loss_accum += total_loss.item()
            optimizer.step()
    
    log('\taccuracy: \t\t{0}%'.format(n_correct / n_examples * 100))
    if is_train:
        log(f'\ttotal loss:\t{total_loss_accum / n_batches}')
    
    return n_correct / n_examples, {'cluster_cost': clst_cost / n_batches, 'separation_cost': sep_cost / n_batches}

def train_pipeline(model, train_loader, train_loader_aug, warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler, 
                   num_train_epochs, num_warm_epochs, push_epochs, push_start, last_layer_convex_optimizations, 
                   class_specific, coefs, img_dir, prototype_img_filename_prefix, prototype_self_act_filename_prefix, 
                   proto_bound_boxes_filename_prefix, patience, model_dir=None, validation=False, data_loader=None, log=print):
    
    accs_per_epoch = []
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        # ---- WARM PHASE ----
        if epoch < num_warm_epochs:
            warm_only(model, log=log)
            _, _obj = train(model, train_loader, warm_optimizer, class_specific, coefs, log=log)
        # ---- JOINT PHASE ----
        else:
            joint(model, log=log)
            acc, _ = train(model, train_loader, joint_optimizer, class_specific, coefs, log=log)
            accs_per_epoch.append(acc)
            joint_lr_scheduler.step()

        # ---- PUSH PHASE ----
        # Push first, then last layer convex optimization
        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_loader_aug,
                prototype_network_parallel=model,
                class_specific=class_specific,
                preprocess_input_function=None,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
                
            # ---- LAST LAYER CONVEX OPTIMIZATION PHASE ----
            last_only(model, log=log)
            for _ in range(last_layer_convex_optimizations):
                _, _ = train(model, train_loader, last_layer_optimizer, class_specific, coefs, log=log)

        if validation:
            val_acc, _ = test(model, data_loader, class_specific, log=log)
            
            if val_acc > best_val_acc:
                epochs_without_improvement = 0
                best_val_acc = val_acc

                if model_dir:
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
                    log(f'\tNew best validation accuracy: {val_acc:.4f} - Model saved')
            else:
                epochs_without_improvement += 1
                log(f'\tNo improvement in validation accuracy for {epochs_without_improvement} epoch(s)')

        if validation and epoch >= num_warm_epochs:
            if epochs_without_improvement >= patience:
                log('Early stop training because: validation acc converged')
                log('Best validation accuracy: {0:.4f}'.format(best_val_acc))
                break
           

def train(model, train_loader, optimizer, class_specific=False, coefs=None, use_l1_mask=True, log=print):
    assert(optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test(model=model, data_loader=train_loader, optimizer=optimizer, 
                          class_specific=class_specific, coefs=coefs, use_l1_mask=use_l1_mask, log=log)

def test(model, test_loader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, data_loader=test_loader, optimizer=None, 
                          class_specific=class_specific, log=log)

def last_only(model, log=print):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.add_on_layers.parameters():
        param.requires_grad = False
    model.prototype_layer.requires_grad = False
    for param in model.last_layer.parameters():
        param.requires_grad = True
    
    log('\tlast layer')

def joint(model, log=print):
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.add_on_layers.parameters():
        param.requires_grad = True
    model.prototype_layer.requires_grad = True
    for param in model.last_layer.parameters():
        param.requires_grad = False
    
    log('\tjoint')

def warm_only(model, log=print):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.add_on_layers.parameters():
        param.requires_grad = True
    model.prototype_layer.requires_grad = True
    for param in model.last_layer.parameters():
        param.requires_grad = True
    
    log('\twarm')
    
def evaluate_prototypes(model, data_loader, num_prototypes_to_plot=None, save_dir=None, log=print):
    model.eval()

    all_min_distances = []
    all_labels = []

    log('\tprototype evaluation')
    # Save all min_distances matrices for all batches
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.cuda()
            _, min_distances = model(data)
            all_min_distances.append(min_distances.cpu())
            all_labels.append(target)
    
    all_min_distances = torch.cat(all_min_distances, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_prototypes = model.prototype_shape[0]
    num_classes = model.num_classes
    prototypes_per_class = num_prototypes // num_classes

    if num_prototypes_to_plot is None:
        num_prototypes_to_plot = min(num_prototypes, num_classes*2)
    
    # For each class plot some prototypes
    prototype_to_plot_per_class = max(1, num_prototypes_to_plot // num_classes)

    fig, axes = plt.subplots(num_classes, prototype_to_plot_per_class,
                             figsize=(5*prototype_to_plot_per_class, 4*num_classes))
    
    fig.suptitle("Prototype Distance Distributions: In-Class vs Out-Class", fontsize=16, fontweight='bold', y=0.995)

    for class_id in range(num_classes):
        proto_start = class_id * prototypes_per_class
        proto_end = (class_id + 1) * prototypes_per_class

        if prototype_to_plot_per_class >= prototypes_per_class:
            proto_indices = list(range(proto_start, proto_end)) # Plot all prototypes of the class
        else:
            proto_indices = list(range(proto_start, proto_start + prototype_to_plot_per_class))

        for idx, proto_idx in enumerate(proto_indices):
            ax = axes[class_id, idx]

            distances = all_min_distances[:, proto_idx].numpy()

            in_class_mask = (all_labels == class_id).numpy()
            distances_in_class = distances[in_class_mask]
            distances_out_class = distances[~in_class_mask]

            mean_intra_class = np.mean(distances_in_class)
            mean_inter_class = np.mean(distances_out_class)

            # Calculate ratio score
            ratio_score = mean_intra_class / (mean_inter_class + 1e-8)

            bins = np.linspace(distances.min(), distances.max(), 50)

            ax.hist(distances_in_class, bins=bins, alpha=0.6, color='green', 
                       label=f'Class {class_id} (n={len(distances_in_class)})', 
                       density=True, edgecolor='black', linewidth=0.5)
            ax.hist(distances_out_class, bins=bins, alpha=0.6, color='red', 
                    label=f'Other classes (n={len(distances_out_class)})', 
                    density=True, edgecolor='black', linewidth=0.5)
            
            # Calculate separation score
            separation = distances_out_class.mean() - distances_in_class.mean()
            
            ax.set_xlabel('Distance', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Prototype {proto_idx} (Class {class_id})\nSeparation: {separation:.2f}\nRatio Score: {ratio_score:.4f}', 
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Color the background based on quality
            if separation > 2.0:  # Good separation
                ax.set_facecolor('#e8f5e9')  
            elif separation < 0:  # Negative separation
                ax.set_facecolor('#ffebee')
            
    plt.tight_layout()

    # Save the figure 
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'prototypes_distance_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Distance distribution plot saved to {save_path}')
    
    plt.show()

def evaluate_confusion_matrix(model, data_loader):
    model.eval()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output, _ = model(data)

            preds = torch.argmax(output, dim=1)

            all_preds.append(preds)
            all_labels.append(target)

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()

    conf_mat = confusion_matrix(y_true, y_pred, normalize='true')

    print(f'Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%')
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='.2f')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('/data01/inginf2025darmez/SonicProtoPNet/saved_models/vgg19/confusion_matrix.png', dpi=300, bbox_inches='tight')    
    