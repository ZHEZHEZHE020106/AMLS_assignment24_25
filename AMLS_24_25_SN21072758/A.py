def gray_to_rgb(img):
        return img.convert("RGB")

def transformImage():
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(gray_to_rgb),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform

def ResNet18():
    import medmnist
    from medmnist import BreastMNIST
    from medmnist import INFO, Evaluator
    import torch
    from torch import nn
    import torchvision
    
    import torch.utils.data as data
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    

    transform = transformImage()

    info = INFO['breastmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', transform=transform, download='true')
    validation_dataset = DataClass(split='val', transform=transform, download='true')
    test_dataset = DataClass(split='test', transform=transform, download = 'true')

    classes = ("Malignant", "Benign" )   #(cancerous) or (non-cancerous)
    batch_size = 128

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    Classification_Model = torchvision.models.resnet18(weights = True)

    num_in_features = Classification_Model.fc.in_features
    for param in Classification_Model.parameters():
        param.requires_grad = False 
    # Replace the last fc layer 
    Classification_Model.fc = nn.Sequential(
        nn.Linear(num_in_features,2),
                                           )  
    print(Classification_Model)

    if torch.cuda.is_available():
        Classification_Model = Classification_Model.cuda()
    
    #loss function
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    
    
    #optimizer
    learning_rate = 0.001
    epoch = 100
    
    optimizer = torch.optim.Adam(Classification_Model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(Classification_Model.parameters(),lr=learning_rate,momentum=0.9)
    
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    
     
    #-------------------------------------Train----------------------------------------
    
    
    
    total_train_step = 0
    Train_accuracy_epoch = []
    Train_loss_epoch = []
    Val_accuracy_epoch = []
    Val_loss_epoch = []
    
    for i in range(epoch):
        print("-------Epoch {} Start-------".format(i+1))
        Classification_Model.train()
        correct = 0
        total = 0
        average_loss = 0
    
        for data in train_loader:
            inputs, targets = data
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
    
            outputs = Classification_Model(inputs)
    
            targets = targets.to(torch.float32)
            targets = targets.squeeze(dim=1).long()
            loss = loss_fn(outputs, targets)
    
            predict = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += (predict == targets).sum().item()
            average_loss += loss.item()
     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
            total_train_step = total_train_step + 1
            # print every 10 batches
            if total_train_step % 10 == 0:
                print("Train Count：{}, Loss: {}".format(total_train_step, loss.item()))
                #print(correct, total)
                print('Accuracy :', 100*correct/total)
                
        Train_accuracy_epoch.append(100*correct/total)
        Train_loss_epoch.append(average_loss/len(train_loader))
        
        print("-------Epoch {} Completed-------".format(i+1))
        
    #-------------------------------------Validation-----------------------------------------
        Classification_Model.eval()  # Set model to evaluation mode
    
        with torch.no_grad():  # Disable gradient computation for validation
            correct = 0
            total = 0
            average_loss = 0
    
            for data in val_loader:  # Use the validation data loader
                inputs, targets = data
            
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
    
                outputs = Classification_Model(inputs)
                targets = targets.to(torch.float32)
                targets = targets.squeeze(dim=1).long()
            
                loss = loss_fn(outputs, targets)
                predict = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predict == targets).sum().item()
                average_loss += loss.item()
    
            Val_accuracy_epoch.append(100 * correct / total)
            Val_loss_epoch.append(average_loss / len(val_loader))
        
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")
            print(f"Validation Loss: {average_loss / len(val_loader):.4f}")
            
        #scheduler.step()

    import matplotlib.pyplot as plt
    # plotting curves
    

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), Val_loss_epoch, label='Validation Loss')
    plt.plot(range(1, epoch + 1), Train_loss_epoch, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), Val_accuracy_epoch, label='Validation Accuracy')
    plt.plot(range(1, epoch + 1), Train_accuracy_epoch, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    Classification_Model.eval()
    correct = 0
    total = 0
    
    # use for calculate AUC
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
    
            outputs = Classification_Model(inputs)
    
            targets = targets.to(torch.float32)
            targets = targets.squeeze(dim=1).long()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum()
            
            all_targets.extend(targets.cpu().numpy())  # true label
            all_outputs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    #print(correct, total)
    #print('Accuracy: ', correct/total)
    
    acc = correct / total
    auc = roc_auc_score(all_targets, all_outputs)
    
    print(f'Accuracy: {acc:.4f}')
    print(f'AUC: {auc:.4f}')

def RandomForest():
    #Random Forest
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    from sklearn import tree
    import medmnist
    from medmnist import BreastMNIST
    from medmnist import INFO, Evaluator
    import torchvision.transforms as transforms
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import warnings
    import matplotlib.pyplot as plt

    DataClass = getattr(medmnist, INFO['breastmnist']['python_class'])
    transform = transformImage()

    train_dataset = DataClass(split='train',transform=transform, download='true')
    validation_dataset = DataClass(split='val',transform=transform, download='true')
    test_dataset = DataClass(split='test',transform=transform, download = 'true')

    X_train = np.array([item[0].numpy().flatten() for item in train_dataset])
    y_train = np.array([item[1] for item in train_dataset])
    X_val = np.array([item[0].numpy().flatten() for item in validation_dataset])
    y_val = np.array([item[1] for item in validation_dataset])
    X_test = np.array([item[0].numpy().flatten() for item in test_dataset])
    y_test = np.array([item[1] for item in test_dataset])

    
    warnings.filterwarnings('ignore')

    rf_clf = RandomForestClassifier(n_estimators=9, random_state=42)
    
    
    rf_clf.fit(X_train, y_train)
    
    
    y_val_pred = rf_clf.predict(X_val)
    
    
    y_test_pred = rf_clf.predict(X_test)
    
    
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    def visualise_tree(tree_to_print):
        plt.figure()
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
        tree.plot_tree(tree_to_print,
                   feature_names = [f'pixel_{i}' for i in range(X_train.shape[1])],
                   class_names=["Benign", "Malignant"]
    , 
                   filled = True,
                  rounded=True);
        plt.show()

    for index in range(0, 1):
        visualise_tree(rf_clf.estimators_[index])

def main():
    print("-------------------------------------Task 1 start---------------------------------------")
    print("Running ResNet-18")
    ResNet18()
    print("Running Random Forest")
    RandomForest()
    print("-------------------------------------Task 1 finished---------------------------------------")
if __name__ == "__main__":
    main()
