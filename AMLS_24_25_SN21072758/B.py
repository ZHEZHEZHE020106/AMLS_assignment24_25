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

def ResNet50():
    import medmnist
    from medmnist import BloodMNIST
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

    info = INFO['bloodmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', transform=transform, download='true')
    validation_dataset = DataClass(split='val', transform=transform, download='true')
    test_dataset = DataClass(split='test', transform=transform, download = 'true')

    classes = ("basophil", "eosinophil","erythroblast", "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)","lymphocyte", "monocyte","neutrophil", "platelet" )  
    batch_size = 128

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    Classification_Model = torchvision.models.resnet50(weights = True)
    num_in_features = Classification_Model.fc.in_features
    for param in Classification_Model.parameters():
        param.requires_grad = False 
        # Replace the last fc layer 
    Classification_Model.fc = nn.Sequential(nn.Linear(num_in_features,8))  
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

    optimizer = torch.optim.SGD(Classification_Model.parameters(),lr=learning_rate,momentum=0.9)
    #------------------------------------------------Train-------------------------------------------------
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
    
    
#------------------------------------------------Validation-------------------------------------------------

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
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), Val_loss_epoch, label='Validation Loss')
    plt.plot(range(1, epoch + 1), Train_loss_epoch, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Validation and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), Val_accuracy_epoch, label='Validation Accuracy')
    plt.plot(range(1, epoch + 1), Train_accuracy_epoch, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation and Train Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Testing

    Classification_Model.eval()
    correct = 0
    total = 0

    # for calculating AUC
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for data in val_loader:  # 验证集数据加载器
            inputs, targets = data
        
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = Classification_Model(inputs)
            _, predicted = torch.max(outputs, 1)
            targets = targets.to(torch.float32)
            targets = targets.squeeze(dim=1).long()
            total += targets.size(0)
            correct += (predicted == targets).sum()
        
            all_targets.extend(targets.cpu().numpy())
            #all_outputs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_outputs.append(torch.softmax(outputs, dim=1).cpu().numpy())


    all_targets = np.array(all_targets)
    all_outputs = np.concatenate(all_outputs, axis=0)

    print("Correct/Total :", correct, "/", total)
    # calculating accuracy
    acc = correct / total
    print(f'Accuracy: {acc:.4f}')

    # calculating auc
    
    auc = roc_auc_score(all_targets, all_outputs, multi_class='ovr')  # 'ovr' means one-vs-rest
    print(f'AUC: {auc:.4f}')
        


def RandomForest():
    #Random Forest
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    from sklearn import tree
    import medmnist
    from medmnist import BloodMNIST
    from medmnist import INFO, Evaluator
    import torchvision.transforms as transforms
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import warnings
    import matplotlib.pyplot as plt

    DataClass = getattr(medmnist, INFO['bloodmnist']['python_class'])
    transform = transformImage()

    train_dataset = DataClass(split='train',transform=transform, download='true')
    validation_dataset = DataClass(split='val',transform=transform, download='true')
    test_dataset = DataClass(split='test',transform=transform, download = 'true')

    X_train = np.array([img[0].numpy().flatten() for img, label in train_dataset])
    y_train = np.array([label for img, label in train_dataset])

    X_val = np.array([img[0].numpy().flatten() for img, label in validation_dataset])
    y_val = np.array([label for img, label in validation_dataset])

    X_test = np.array([img[0].numpy().flatten() for img, label in test_dataset])
    y_test = np.array([label for img, label in test_dataset])

    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_val_pred = rf_clf.predict(X_val)
    y_test_pred = rf_clf.predict(X_test)


    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    
    warnings.filterwarnings('ignore')


def SVM():
    import numpy as np
    import medmnist
    from medmnist import BloodMNIST
    from medmnist import INFO, Evaluator
    import torchvision.transforms as transforms
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    DataClass = getattr(medmnist, INFO['bloodmnist']['python_class'])
    transform = transformImage()

    train_dataset = DataClass(split='train',transform=transform, download='true')
    validation_dataset = DataClass(split='val',transform=transform, download='true')
    test_dataset = DataClass(split='test',transform=transform, download = 'true')

    # Load data and labels
    X_train, y_train = train_dataset.imgs, train_dataset.labels
    X_val, y_val = validation_dataset.imgs, validation_dataset.labels
    X_test, y_test = test_dataset.imgs, test_dataset.labels
    
    x_train = np.array(train_dataset.imgs) 
    y_train = np.array(train_dataset.labels)

    x_val = np.array(validation_dataset.imgs) 
    y_val = np.array(validation_dataset.labels)

    x_test = np.array(test_dataset.imgs) 
    y_test = np.array(test_dataset.labels)

    # flatten images
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_val_flatten = x_val.reshape(x_val.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)
    # standardlization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flatten)
    x_val_scaled = scaler.fit_transform(x_val_flatten)
    x_test_scaled = scaler.transform(x_test_flatten)

    # train
    svm = SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    svm.fit(x_train_scaled, y_train.ravel())

    # evluating on validation set
    y_val_pred = svm.predict(x_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.2f}')
    print('Validation Classification Report:')
    print(classification_report(y_val, y_val_pred, target_names=INFO['bloodmnist']['label']))

    # testing
    y_test_pred = svm.predict(x_test_scaled)

    # evaluating on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print('Test Classification Report:')
    print(classification_report(y_test, y_test_pred, target_names=INFO['bloodmnist']['label']))

def main():
    print("-------------------------------------Task 2 start---------------------------------------")
    print("Running ResNet-50")
    ResNet50()
    print("Running Random Forest")
    RandomForest()
    print("Running SVM")
    SVM()
    print("-------------------------------------Task 2 finished---------------------------------------")
if __name__ == '__main__':
    main()