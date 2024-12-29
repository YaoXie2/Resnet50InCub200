import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm

# from model import resnet50
import matplotlib.pyplot as plt

print('start... hello world')

for i in range(100):
    print(i)
    
for j in range(100):
    print('xxxxxxxxxxx')

for k in range(3):
    print(f'aaaaa')

def main():
    # device = torch.device("cuda:0,cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                    transforms.Resize(512),
                                    transforms.RandomResizedCrop(448), # 随机选取原输入图片中的448×448的部分
                                     transforms.RandomHorizontalFlip(), # 随机旋转一定角度
                                     transforms.ToTensor(),             # 转化为tensor矩阵
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(512),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    '''
    下面这一步设置了数据集的labels，具体逻辑就是
    前期工作:用户需要在root路径下,把每一类的图片单独归纳成一个文件夹
    datasets.ImageFolder的作用: 该方法能够自动生成一个元组(图片,图片所在文件夹的下标值)
    '''
    train_dataset = datasets.ImageFolder(root="./train", # 这一步已经确定了每一幅图像对应的答案应该是文件夹的索引值
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    bird_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in bird_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="./val",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))

    # ----------------迁移学习部分-----------------------------------------------------------------
    net = models.resnet50(pretrained=False)
    model_weight_path = "./resnet50-19c8e357.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path))
    # for param in net.parameters():
    #     param.requires_grad = False
    # ----------------------------------------------------------------------------------------------


    # change fc layer structure
    # 这里需要先获取到预训练模型中的最后一个输出的全连接层然后再构建一个全连接层
    in_channel = net.fc.in_features 
    net.fc = nn.Linear(in_channel, 200) # 因为最后只需要划分出200个类别,所以要把原来的全连接层输出改为200
    net.to(device)
    # 进行分布式训练
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:6666', rank=0, world_size=1)
    # net = nn.parallel.DistributedDataParallel(net)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.002,weight_decay=0.00005,momentum=0.9)

    epochs = 88
    best_acc = 0.0
    save_path = './resNet50.pth'
    train_steps = len(train_loader)
    val_accuracy_list = []
    train_accuracy_list=[]
    epochs_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        # train
        net.train()
        if(epoch==20):
            optimizer = optim.SGD(net.parameters(), lr=0.001,weight_decay=0.00005,momentum=0.9)
        elif(epoch==30):
            optimizer = optim.SGD(net.parameters(), lr=0.0005,weight_decay=0.00005,momentum=0.9)
        elif(epoch==50):
            optimizer = optim.SGD(net.parameters(), lr=0.0001,weight_decay=0.00005,momentum=0.9)            
        
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

        # validate
        net.eval()
        val_acc = 0.0  # 累计验证集中的所有正确答对的个数
        train_acc=0.0  # 累计训练集中所有正确答对的个数
        val_loss = 0.0 # 累计验证集中所有误差
        train_loss=0.0 # 累积训练集中所有误差
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for train_data in train_bar:
                train_images,train_labels=train_data
                train_outputs=net(train_images.to(device))
                tmp_train_loss=loss_function(train_outputs,train_labels.to(device))
                train_predict=torch.max(train_outputs,dim=1)[1]
                train_acc+=torch.eq(train_predict, train_labels.to(device)).sum().item()                
                train_loss+=tmp_train_loss.item()
                train_bar.desc = "valid in train_dataset epoch[{}/{}]".format(epoch + 1,epochs)
            
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_outputs = net(val_images.to(device))
                tmp_val_loss = loss_function(val_outputs, val_labels.to(device))
                val_predict = torch.max(val_outputs, dim=1)[1]
                val_acc += torch.eq(val_predict, val_labels.to(device)).sum().item()
                val_loss+=tmp_val_loss.item()               
                val_bar.desc = "valid in val_dataset epoch[{}/{}]".format(epoch + 1,epochs)

        train_accurate=train_acc/train_num
        val_accurate = val_acc / val_num

        if(val_accurate>best_acc):
            best_acc=val_accurate
        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f' 
        %(epoch + 1, train_loss / train_num, train_accurate, val_loss/val_num, val_accurate))

        # 构造各个参数的列表，准备画图
        val_accuracy_list.append(val_accurate)
        train_accuracy_list.append(train_accurate)
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss/ val_num )
        epochs_list.append(epoch+1)
        torch.save(net.state_dict(), save_path) # 保存模型

        # train_acc && val_loss
        plt.figure()
        plt.plot(epochs_list,val_accuracy_list,color="red",label="val_acc")
        plt.plot(epochs_list,train_accuracy_list,color="green",label="train_acc")
        plt.xlabel("epochs")
        plt.ylabel("Acc")
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0,len(epochs_list),20)])
        acc_gap=[i*0.2 for i in range(0,min(int(len(epochs_list)/2+1),6))]
        acc_gap.append(max(val_accuracy_list))
        acc_gap.append(max(train_accuracy_list))
        plt.yticks(acc_gap)
        plt.grid()
        plt.legend()
        plt.savefig("Acc.jpg")

        # train_loss && val_loss
        plt.figure()
        plt.plot(epochs_list,train_loss_list,color="red",label="train_loss")
        plt.plot(epochs_list,val_loss_list,color="green",label="val_loss")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0,len(epochs_list),20)])
        plt.grid()
        plt.legend()
        plt.savefig("Loss.jpg")

    print('Finished Training')
    print("the best val_accuracy is : {}".format(best_acc))

if __name__ == '__main__':
    main()
