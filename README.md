以下源自我的csdn博客内容: [入门ResNet，在Cub200数据集上复现Resnet50](https://blog.csdn.net/messyking/article/details/113933317?spm=1001.2014.3001.5501)
# 1.**背景问题**
(1).如果只是单纯地把卷积层和池化层进行堆叠，造成的问题就会有梯度消失和梯度爆炸，梯度消失是指当在某一层进行BP的时候，误差为一个小于零的数，那不断相乘，就会趋近于零。梯度爆炸则是指某一层的开始误差都是大于1的数，直接相乘就会导致梯度爆炸。这种情况的处理方法就是对数据进行标准化处理和bn标准化处理特征图。
(2).退化问题就是本来训练到20层已经达到了99%，但是30层训练之后的正确率反而小于99%。resnet通过残差结构(就是那条捷径)解决了退化问题，使之层数增加，准确率是一直增加。所以它可以一直堆叠层数，从而提高识别准确率。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5fa86b784a9579eb67e44d6d68988243.png)
# 2.**ResNet的亮点**
(1) 网络堆叠层数增加
(2) 提出了residual模块
(3) 提出bn，取代了dropout方法
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c23bf943f3b458e85b1b6d1bfde80926.png)
# 3.**ResNet中层数不同的残差结构**

==首先需要明确几个概念，残差层是指Conv_2，Conv_3这种包含多个残差结构的组合，然后残差结构是指下图中单个residual结构==
(1) 下面是两种残差结构(也就是残差层里的基本组成单位)，左边是用于ResNet34，右边是用于ResNet50/101/152 **(默认stride都是1)**
(2) 途中画方框的地方是这两种残差结构的input的channel都是256，各自所需要的参数个数。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0f4058778297143778fc680470c77996.png)
# 4.**ResNet中同样层数的残差结构捷径是虚线和捷径实现的区别**
(1) 首先，带有虚线的残差结构只在ResNet18,34中的Conv_3，4,5的第一个残差结构出现，和ResNet50.101.152中的每一残差层的第一个残差结构出现。
(2) 带虚线的残差结构的功能是来改变input的shape和channels的，比如ResNet34中的Conv_3中的第一个残差结构(如下图右边那个residual结构)，主线的第一个卷积层因为stride=2,所以能够将shape降为原来的一半，同时也将channels改为了128，第二个卷积层的stride=1，所以shape不变，channels继续保持输出128，虚线的卷积层的stride=2，改变了shape，channels变成了128。同理ResNet50的Conv_3的第一个残差结构就是下下图。
(3) 除了ResNet18,34中的Conv_2层之外，其余类型的残差层都是虚线残差结构+实现残差结构的组合。并且一般第一个残差结构是虚线，然后接下去的都是实线残差结构。而虚线的残差结构是用来改变input的shape和channels的，实现是保持虚线改变之后的shape和channels直到输出到下一个Conv。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c3d79be5750ac4be3066543614c2e62a.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/865469a04c333d5275e875863bdc0056.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/89d37eff6a222bc0a21ffb9358f597b4.png)

# 5.**BN层**
## (1) **BN与Dropout的区别**
dropout只适合用于全连接层的失活，而BN更适用于卷积层得出的特征图，且降低过拟合的效果更佳。
[dropout VS bn](http://www.360doc.com/content/19/0318/21/99071_822522365.shtml)
## (2) **背景**
对input的image进行normalization使之符合特定的正态分布能够加速网络的收敛。同理也可对学习过程中产生的feature map进行标准化处理。BN就是使得每一个feature map的每一个值都满足均值为0，方差为1的分布律。下面对下图中的参数进行一下解释。
## (3) **参数解释**
B：m幅特征图。
μB：一批特征图中所有同一个通道的数值均值，输入有多少个channel就有多少个μ
方差（字母不会打。。）：一批特征图中所有同一个通道的数值方差
xi：对每一个特征图中的每一个维度中的数值进行正值化后的数值
> 因为如果只是通过xi来得出的只是一个均值为零，方差为1的分布，有时候不一定是最好的分布，所以要在下面设置一个线性变换通过深度学习确定两个参数α和β，从而找到一个加速收敛效果最佳的分布

## (4)**需要注意的问题**
1)训练时要将traning参数设置为True，在验证时将trainning参数设置为False。在pytorch中可通过创建模型的model.train()和model.eval()方法控制。

2)batch size尽可能设置大点，设置小后表现可能很糟糕，设置的越大求的均值和方差越接近整个训练集的均值和方差。

3)建议将bn层放在卷积层（Conv）和激活层（例如Relu）之间，且卷积层不要使用偏置bias，因为没有用，参考下图推理，即使使用了偏置bias求出的结果也是一样的。

[BN算法解析](https://blog.csdn.net/qq_37541097/article/details/104434557)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a7106e142b9d6043588c6843f43ab497.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5d49fdd8dfb3afbbaae91eeaee7af8cd.png)
## (5) **各个类型的resnet的下载地址**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/270cdd90a6dd4336990e215b8eb3ce15.png)
# 6.matplotlib画图小技巧
(1) [画两幅图的方法](https://www.zhihu.com/question/280931066?sort=created)

# 7.调参过程遇到的问题及解决方法(以笔者自己的遇到的cub数据集为例)
## (1) 导入pretrain
预训练模型，一般使用pytorch官网预训练的模型，上面有下载链接。

## (2) 数据集预处理(也叫数据增强)
Resize到512*512，然后randomCrop取448 * 448

## (3) solver(优化器的选取)
**1)优化器算法原理步骤**
>1.计算目标函数关于当前参数的梯度
>2.根据历史梯度计算一阶动量和二阶动量
>3.计算当前时刻的下降梯度
>4.根据下降梯度进行更新

**2)各种优化器及其发展过程**
固定学习率的优化算法
>==BGD==：Batch Gradient Descent，用整个训练集去求梯度然后更新整个训练集。
==SGD==：Stochastic Gradient Descent，随机选取一个mini batch求出对应的梯度之后用这个梯度去更新整个权重矩阵。
==SGDM==：Stochastic Gradient Descent with Momentum，SGD的基础上引入了一阶动量Momenbtum（一阶动量是各个时刻梯度方向的指数移动平均值，约等于最近 1/(1-β1) 个时刻的梯度向量和的平均值。），SGD最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。加入动量可以抑制这个震荡，能够加速收敛。
==NAG==：Nesterov Accelerated Gradient，在SGDM基础上修改，就是先按照惯性先走一步，然后再看当前梯度再用SGD走。

自适应学习率的优化算法
>==AdaGrad==:上面的固定学习率优化算法并没有考虑到不同的权重参数对学习率的要求不同，可能一些权重经过大量的训练之后就只需要进行微小的细调就可以了，而一些权重参数可能还需要进行较大的学习变化，那怎么判断那些参数已经训练过很多次不用再进行大调呢，这时候就需要引入二阶动量（该维度上，迄今为止所有梯度值的平方和），更新越频繁，二阶动量就越大，对应学习率就应该越小，反之，从而进行自适应调整学习率。
==AdaDelta / RMSProp==:不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。避免了二阶动量持续累积、导致训练过程提前结束的问题。
==Adam==:我们看到，SGD-M在SGD基础上增加了一阶动量，AdaGrad和AdaDelta在SGD基础上增加了二阶动量。把一阶动量和二阶动量都用起来，就是Adam了——Adaptive + Momentum。
[各种优化器](https://blog.csdn.net/jiachen0212/article/details/80086926)

**3)SGD的理解**
原理概括
>总的来说，SGD就是随机选取一个mini batch的数据然后求出这个mini batch的损失函数的梯度然后用这个梯度去更新总的权重矩阵。

对比于BGD的优势
>（a）BGD容易陷入original-loss的奇点，而SGD不容易陷入；
>（b）SGD也不会陷入minibatch-loss的奇点。
（具体解释可以看这个参考链接，最重要就是sgd就是引入了一个randomness，使之能较快跳出"困局"）

SGD的劣势
>SGD走的路径比较曲折（震荡），就是花的时间比较长，尤其是batch比较小的情况下。
[sgd详解](https://blog.csdn.net/weixin_46301248/article/details/105883723?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242)

**4) SGD VS Adam**
>Adam等自适应学习率算法对于稀疏数据具有优势，且收敛速度很快；但精调参数的SGD（+Momentum）往往能够取得更好的最终结果。
[神经网络优化算法如何选择Adam，SGD](https://blog.csdn.net/u014381600/article/details/72867109)

>**优化算法的tricks**
>(1) 首先，各大算法孰优孰劣并无定论。如果是刚入门，优先考虑SGD+Nesterov Momentum或者Adam。
>(2) 选择你熟悉的算法——这样你可以更加熟练地利用你的经验进行调参。
>(3) 充分了解你的数据——如果模型是非常稀疏的，那么优先考虑自适应学习率的算法。
>(4) 根据你的需求来选择——在模型设计实验过程中，要快速验证新模型的效果，可以先用Adam进行快速实验优化；在模型上线或者结果发布前，可以用精调的SGD进行模型的极致优化。
>(5) 先用小数据集进行实验。有论文研究指出，随机梯度下降算法的收敛速度和数据集的大小的关系不大。.因此可以先用一个具有代表性的小数据集进行实验，测试一下最好的优化算法，并通过参数搜索来寻找最优的训练参数。
>(6) 考虑不同算法的组合。先用Adam进行快速下降，而后再换到SGD进行充分的调优。切换策略可以参考本文介绍的方法。数据集一定要充分的打散（shuffle）。这样在使用自适应学习率算法的时候，可以避免某些特征集中出现，而导致的有时学习过度、有时学习不足，使得下降方向出现偏差的问题。
>(7) 训练过程中持续监控训练数据和验证数据上的目标函数值以及精度或者AUC等指标的变化情况。对训练数据的监控是要保证模型进行了充分的训练——下降方向正确，且学习率足够高；对验证数据的监控是为了避免出现过拟合。
>(8) 制定一个合适的学习率衰减策略。可以使用定期衰减策略，比如每过多少个epoch就衰减一次；或者利用精度或者AUC等性能指标来监控，当测试集上的指标不变或者下跌时，就降低学习率。
[Adam那么棒，为什么还对SGD念念不忘 (3)—— 优化算法的选择与使用策略](https://www.iteye.com/blog/wx1568037608-2444867)

## (4) 深度学习中的超参数
**1)超参数列表**([深度学习中的超参数](https://blog.csdn.net/aoxuerenwudi/article/details/109208500))
>==momentum==:动量，sgd收敛速度较慢，加上动量，可以加大收敛速度和更有效避开鞍点。一个小的trick是，当刚开始训练的时候，把动量设小，或者直接就置为0，然后慢慢增大冲量，有时候效果比较好。
>==weight decay==: 权重衰减，也就是正则项的前面的一个系数。
>> **正则项的定义**：就在原损失函数后面加上一个值，这个值等于所有的权重的平方和除以2n再乘以一个系数，那个系数就是weight decay。
>**正则项的作用**：
（1）使用正则项既不是为了提高收敛精确度也不是为了提高收敛速度，其最终目的是**防止过拟合**。所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
（2）如果过拟合，调大这个参数；如果欠拟合，调小这个参数。
> **正则项防止过拟合的原理**：
（1）首先L2正则项具有降低w的作用，从模型的复杂度上解释：较小的权值w，从某种意义上说，表示网络的复杂度较低，对数据的拟合更好（这个法则也叫做奥卡姆剃刀），而在实际应用中，也验证了这一点，L2正则化的效果往往好于未经正则化的效果。
（2）从数学方面的解释：过拟合的时候，拟合函数的系数往往非常大，为什么？如下图所示，过拟合，就是拟合函数需要顾忌每一个点，最终形成的拟合函数波动很大。在某些很小的区间里，函数值的变化很剧烈。这就意味着函数在某些小区间里的导数值（绝对值）非常大，由于自变量值可大可小，所以只有系数足够大，才能保证导数值很大。而正则化是通过约束参数的范数使其不要太大，所以可以在一定程度上减少过拟合情况。
>
>==learning rate==:学习率，决定了权值更新的速度。设置得太大会使结果超过最优值，太小会使下降速度过慢。一个有效平衡训练时间和收敛精度的方法就是**学习率衰减**。
>[pytorch必须掌握的的4种学习率衰减策略](https://zhuanlan.zhihu.com/p/93624972)

## (5) 分布式训练
专门用于解决batch size规模大于单个gpu内存的情况。
[分布式训练](https://zhuanlan.zhihu.com/p/86441879)

## (6) 输出网络中的参数的方法
[model.parameters()与model.state_dict()](https://zhuanlan.zhihu.com/p/270344655)
[pretrain数据与导入之后的数据产生差别的原因](https://blog.csdn.net/qq_34132310/article/details/107384294)

**最后附上本人复现resnet50的github链接**[Resnet50InCub200](https://github.com/YaoXie2/Resnet50InCub200)
