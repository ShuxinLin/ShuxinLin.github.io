---

layout: post

title: DEEP LEARNING 笔记 part.1
---

很早已经看的一个deep learning的博客[^blog]，一点笔记供以入门。

[^blog]: http://blog.csdn.net/zouxy09/article/details/8775360

<!-- more -->

## 机器学习与深度学习
机器学习：`pre-processing` -> `feature extract.` -> `feature selection` -> `inference,prediction`

`feature特征表达` 对于最终算法的准确性非常关键，计算和测试的主要工作都耗在这一部分。但，这块实际上都是人工完成。靠人工提取feature。

好的特征：不变性与可区分性。
- 不变性: 大小、尺度、旋转等。
- 可区分性：
> sift 算法

Deep learning (unsupervised feature learning): 为了自动学习一些特征。
> deep learning 的比较正式的定义： 
> There are a number of ways that the field of deep learning has been characterized. Deep learning is a class of machine learning training algorithms that：
> - use many layers of nonlinear processing units for feature extraction and transformation. The algorithms may be supervised or unsupervised and applications include pattern recognition and statistical classification.
> - are based on the (unsupervised) learning of multiple levels of features or representations of the data. Higher level features are derived from lower level features to form a hierarchical representation.
> - are part of the broader machine learning field of learning representations of data.
> - learn multiple levels of representations that correspond to different levels of abstraction; the levels form a hierarchy of concepts.
> - form a new field with the goal of moving toward artificial intelligence. The different levels of representation help make sense of data such as images, sounds and texts

## Feature
特征对机器学习非常关键。对于一张图片，像素级别的特征没有意义。

### 结构性特征表示
小块的图形可以由基本的edge通过线性组合构成。更结构化，更复杂的，具有概念性的图片，需要更高层次的特征表示，如V2，V4。这是一个层次递进的过程，高层 表达由低层表达的组合而成。比如，V1提取出的basis是edge，则V2是V1这些basis的组合，这时V2又得到V3的basis，V3就由这些basis组合而成。
直观上说，就是找到小patch再将其进行组合，就得到上一层的feature，递归地向上learning feature。
> 一个人在看一个doc的时候，眼睛看到的是word（V1），由这些word在大脑里自动切词形成term(V2)，在按照概念组织的方式，先验的学习，得到topic(V3)，然后再进行高层次的learning。

### How many features?
任何一种方法，特征越多，给出的参考信息就越多，准确性会得到提升。但特征多意味着计算复杂，探索的空间大，可以用来训练的数据在每个特征上就会稀疏，都会带来各种问题，并不一定特征越多越好。

## deep learning 
以上得出的结论就是，Deep learning需要多层来获得更抽象的特征表达。

### deep learning的基本思想
假设我们由一个系统S，有n层（S1,...Sn），它的输入是I，通过n层后，输出是O。如果输出O等于输入I，代表没有信息损失。当这个思想应用到deep learning，则假设我们设计的这个系统S（n层），我们通过调整系统中的参数，使得我的输出等于输入，那么我们就可以自动地获取得到输入I的一系列层次特征，即S1,...Sn

 深度学习的实质，是通过构建具有很多隐层的机器学习模型和海量的训练数据，来学习更有用的特征，从而最终提升分类或预测的准确性。因此，“深度模型”是手段，“特征学习”是目的。区别于传统的浅层学习，深度学习的不同在于：1）强调了模型结构的深度，通常有5层、6层，甚至10多层的隐层节点；2）明确突出了特征学习的重要性，也就是说，通过逐层特征变换，将样本在原空间的特征表示变换到一个新特征空间，从而使分类或预测更加容易。与人工规则构造特征的方法相比，利用大数据来学习特征，更能够刻画数据的丰富内在信息。

### 深度学习 vs 浅层学习

当前的多数回归、分类问题为浅层学习，或者具有一层hidden layer的浅层学习。其局限性在于有限样本和计算单元情况下对复杂函数的表示能力有限，针对复杂分类问题其泛化能力受到一定制约。
深度学习可通过学习一种深层非线性网络结构，实现复杂函数逼近，表征输入数据分布式表示，并展现了强大的从少数样本集中学习数据集本质特征的能力。（多层的好处是可以用较少的参数表示复杂的函数）

### 深度学习 vs 神经网络

深度学习的概念源于人工神经网络的研究，可以理解为神经网络的发展。
二者的相同在于都采用了神经网络相似的分层结构，系统由包括输入层、隐层（多层）、输出层组成的多层网络，只有相邻层节点之间有连接，同一层以及跨层节点之间相互无连接，每一层可以看作是一个logistic regression模型。
训练机制上：神经网络采用back propagation算法，deep learning采用layer-wise机制。
> BP算法: 采用迭代的算法来训练整个网络，随机设定初值，计算当前网络的输出，然后根据当前输出和label之间的差去改变前面各层的参数，直到收敛（整体是一个梯度下降法）。
> 如果采用back propagation的机制，对于一个deep network（7层以上），残差传播到最前面的层已经变得太小，出现所谓的gradient diffusion（梯度扩散）。
> - 梯度越来越稀疏：从顶层越往下，误差校正信号越来越小；
> - 收敛到局部最小值：尤其是从远离最优区域开始的时候（随机值初始化会导致这种情况的发生）
- 我们只能用有标签的数据来训练：但大部分的数据是没标签的，而大脑可以
从没有标签的的数据中学习

### deep learning训练过程

使用自下而上的非监督学习:
采用无标定数据分层训练各层参数。训练时先学习这一层的参数，是的输出与输入差别最小，能够学习到数据本身的结构，得到比输入更具有表示能力的特征。然后将输出作为下一层的输入，分别得到各层的参数。
自顶而下的监督学习：
通过带标签的数据训练，误差自顶向下传输，对网络进行微调。

## autoencoder
自动编码器：尽可能复现输入信号，
通过调整encoder和decoder的参数，使得重构误差最小，就可以得到输入的编码code，也就是特征；再经过多层编码，保证输入输出都是相似，且都是原始输入的不同表达。我们觉得它越抽象越好。

为了实现分类，我们在antoencoder的最顶添加一个分类器，然后通过监督训练方法去训练。
我们可能需要通过监督学习进行微调：只调整分类器，或调整整个系统。

## Sparse coding 稀疏编码
稀疏编码算法是一种无监督学习方法，它用来寻找一组完备的基向量来更高效地表示样本数据。它们能更有效地找出隐含在输入数据内部的结构与模式。

**步骤**：
`training`：给定样本，学习得到一组基，也就是字典集。样本就是有这个基的线性组合。
如何选择基：
目标函数|线性组合-原样本|最小。
未知数：字典集和权重值。
固定字典集，调整权重，使目标函数最小。
固定权重，调整字典集，使目标函数最小。
直至收敛，得到字典集。

`coding`：
给定一个新的输入样本，由上面得到的字典集，为使目标函数最小，求得权重值。从而得到稀疏向量：基的线性组合。
这个稀疏向量就是输入样本的稀疏表达。

> 稀疏性：使得权重的个数不为零的只有少量，即只有少量的基进行叠加。
