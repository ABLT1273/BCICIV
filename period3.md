EEG分类模型复现指南：TCN、ATCNet、DRSN与LaBraM
目录
时序增强期：长程依赖阶段（2020–2021年）
TCN：引入膨胀卷积的长程依赖建模
EEGNet：轻量级卷积基准模型
TCN在EEG分类中的PyTorch实现要点
融合期：注意力机制阶段（2022–2023年）
ATCNet：TCN与注意力机制的融合
注意力机制的作用：学会“抓重点”
ATCNet的PyTorch实现要点
跨越期：去噪与鲁棒性阶段（2023–2024年）
深度残差收缩网络（DRSN）：自动去噪的鲁棒模型
DRSN在EEG分类中的应用
DRSN的PyTorch实现要点
爆发期：大模型与通用表征阶段（2024年至今）
LaBraM：迈向通用“大脑表征”的大模型
LaBraM的PyTorch复现要点
其他EEG大模型进展
结语
时序增强期：长程依赖阶段（2020–2021年）
TCN：引入膨胀卷积的长程依赖建模
在EEG分类的早期，常用的一维卷积神经网络（如EEGNet等）往往受限于扫描窗口短，难以捕捉长时段的时序依赖。为解决这一问题，时间卷积网络应运而生。TCN通过膨胀卷积扩大感受野，使模型不仅能看到局部的波形，还能感知数秒之前的信号，从而有效捕捉长程依赖。与传统RNN/LSTM相比，TCN采用因果卷积结构，训练过程更稳定，不易出现梯度消失/爆炸问题。

TCN在EEG分类任务中表现出色，例如在运动想象分类中，TCN模块可以提取高级时序特征，显著提升分类准确率。同时，TCN还可与空间特征提取模块结合，如MASA-TCN通过空间感知时序层在EEG情绪分类中同时提取空间-频谱模式。总体而言，TCN的引入为EEG分类模型提供了更广阔的时序视野，奠定了后续模型融合注意力机制的基础。

EEGNet：轻量级卷积基准模型
在讨论TCN之前，需要提到EEGNet这一经典基线模型。EEGNet是2019年提出的轻量级卷积网络，专为EEG信号设计，通过时域卷积 + 深度可分离卷积提取时空特征。它参数量小（约2K）却能在多种EEG任务中取得不错效果，成为广泛采用的基准模型。

EEGNet使用较小的卷积核（例如64点的时间核）处理EEG片段，虽然能提取局部时序特征，但对长时依赖的建模能力有限。这促使研究者探索更大感受野的时序模型，如TCN。TCN在架构上可以视为对EEGNet时序部分的扩展，通过膨胀卷积将局部感受野扩展到全局，从而弥补EEGNet在长程依赖上的不足。

TCN在EEG分类中的PyTorch实现要点
在PyTorch中复现TCN模型，关键在于构建膨胀因果卷积模块。PyTorch的nn.Conv1d支持dilation参数来实现膨胀卷积。需要注意以下几点：

因果卷积：确保模型只依赖过去的信息，不使用未来数据。可以通过在卷积层前添加适当大小的零填充（仅填充序列左侧）来实现，或在PyTorch中使用padding=0并手动调整输入。一些库（如pytorch-tcn）已经封装了可切换因果/非因果的卷积层，可直接使用。
膨胀率设置：通常按照2的幂次递增膨胀率（例如第一层dilation=1，第二层dilation=2，第三层dilation=4，以此类推）。这能让卷积核的感受野呈指数级增长，迅速覆盖整个序列长度。例如，若输入EEG片段长度为1秒，采样率250Hz，总样本点250，一个膨胀率为[1,2,4,8,16,…]的TCN堆叠若干层后，顶层卷积核可"看到"整个1秒甚至更长的历史信号。
残差连接：TCN借鉴了ResNet的结构，每个膨胀卷积块通常包含残差连接以稳定训练。实现时可参考ResNet的基本块结构，在每个卷积层后加批归一化和ReLU激活，然后将输入加到输出上，形成残差单元。
网络深度与感受野：根据任务需要调整TCN的深度（堆叠多少残差块）。理论上，足够深的TCN可以感受整个输入序列。但过深可能导致梯度消失或计算开销增加。实践中常设置膨胀率重置策略，当膨胀率增长到一定程度后重新从1开始循环，以避免过大的填充和内存占用。
下面给出一个简化的PyTorch TCN模块实现示例，用于提取一维EEG信号的长程特征：


import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(self.bn(out))
        return out

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation  # 因果卷积需要的左侧填充
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, padding))
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
上述代码构建了一个多层膨胀卷积网络。每个TCNBlock包含一个卷积层、批归一化和ReLU激活。通过递增dilation，每一层都能以不同尺度"回看"过去的信号。最终，TCN模型可以将原始EEG信号转换为包含长程时序特征的高维表示，用于后续分类器（如全连接层）输出分类结果。

融合期：注意力机制阶段（2022–2023年）
ATCNet：TCN与注意力机制的融合
受到NLP领域自注意力机制成功的影响，EEG分类模型在2022年左右开始引入注意力机制，以学会"抓重点"。ATCNet（Attention Temporal Convolutional Network）是这一时期的代表模型之一，由Altaheri等人于2022年提出。

ATCNet在TCN基础上融合了多头自注意力机制，使模型能够自动识别EEG信号中哪些时间段对分类最有贡献。其架构可以概括为：卷积特征提取 → 时间窗口划分 → 注意力+TCN并行编码 → 分类。具体而言，ATCNet首先通过一个EEGNet风格的卷积块提取时空特征，然后将特征沿时间轴划分为若干重叠的滑动窗口。对于每个时间窗口，模型并行地通过多头自注意力模块和TCN残差块提取特征。注意力模块计算不同时间点的重要性权重，使模型关注关键波段；TCN模块则继续捕捉窗口内的长短期时序模式。最后，将各窗口的特征汇总（例如取平均或拼接）后送入分类层。

ATCNet的提出标志着EEG分类模型从单纯依赖时序卷积，转向卷积+注意力的融合架构。这种设计借鉴了Vision Transformer的思想，但针对EEG信号做了特殊优化：用卷积投影代替线性投影，用并行窗口编码器代替深层堆叠的Transformer编码器。实验表明，ATCNet在BCI竞赛IV 2a等运动想象数据集上取得了81%以上的分类准确率，显著优于当时的基准模型。这证明了注意力机制能够有效提升EEG分类性能，特别是在需要关注特定时间段的任务中。

注意力机制的作用：学会"抓重点"
注意力机制为EEG模型带来了可解释性和性能提升两方面的优势。一方面，通过可视化注意力权重，可以直观看到模型关注的信号片段。例如，在运动想象分类中，ATCNet往往对运动想象任务的提示时段和实际想象执行时段给予高权重，而对静息或准备时段关注度较低。这与人类专家的关注重点不谋而合，说明模型学会了"聚焦"于关键时间段。

另一方面，注意力机制通过加权聚合不同时间步的特征，提高了模型的区分能力。相比固定权重地处理所有时间步，自注意力能动态调整各时间步的贡献度，从而在复杂的脑电信号中提取更具判别力的模式。

ATCNet的PyTorch实现要点
在PyTorch中复现ATCNet，需要将卷积特征提取、多头自注意力和TCN三个模块有机结合。实现时可以参考Braindecode库中的ATCNet实现，或直接使用其提供的模型类。关键点包括：

特征提取卷积块：借鉴EEGNet的结构，先用一个时域卷积（例如64点长的一维卷积）提取时域特征，再用深度可分离卷积进行空间滤波。输出特征图的形状通常为(batch, channels, time_steps, 1)。
滑动窗口划分：将时间轴划分为若干窗口（例如5个窗口，每个窗口覆盖一定时间长度，窗口之间可以有重叠）。实现时可以在时间维度上使用nn.Unfold或手动切片来提取各窗口的特征。
多头注意力模块：为每个窗口的特征应用多头自注意力。PyTorch提供了nn.MultiheadAttention模块，可方便地实现这一功能。需要注意调整输入形状以符合该模块的要求（例如将(batch, channels, time_steps)转置为(time_steps, batch, channels)作为查询、键、值输入）。注意力模块的输出形状与输入相同，每个时间步的特征已经融合了全局信息。
TCN残差块：对注意力后的特征继续应用TCN模块进行时序建模。可以设计一个TCN残差块，包含两层膨胀卷积，并采用残差连接。由于每个窗口已经经过注意力融合全局信息，TCN在此可以进一步提取局部时序模式并保持梯度稳定。
并行处理与融合：注意力模块和TCN模块可以并行作用于每个窗口的特征，然后将结果相加或级联。最后，将所有窗口的特征聚合。一种方法是将各窗口的特征展平后拼接成一个长向量，输入全连接层分类；另一种方法是对各窗口的输出logits取平均作为最终预测。ATCNet原论文采用后者，即每个窗口独立输出分类结果再平均，以降低过拟合风险。
Braindecode库中的ATCNet实现提供了完整的PyTorch代码，可直接用于训练和测试。开发者也可以根据上述要点自行实现ATCNet的核心模块。需要注意调整超参数，例如注意力头数、TCN深度和膨胀率、窗口数量和大小等，以在特定数据集上取得最佳效果。总体而言，ATCNet的复现难度中等，但需要正确融合卷积、注意力和TCN三部分，是EEG分类模型从纯卷积向Transformer过渡的典型范例。

深度残差收缩网络（DRSN）：自动去噪的鲁棒模型
脑电信号噪声强、个体差异大，是EEG分类长期面临的挑战。为了提升模型在复杂干扰下的稳定性，研究者引入了深度残差收缩网络（Deep Residual Shrinkage Network, DRSN）。DRSN是在ResNet基础上融入软阈值收缩（Soft Shrinkage）机制的一种网络。其核心思想是：在残差块中插入自动收缩阈值模块，像“手术刀”一样实时剔除信号中的噪声成分。具体来说，DRSN在每层特征后增加一个可学习的阈值，对特征进行软阈值化（soft thresholding）操作，将接近于零的噪声特征置为零，从而突出有用信号。与固定阈值的传统去噪不同，DRSN中的阈值是通过一个小型子网络自动学习得到的，能够根据输入信号自适应调整。这使得模型在训练过程中学会了如何去噪，而无需人工设定阈值。

DRSN最初由赵明航等人提出用于机械故障诊断，但很快被引入EEG信号处理领域。EEG信号中的噪声来源多样，包括肌电干扰、眼动伪迹、工频干扰等。DRSN通过在特征提取过程中自动抑制噪声分量，提高了模型对强噪声、个体差异的鲁棒性。例如，在麻醉深度估计等任务中，研究者采用DRSN对EEG进行特征提取，相比传统ResNet获得了更稳定的分类性能。总体而言，DRSN标志着EEG分类模型从单纯提升分类精度，转向提升模型鲁棒性的新阶段。
DRSN在EEG分类中的应用
DRSN作为ResNet的一种变体，可以方便地替换EEG分类模型中的特征提取部分。例如，在运动想象分类或情绪识别中，可以将原有的ResNet或EEGNet骨干网络替换为DRSN，以增强去噪能力。DRSN的结构与ResNet类似，由多个残差块组成，每个残差块包含卷积、批归一化和ReLU。

不同之处在于，DRSN在残差块的输出端增加了一个收缩模块，对特征进行软阈值化。这个收缩模块内部通过全局平均池化和两层全连接网络来生成阈值。具体流程是：

首先计算特征图绝对值的全局平均，得到一个与通道数相等的向量
然后通过两个全连接层（中间有ReLU激活，输出经过Sigmoid）将该向量映射到(0,1)范围的系数
最后将该系数与全局平均向量相乘，得到每个通道的阈值
这个阈值被用于对特征进行软阈值化，即x_shrink = sign(x) * max(abs(x) - threshold, 0)。经过软阈值化后，接近零的噪声特征被抑制，而显著的特征得以保留并传递到下一层。

DRSN在EEG分类中的优势体现在两个层面：

自适应去噪：传统去噪需要在预处理阶段手工设计滤波器或阈值，而DRSN将去噪融入模型训练，自动学习最优阈值，减少了对人工经验的依赖
特征增强：通过抑制噪声，模型能够提取更纯净的信号特征，避免噪声干扰导致的判别力下降
例如，在存在大量伪迹的EEG数据上，DRSN往往能取得比普通ResNet更高的分类准确率，因为后者可能将噪声误认为特征而DRSN能够自动忽略噪声分量。

DRSN的PyTorch实现要点
实现DRSN的关键在于构建残差收缩块。下面给出一个简化的PyTorch实现示例，展示如何在残差块中加入收缩模块：


import torch
import torch.nn as nn
import torch.nn.functional as F

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size=1):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_abs = torch.abs(x)                    # 取绝对值
        gap = self.gap(x_abs)                   # 全局平均池化
        gap = torch.flatten(gap, 1)            # 展平为 (batch, channel)
        scale = self.fc(gap)                    # 通过全连接得到系数 (batch, channel)
        threshold = torch.mul(gap, scale)      # 得到阈值
        threshold = torch.unsqueeze(threshold, 2)  # 增加维度 (batch, channel, 1)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub                       # 全零张量，用于 ReLU 实现 max(.,0)
        x_shrink = torch.mul(torch.sign(x), torch.max(sub, zeros))
        return x_shrink

class ResidualBlockShrink(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockShrink, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shrink = Shrinkage(out_channels, gap_size=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shrink(out)      # 在第二个卷积后进行收缩
        if self.conv1.stride != (1,) or in_channels != out_channels:
            # 如果维度不匹配，需要调整残差连接的维度
            residual = F.avg_pool1d(residual, kernel_size=1, stride=self.conv1.stride)
            residual = torch.cat([residual, residual], dim=1)  # 通过复制通道进行填充
        out += residual
        return self.relu(out)
上述代码定义了一个残差收缩块。其中Shrinkage类实现了阈值的学习和软阈值化操作。在ResidualBlockShrink中，特征在经过两个卷积层后，先进行收缩去噪，再与残差连接相加。通过堆叠多个这样的残差收缩块，可以构建完整的DRSN模型。

实现时还需注意维度匹配：当特征图下采样或通道数改变时，残差连接需要相应调整。在示例中，我们采用了一种简单的策略——通过复制通道来填充维度。在实际应用中，也可以使用1x1卷积来调整维度，以减少信息冗余。

总的来说，DRSN的PyTorch实现难度在于正确设计阈值学习子网络和软阈值化流程。一旦实现，即可将其模块化地插入现有的ResNet类模型中。在训练DRSN时，由于引入了额外的子网络，可能需要较多的迭代次数才能收敛，但最终模型往往能在噪声环境下取得更优的泛化性能。这使其成为EEG分类中提升鲁棒性的有力工具。

爆发期：大模型与通用表征阶段（2024年至今）
LaBraM：迈向通用"大脑表征"的大模型
近年来，大规模预训练模型在NLP和计算机视觉领域取得了巨大成功。这股浪潮也影响到了EEG信号处理领域。研究者开始探索构建通用的EEG基础模型，即在海量多源EEG数据上预训练一个模型，使其学会EEG信号的底层表征，然后针对不同下游任务进行微调。LaBraM（Large Brain Model）是2024年提出的代表性工作，被誉为首个面向EEG的大型基础模型。

LaBraM的创新点在于打破数据集壁垒，实现跨任务学习。传统上，EEG模型往往针对特定数据集或任务设计，例如有的模型专注于运动想象分类，有的专注于情绪识别，各自训练数据有限，模型规模也受限。LaBraM通过统一建模思路，将不同来源、不同任务的EEG数据融合起来：它将EEG信号按照通道划分为"EEG通道Patch"，并引入向量量化神经频谱预测（VQ-NSP）作为预训练目标。

简单来说，LaBraM首先训练一个神经Tokenizer，将连续的EEG通道片段编码为离散的"神经词汇"（类似于文本中的Token）。然后，LaBraM采用Transformer架构，通过掩码预测的方式进行预训练：随机遮蔽一部分EEG通道Patch，让模型预测其对应的神经Token。这种预训练方式类似于BERT在文本上的做法，使模型学会理解EEG信号的底层结构和模式。

经过大规模预训练后，LaBraM在各种下游任务上展现出强大的泛化能力。研究者从约20个不同数据集中收集了总计约2500小时的EEG信号用于预训练LaBraM。随后，在包括异常检测、事件类型分类、情绪识别、步态预测等多种下游任务上对LaBraM进行微调，结果显示LaBraM全面超越了当时各领域的最佳模型。

这意味着，通过预训练，LaBraM学到了一种通用的EEG表征，能够适应不同任务的需求，而不再局限于单一任务。这一成果标志着EEG分类模型从"一任务一模型"时代，迈向了"通用大脑表征"时代。

LaBraM的PyTorch复现要点
LaBraM作为一个大型预训练模型，其PyTorch实现涉及数据预处理、Tokenizer训练、Transformer预训练和下游任务微调等多个环节。幸运的是，LaBraM的作者开源了官方实现代码，为研究者提供了宝贵的参考。下面简要介绍复现LaBraM的几个关键步骤：

数据预处理：收集不同数据集的EEG信号，并进行统一的预处理。例如，去除不相关的导联，滤波（0.1–75Hz带通，50Hz陷波），降采样到统一频率（如200Hz）等。然后，将EEG信号按照通道划分为固定长度的片段，例如每个通道取200ms的信号作为一个Patch。这些Patch将作为模型输入的基本单元。
训练神经Tokenizer：LaBraM使用**向量量化变分自编码器（VQ-VAE）**的思想来训练Tokenizer。具体实现中，可以先训练一个VQ-VAE模型，将EEG通道Patch编码为离散的码本索引。这一步相当于构建EEG信号的"词汇表"。PyTorch中可以使用nn.Embedding来实现离散码本的嵌入，通过最小化重构误差和码本匹配损失来训练Tokenizer。
Transformer预训练：构建一个Transformer模型（如BERT架构），输入为EEG通道Patch序列（经过Tokenizer编码后的Token序列）。采用掩码预测任务：随机选取一定比例的Token进行遮蔽，让模型预测这些位置原本的Token。这一过程与BERT类似，可以使用PyTorch的Transformer模块实现。需要注意的是，EEG通道Patch序列可能非常长，因此Transformer的序列长度可能很大。为了高效训练，可以采用分段训练或窗口滑动等策略，将长序列切分为较短的片段进行训练。
下游任务微调：在预训练完成后，将Transformer模型作为特征提取器，针对具体任务添加分类头进行微调。例如，对于分类任务，可以在Transformer输出之上添加一个线性层输出类别概率。微调时，可以采用较小的学习率，以避免破坏预训练学到的通用表征。LaBraM官方代码提供了针对TUAB、TUEV等数据集的微调脚本，可供参考。
复现LaBraM需要一定的工程投入，包括搭建大规模数据管道、训练高性能Transformer等。但其带来的收益是显著的——通过预训练，模型能够利用跨数据集的知识迁移，在数据量有限的任务上取得更好效果。这对于EEG这一数据标注成本高、个体差异大的领域尤为重要。LaBraM的成功也催生了更多关于EEG大模型的研究，例如探索更高效的预训练方法、将LaBraM应用于新的下游任务等。

其他EEG大模型进展
除了LaBraM，近期还有一些工作探索EEG领域的大型模型。例如，EEG-Transformers尝试直接将Transformer应用于EEG分类，通过自监督预训练提升性能。一些研究将生成式预训练引入EEG，如使用GPT-style模型生成EEG信号片段，从而学习EEG的统计特性。

此外，还有工作将EEG与其它模态（如fNIRS、行为数据）结合，构建多模态大模型，以期获得更全面的脑信息表征。这些探索共同推动了EEG分类技术向通用人工智能方向迈进。可以预见，在不久的将来，我们将拥有类似NLP领域的大型EEG模型，只需少量微调即可应用于各种脑机接口和神经科学研究任务。

结语
从时序增强的TCN，到融合注意力的ATCNet，再到自适应去噪的DRSN，最后到预训练大模型LaBraM，EEG分类模型在过去几年经历了快速演进。每一阶段的代表性模型都解决了前一代模型的痛点：

TCN解决了长程依赖问题
ATCNet实现了对关键时段的聚焦
DRSN提升了噪声环境下的鲁棒性
LaBraM则开启了跨任务通用表征的新篇章
这些模型的提出和复现，离不开PyTorch等深度学习框架的支持。通过PyTorch，研究者能够快速实现和验证新想法，将理论模型转化为实际可运行的系统。

展望未来，EEG分类模型将继续朝着更大型、更智能、更通用的方向发展。我们有理由相信，随着数据和算力的增长，以及模型架构的创新，脑电信号的分析将变得更加高效和精准，为脑机接口和神经科学带来更大的突破。

参考资源
官方代码仓库
LaBraM: https://github.com/935963004/LaBraM
ATCNet (TensorFlow): https://github.com/Altaheri/EEG-ATCNet
ATCNet (PyTorch): https://github.com/braindecode/braindecode
DRSN: https://github.com/zhao62/Deep-Residual-Shrinkage-Networks
DRSN (PyTorch): https://github.com/liguge/Deep-Residual-Shrinkage-Networks-for-intelligent-fault-diagnosis-DRSN-
TCN: https://github.com/paul-krug/pytorch-tcn
推荐阅读
Bai et al. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling” (TCN原始论文)
Altaheri et al. “Physics-informed attention temporal convolutional network for EEG-based motor imagery classification” (ATCNet论文)
Zhao et al. “Deep residual shrinkage networks for fault diagnosis” (DRSN论文)
LaBraM论文: “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI”
注意：本文档中的代码示例为简化版本，实际应用中可能需要根据具体任务和数据集进行调整。建议参考官方代码仓库获取完整实现。