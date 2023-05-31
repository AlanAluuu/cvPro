## 一、CUDA的安装

首先安装CUDA。这里先简单介绍一下cuda的作用。 CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。 它包含了CUDA指令集架构（ISA）以及GPU内部的并行计算引擎。计算行业正在从只使用CPU的“中央处理”向CPU与GPU并用的“协同处理”发展。为打造这一全新的计算典范，NVIDIA（英伟达）发明了CUDA（Compute Unified Device Architecturem，统一计算设备架构）这一编程模型，是想在应用程序中充分利用CPU和GPU各自的优点。

图形处理器 GPU是显卡上的一块芯片。每台计算机必有主板CPU，但少数计算机可能没有显卡。显卡的全称是“显示适配器”，显卡最初被发明是单纯为了大型 3D 游戏用，后来被发现还可以用来顺带加速 PyTorch 的运行速度（比 CPU 快 10-100 倍）。查看自己的计算机的显卡为：**任务管理器——性能——左侧栏**划到最下面。我的电脑上（win11）如下图所示

![image-20230531000932721](C:\Users\Lulu\Desktop\cv\image\pytorch安装1.png)

下载完cuba安装前首先要查看显卡支持的最高CUDA的版本，以便下载对应的CUDA安装包。这里使用cmd命令行查看。输入nvidia-smi，我的电脑上是11.0。![image-20230530165538951](C:\Users\Lulu\Desktop\cv\image\pytorch安装2.png)

也可以打开控制面板->搜索NVIDIA->双击进入NVIDIA控制面板->点击帮助->系统信息->组件。我的最高支持是11.0.208.

![image-20230530165921502](C:\Users\Lulu\Desktop\cv\image\pytorch安装3.png)



安装torch 之前我们首先需要清楚的了解torch版本和内置cuda以及python之间的关系表，以防后续安装产生很大的问题。（查看本机的python版本为python -V）。这是我截取B站上某位up主整理的表格，而我第一次安装的时候由于python版本为3.10，下载的cuda版本为11.0所以导致了冲突，后续也不能下载。

![image-20230531001432842](C:\Users\Lulu\Desktop\cv\image\pytorch安装4.png)

这里我决定下载cuda 版本为11.0的。这里给出查询官方安装文档的地址：[CUDA Toolkit Archive | NVIDIA Developer ](https://developer.nvidia.com/cuda-toolkit-archive)。

这个网站可以下载不同版本的cuda以及查看在线文档。因为我的操作系统是win11，但是安装11.3.0的时候没有win11的选项，查了资料发现选win10的也可以。所以我下载的环境是这亚子的。

![image-20230530171418415](C:\Users\Lulu\Desktop\cv\image\pytorch安装5.png)



然后下载完毕后双击进行安装，一路安装下去即可。这里我主要参考了这个文章，[win11配置深度学习环境GPU - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/460806048)。这里提供一个测试安装成功与否的方式。命令行win+R打开命令行。输入：nvcc -V可以查看到相应的版本信息即说明安装成功。

## 二、cuDNN的下载

下面下载cuDNN，NVIDIA的cuDNN是针对深度神经网络计算的一个GPU加速库，为标准例程提供了高度优化的实现，比如向前、向后卷积，池化，规范化和激活层。其是NVIDIA Deep Learning SDK的一部分。

cuDNN官网下载链接：https://developer.nvidia.com/rdp/cudnn-archive。这里我选择的是最新版本的CUDA11.x。

![image-20230530174438436](C:\Users\Lulu\Desktop\cv\image\pytorch安装6.png)

下载cudnn需要到NVIDIV官网注册，这一步我稍微卡壳了一下，因为没有办法注册账号，看了网上也有人没有办法注册，这里提供俩思路去下载，一个是直接网上搜安装包，csdn上有很多人都提供了这个的安装包，但是要注意下载的cudnn 版本要和cuda版本相对应；另一种方法是可以到淘宝上搜一下，下载对应安装包就一块多。

下载完后解压cudnn，解压cudnn后的界面是下面的样子。

![image-20230530205128117](C:\Users\Lulu\Desktop\cv\image\pytorch安装7.png)

cudnn后里面有bin，include，lib三个文件夹；而打开上面安装好的CUDA目录，里面也存在bin，include，lib三个文件夹，只要将**cudnn中bin，include内的文件全选复制到CUDA中的bin，include**内即可。而对于cdnn里的lib文件夹，里面还存在一个x64文件夹，而CUDA中lib文件中存在Win32和x64文件，于是这时把**cudnn中lib里的x64文件夹拷贝所有内容到CUDA里lib中x64**文件夹中去。

这里如果你忘记安装的cuba的地址在哪了，就可以去高级属性里面查看环境变量Path ，第一条就是我们刚安装的cuba的bin 的地址。如下所示：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin

![image-20230530204136535](C:\Users\Lulu\Desktop\cv\image\pytorch安装8.png)

验证cuda是否安装成功，首先win+R启动cmd，进入到CUDA安装目录下的 ...\extras\demo_suite，我的就是C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\demo_suite，然后分别运行bandwidthTest.exe和deviceQuery.exe，返回Result=PASS表示cuda安装成功，同样运行deviceQuery.exe也可得到PASS。

![image-20230530204801462](C:\Users\Lulu\Desktop\cv\image\pytorch安装9.png)

![image-20230530204844941](C:\Users\Lulu\Desktop\cv\image\pytorch安装10.png)

## 三、Anaconda的安装

这一步的话我是因为之前下载过，所以就可以直接省略了，可以进入官网https://link.zhihu.com/?target=https%3A//www.anaconda.com/products/individual下载。这里仅仅提供一种判断下载成功与否的判断。win+R打开命令行输入conda list,如果显示下面的图片，就说明下载成功了。

![image-20230530205849371](C:\Users\Lulu\Desktop\cv\image\pytorch安装11.png)

如果显示'conda' 不是内部或外部命令，也不是可运行的程序或批处理文件。则说明环境变量没有配置好。

## 四、pytorch的下载

我本机上的python是3.10的，根据pytorch和python的对应关系，显然不能下载成功。如果想从python3.10降低到python3.6，则需要多安装多一个python3.6的环境。这个是在Anaconda Prompt下操作的。

打开Anaconda Prompt，先配置镜像源，base环境里面输入下面的内容

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 
conda config --set show_channel_urls yes
```

然后在 anaconda prompt中输入以下命令

```shell
conda create -n py36 python=3.6 anaconda
```

py36是这个环境变量的名字,python=3.6是我们要确定的当前环境的python版本数。输入过后如果要安装什么包输入y就行。

如果想要激活这个环境，就输入activate + 环境名称

```shell
activate py36
```

左边的环境就从base（基本环境），变成了py36环境。(如果进行pip list的时候提醒你需要更新pip，按照命令更新即可)

![image-20230531004936356](C:\Users\Lulu\Desktop\cv\image\pytorch安装12.png)

下面我们就需要在这个虚拟的环境中进行下载pytorch

这里先大概解释一下conda环境下安装pytorch的语句

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

这是用于在conda环境中安装pytorch深度学习框架的命令语句，其中，pytorch==1.13.1表示安装1.13.1版本的pytorch，torchvision==0.14.1表示安装对应版本的图像处理库，torchaudio==0.13.1表示安装对应版本的音频处理库，pytorch-cuda=11.6表示安装支持CUDA 11.6的pytorch版本。最后的-c pytorch -c nvidia表示从pytorch和nvidia源中下载安装包。其实给定了 pytorch 版本后，torchvision 和 torchaudio 也唯一确定了。

然后要注意一下pytorch和cuda版本对应关系。再次重申一下这张图。

![image-20230531001432842](C:\Users\Lulu\Desktop\cv\image\pytorch安装13.png)

所以这里我安装了1.7.0版本的pytorch，也就是下面的语句。这里版本对应的语句信息可以在这个网站中找到不同版本的cuda以及不同环操作系统下的pytorch在conda环境下下载的命令：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

![image-20230530213619845](C:\Users\Lulu\Desktop\cv\image\pytorch安装14.png)

然后，cuda11.0对应下面的语句。

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

注意这行命令需要在刚刚激活的py36环境里面运行，因为我的base环境里面的python是3.10，没办法成功下载pytorch，会产生版本冲突。即如下所示

![image-20230531010345732](C:\Users\Lulu\Desktop\cv\image\pytorch安装15.png)

接下来的下载的速度很慢..............................................................下载完毕后用下面的方式测试是否安装成功

![image-20230531121232817](C:\Users\Lulu\Desktop\cv\image\pytorch安装16.png)

最后 exit()退出python程序就行。

## 五、PyCharm的下载

PyCharm 是一款功能强大的 Python 编辑器，官网地址如下[PyCharm：JetBrains为专业开发者提供的Python IDE](https://www.jetbrains.com/zh-cn/pycharm/)。进入官网下载社区版本即可。下载过后双击可执行文件，即可以进行安装。安装路径我选择到了D盘。注意这一部分推荐选择所有

![image-20230531123002736](C:\Users\Lulu\Desktop\cv\image\pytorch安装17.png)

之后一种next安装即可。然后创建new project，因为当时创建了虚拟环境py36并在这个虚拟环境中安装的Pytorch，所以需要在pycharm中配置conda的虚拟环境。

找到之前创建的py36的路径，我的是下面

![image-20230531130635031](C:\Users\Lulu\Desktop\cv\image\pytorch安装18.png)

创建工程时选择Previously configured interpreter，再选择Add Interpreter，注意不要选择New environment using。

![image-20230531130753288](C:\Users\Lulu\Desktop\cv\image\pytorch安装19.png)



然后选择自己创建的虚拟环境py36下的python.exe作为这个项目的python解释器。

![image-20230531131048095](C:\Users\Lulu\Desktop\cv\image\pytorch安装20.png)

然后创建一个python文件进行测试，内容如下

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

但是很奇怪，我报了下面的错误

```
C:\Users\Lulu\.conda\envs\py36\lib\site-packages\numpy\__init__.py:138: UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package, see http://github.com/IntelPython/mkl-service
  from . import _distributor_init
Traceback (most recent call last):
  File "D:\PyCharmPro\test.py", line 1, in <module>
    import torch
  File "C:\Users\Lulu\.conda\envs\py36\lib\site-packages\torch\__init__.py", line 190, in <module>
    from torch._C import *
ImportError: numpy.core.multiarray failed to import
```

参考了csdn上一位博主的文章解决了。就是在创建的虚拟环境py36中输入pip install -U numpy

原文链接如下[(11条消息) 【Pytorch】import torch报错from torch._C import *__冷山_的博客-CSDN博客](https://blog.csdn.net/qq_40905896/article/details/108504472?ops_request_misc=%7B%22request%5Fid%22%3A%22168552193116800182791481%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=168552193116800182791481&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-108504472-null-null.142^v88^insert_down28v1,239^v2^insert_chatgpt&utm_term=line 190%2C in      from torch._C import * ImportError%3A numpy.core.multiarray failed to import&spm=1018.2226.3001.4187)

。解决后再次运行代码，如出如下

![image-20230531164009555](C:\Users\Lulu\Desktop\cv\image\pytorch安装21.png)

这样环境就配置好了！！！！！！