直接python main.py即可运行

需要按照如下格式构造在当前目录下构造一个文件夹。  
## Repository structureasd
------------- 
-data(dir)

    -train_scene (dir)  
        -1(image)  
        -2  
        -3          
        ...  
    -train_ped (dir)  
        -1(image)  
        -2
        -3
        ...
    -test_scene (dir)
        -1(image)
        -2
        -3
        ...
    -test_image(dir)
        -1
        -2
        -3
        ...
    train_label.txt
    test_label.txt
这个是规范的数据存储格式，我们把行人的图像和场景的图像分开存储，且他们的名字一一互相对应，并且train和test已经随机的划分开了。

此外，也可以把scene、ped的存储位置及名称对应的存储，读的时候顺序读即可。

有问题随时问我 应该还有bug 跑起来才可以改。


目前的框架是

场景图像-> resnet（50或者101或者152）-> vector[1024]


行人图像-> resnet（50或者101或者152）-> vector[1024]  -> 两个vector的concat -> vector[2048] -> output
