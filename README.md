# 程序内容   
这两个文件是基于tensorflow实现YOLOv3的网络及损失函数部分。   
用于学习YOLOv3的网络及其损失函数是如何构造的   
不同模型除了网络与损失函数之外其余部分的实现都比较类似。   

YOLOV3数据集构造过程：   
　　step1.从原始文件(annotation_file)中读取的一张图片原始信息为true_boxes(维度[m, 5]):   
　　　　　[[true_box1_x1, true_box1_y1, true_box1_w, true_box1_h, class_info]  
　　　　　　[true_box2_x1, true_box2_y1, true_box2_w, true_box2_h, class_info],   
　　　　　　...]
　　step2.从step1得到的原始信息中我们可以得到该图片的GT框的wh集合true_wh(维度[m, 2])   
　　step3.由kmeans得到的先验anchor框的信息为anchor_wh(维度：[9, 2])：  
　　step4.将true_wh的维度变为[m, 1, 2], anchor_wh的维度变为[1, 9, 2]  
　　step5.将true_wh与anchor_wh求iou,得到iou矩阵IOU:[m, 9]  
　　step6.对IOU的第二维求argmax，得到有m个元素的列表  
　　　　　列表中第i个元素x：代表anchors中与该张图片第i个true_box的IOU最大的是 第x个anchor  

　　step7.我们要从3个scale去初始化第l张图片的label矩阵y_ture[l]:  
　　　　　而在每一个scale中又有三个不同大小的anchor，因此label矩阵具体如下：  
　　　　　y_true[l][0]:将图片分成13*13的格子　[13, 13, 3, 255]　(对应anchor：[116, 90], [156, 198], [373, 326])  
　　　　　y_true[l][1]:将图片分成26*26的格子　[13, 13, 3, 255]　(对应anchor：[30, 61], [62, 45], [59, 119])  
　　　　　y_true[l][2]:将图片分成52*52的格子　[52, 52, 3, 255]　(对应anchor：[10, 13], [16, 30], [33, 23])  
		  
　　step8.将true_boxes的信息填进y_true矩阵  
　　　　　根据step6得到的IOU矩阵可以判断每个true_box应该整合进y_true的哪个位置  
　　　　　例如：true_box:[200, 200, 115，91， 5] => x:200, y:200, w:115, h:91, class_info:5   
　　　　　　　　step8.1　wh与[116, 91]的IOU最大=>  信息应该填入y_true[l][0]=> 是将图片分成13*13的格子  
　　　　　　　　step8.2　x=200，y=200，图片分成13*13格子 => 计算 信息应该填入哪个格子    
　　　　　　　　　　　　　x_g = 200 / 416 * 13 = 6.26 => 第7列   
　　　　　　　　　　　　　y_g = 200 /416 *13 = 6.26 => 第7行   
				step8.3　根据上面两步可得该true_box的信息放入第一个sacle的第7行第7列的格子   
				step8.4　填入信息：   
　　　　　　　　　　　　　y_true[l][0][int(x_g), int(y_g), 0, 0:4] = true_box[0:4] //box位置   
　　　　　　　　　　　　　y_true[l][0][int(x_g), int(y_g), 0, 4+true_box[4]] = 1 //类别   
	step9.将所有的true_box填入y_true矩阵后，得到数据集的label信息。  
				
				
	
YOLOv3损失函数构造过程：       
![result_2](https://github.com/Liu-Yicheng/Fast-RCNN/raw/master/result/2.jpg)      


				


