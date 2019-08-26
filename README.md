# Tusimple 数据集衡量指标:
 acc; fp; fn
 
排榜按照acc排，一帧处理时间大于200ms算作预测失败.

## 流程：
1.读取pred.json，gt.json ；

2.获取json文件的每一行信息（对应不同帧）；

3.对每一帧的每条车道线进行 acc_l, fp_l, fn_l 进行计算；

4.对每一帧的所有车道线精度取平均，得到每帧的精度 acc_f, fp_f, fn_f ；

5.对json文件的所有帧取平均，得到总的final ret: acc,fp,fn.


### 步骤3中车道线精度计算：
1). 根据gt数据计算每条车道线的角度（theta = artan(k)）;

2). 根据两点之间距离为20的阈值设定，计算pred_x和gt_x之间的容许误差 th = 20/cos(theta);

3). 如果(|pred_x - gt_x| < th) ，则pred_x为预测正确的点标记为1；否则，标记为0.
      acc_l_tmp = sum(正确的点) / sum(总点数)

4). 一条真值车道线和所有pred的车道线进行计算比较，acc_l = max(acc_l_tmps),最大值为对应位置匹配上的车道线；
    
    该条车道线是否存在判断： thresh = 0.85, acc_l　> 0.85 匹配，否则，　 fn += 1.

