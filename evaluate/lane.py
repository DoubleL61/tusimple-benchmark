import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json


class LaneEval(object):
    lr = LinearRegression() # 线性回归实例化 --ll20190826
    pixel_thresh = 20     # 两个点之间的(dis)容许误差范围  --ll20190826
    pt_thresh = 0.85 # 线的精度阈值 --ll20190826

    @staticmethod
    # 计算这条车道线的坐标进行直线拟合后的角度（斜率） --ll20190826
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)   # 线性回归拟合线性模型 --ll20190826
            k = LaneEval.lr.coef_[0]   # 获取斜率 --ll20190826
            theta = np.arctan(k)       # 转换成对应的角度 --ll20190826
        else:
            theta = 0
        return theta

    @staticmethod
    # 计算线的精度 --ll20190826 pred： 一条车道线对应的横坐标向量
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        # 这条车道线上阈值范围内的点数占总点数的比例作为线的精度 --ll20190826
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt) # 阈值范围内为1，否则为0 --ll20190826

    @staticmethod
    # 对json文件的一行（单帧）进行比较 --ll20190826
    # pred：预测的横坐标；gt:真值的横坐标；y_samples:真值的纵坐标 --ll20190826
    def bench(pred, gt, y_samples, running_time):
        # 预测的点数和真值点数是否相等 --ll20190826
        if any(len(p) != len(y_samples) for p in pred):  # p: 一条车道线对应的横坐标 --ll20190826
            raise Exception('Format of lanes error.')
        # 小于5fps视为检测失败；最多可检测5条，检测过多视为失败 --ll20190826
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        # 计算每条车道线的坐标进行直线拟合后的角度（斜率） --ll20190826 angel=[l1_theta, l2_theta, l3_theta, l4_theta]
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt] # x_gts：其中一条车道线的横坐标
        # 每条车道线对应的容许误差。两个点之间的容许误差，计算两个横坐标之间的容许误差 --ll20190826
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs): # 真值和阈值做压缩 --ll20190826
            # accs：label中的一条车道线和预测里的所有车道线进行精度计算，最大的那个为匹配的那个。 --ll20190826
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc) # 记下每一条车道线的精度 --ll20190826
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:  # TODO: 不懂这个判断 --ll20190826
            fn -= 1
        s = sum(line_accs) # 这一帧所有车道线的累计精度  --ll20190826
        if len(gt) > 4:
            s -= min(line_accs) # 只保留最大概率的四个 --ll20190826
        # 返回平均每条精度作为该帧精度， fp, fn --ll20190826
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    # 对整个json进行循环比较 --ll20190826
    def bench_one_submit(pred_file, gt_file):
        try:
            # 读取预测的json文件 --ll20190826
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]  # line： 对应一个clips--ll20190826
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        # 读取标签json文件 --ll20190826
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        # 判断两者长度是否相等 --ll20190826
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        # 获取clips的路径  --ll20190826
        gts = {l['raw_file']: l for l in json_gt}
        # 初始化 --ll20190826
        accuracy, fp, fn = 0., 0., 0.
        # 循环判断json文件里的预测结果,每次处理一帧  --ll20190826
        for pred in json_pred:
            # 异常判断 --ll20190826
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            # 获取预测的json的三行信息 --ll20190826
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']  # 二维list, 存放一帧里的所有车道线的横坐标 --ll20190826
            run_time = pred['run_time']
            # 异常判断  --ll20190826
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            # 读取raw_file当前帧对应的真值 --ll20190826
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            # 对当前帧的真值和预测结果进行比较 --ll20190826
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            # 对每一帧的平均精度、fp和fn进行累加 --ll20190826
            accuracy += a
            fp += p
            fn += n
        num = len(gts) # 获取帧数 --ll20190826
        # the first return parameter is the default ranking parameter
        return json.dumps([
            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
        ])


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]) # pred_file, gt_file  --ll20190826
    except Exception as e:
        print e.message
        sys.exit(e.message)
