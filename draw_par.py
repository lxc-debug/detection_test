import matplotlib.pyplot as plt
import re
import numpy as np

# draw_list=['./log_tmp/save/inception_base_one.log','./log_tmp/save/inception_base_two.log','./log_tmp/save/inception_base_three.log']
draw_list=['./log_tmp/save/inception_base_one.log','./log_tmp/save/inception_base_two.log']


# pattern=re.compile(r'epoch:.*?eval_acc:\s*([\d.]+).*?test dataset acc:.*?auc_roc_score:')
pattern=re.compile(r'eval_acc:\s*([0-9.]+)(?=\n[^\n]*test dataset acc)',re.MULTILINE)   # 这个非常重要，用来控制.是否能够匹配换行符

eval_list=list()

for draw_item in draw_list:
    with open(draw_item) as fp:
        context=fp.read()

    tmp_li=pattern.findall(context)
    for item in tmp_li:
        eval_list.append(float(item))
    

# plt.plot(list(range(len(eval_list))),eval_list)
# plt.savefig('./test.jpg')

arr=np.array(eval_list)  

print(arr.reshape(-1,10))