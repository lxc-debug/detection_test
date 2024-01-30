import re

with open('log_tmp/save/all_archi_base_attach_seed1.log',mode='r') as fp:
    all_archi_base_attach=fp.read()

with open('log_tmp/save/all_archi_bin_attach_seed1.log',mode='r') as fp:
    all_archi_bin_attach=fp.read()

with open('log_tmp/save/all_noarchi_base_seed1.log',mode='r') as fp:
    all_noarchi_base=fp.read()

with open('log_tmp/save/all_noarchi_bin_seed1.log',mode='r') as fp:
    all_noarchi_bin=fp.read()

with open('log_tmp/save/all_archi_base_useq_seed1.log',mode='r') as fp:
    all_archi_base_useq=fp.read()

with open('log_tmp/save/all_archi_bin_useq_seed1.log',mode='r') as fp:
    all_archi_bin_useq=fp.read()

# pattern=re.compile(r'auc_roc_score:([\d\.]+)')
# all_archi_base_li=pattern.findall(all_archi_base)
# all_archi_bin_li=pattern.findall(all_archi_bin)
# all_noarchi_base_li=pattern.findall(all_noarchi_base)
# all_noarchi_bin_li=pattern.findall(all_noarchi_bin)
# all_archi_base_useq_li=pattern.findall(all_archi_base_useq)
# all_archi_bin_useq_li=pattern.findall(all_archi_bin_useq)



# all_archi_base_li=[float(item) for item in all_archi_base_li]
# all_archi_bin_li=[float(item) for item in all_archi_bin_li]
# all_noarchi_base_li=[float(item) for item in all_noarchi_base_li]
# all_noarchi_bin_li=[float(item) for item in all_noarchi_bin_li]
# all_archi_base_useq_li=[float(item) for item in all_archi_base_useq_li]
# all_archi_bin_useq_li=[float(item) for item in all_archi_bin_useq_li]


# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_archi_base_li)}',all_archi_base)[0]
# print(f'{"all_archi_base":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_archi_base_li)}')

# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_archi_bin_li)}',all_archi_bin)[0]
# print(f'{"all_archi_bin":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_archi_bin_li)}')

# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_noarchi_base_li)}',all_noarchi_base)[0]
# print(f'{"all_noarchi_base":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_noarchi_base_li)}')

# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_noarchi_bin_li)}',all_noarchi_bin)[0]
# print(f'{"all_noarchi_bin":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_noarchi_bin_li)}')

# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_archi_base_useq_li)}',all_archi_base_useq)[0]
# print(f'{"all_archi_base_useq":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_archi_base_useq_li)}')

# train_acc,eval_acc,test_acc=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:'+f'{max(all_archi_bin_useq_li)}',all_archi_bin_useq)[0]
# print(f'{"all_archi_bin_useq":20} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{max(all_archi_bin_useq_li)}')


res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_archi_base_attach)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_archi_base_attach":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')

res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_archi_bin_attach)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_archi_bin_attach":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')

res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_noarchi_base)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_noarchi_base":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')

res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_noarchi_bin)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_noarchi_bin":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')

res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_archi_base_useq)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_archi_base_useq":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')

res=re.findall(r'train_acc:\s+(\d+\.\d+)[^\n]*eval_acc:\s+(\d+\.\d+)[^\n]*\n[^\n]*test dataset acc:\s+(\d+\.\d+)[^\n]*auc_roc_score:(\d+\.\d+)',all_archi_bin_useq)
train_acc,eval_acc,test_acc,auc_roc_score=max(res,key=lambda x:float(x[1]))
print(f'{"all_archi_bin_useq":25} train_acc:{train_acc}|eval_acc:{eval_acc}|test_acc:{test_acc}|auc_roc:{auc_roc_score}')