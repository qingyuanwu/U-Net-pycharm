import torch

# SR : Segmentation Result
# GT : Ground Truth

'''
召回率（Recall）和准确率（Accuracy）是两种用于评估分类模型性能的指标，它们关注的方面略有不同：

准确率（Accuracy） 衡量的是模型正确分类所有样本的能力。准确率计算为真正例和真负例（True Negatives，即模型正确分类为负类的样本）
的数量除以所有样本的总数。准确率关注的是模型正确分类的样本比例，包括正类和负类。高准确率表示模型在整体上的分类表现较好。


召回率（Recall） 衡量的是模型成功捕获正类样本的能力。召回率计算为真正例（True Positives，即模型正确分类为正类的样本）的数量除以真正例的数量与假负例
（False Negatives，即实际为正类但被错误分类为负类的样本）的数量之和。召回率关注的是模型是否漏掉了正类样本，即偏向避免假阴性。
高召回率表示模型成功捕获了正类样本，但它可能伴随着更高的误报率。

区别：

召回率侧重于正类样本的捕获，不太关心负类样本的分类情况。它对于那些不能错过任何正类样本的应用非常重要，但在不平衡的数据集中，召回率可能会高而准确率低。
准确率关注的是整体分类的正确性，包括正类和负类的分类。它在平衡数据集中通常很有用，但在不平衡数据集中可能不够敏感。

在二元分类问题中，有四种可能的分类结果，这些结果可以用来计算各种性能指标，其中包括真正例、假正例、真负例和假负例：

真正例（True Positives，TP）：这是模型正确地将正类样本分类为正类的情况。换句话说，这是实际为正类的样本，模型也正确地预测为正类。

假负例（False Negatives，FN）：这是模型错误地将正类样本分类为负类的情况。这表示实际为正类的样本被模型错误地预测为负类。

真负例（True Negatives，TN）：这是模型正确地将负类样本分类为负类的情况。实际为负类的样本被模型正确地预测为负类。

假正例（False Positives，FP）：这是模型错误地将负类样本分类为正类的情况。这表示实际为负类的样本被模型错误地预测为正类。
'''

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold
    corr = torch.sum(SR==GT).item()
    total_pixels = GT.numel()
    acc = corr/(total_pixels + 1e-6)
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold
    # TP: True Positives, FN: False Negatives
    TP = ((SR == 1) & (GT == 1)).sum().item()            # .sum().item()的数据类型是int，torch.sum()的数据类型是tensor
    FN = ((SR == 0) & (GT == 1)).sum().item()

    SE = TP / (TP + FN + 1e-6)
    
    return SE

'''
特异性是用于评估二元分类模型性能的一个重要指标，特别是在处理医学诊断、安全系统等领域时。它衡量了模型正确预测负例（Negative）的能力，即模型正确识别未发生事件的能力。

特异性的高低对应着以下情况：

高特异性：模型能够很好地将负例分类为负例，减少了假阳性率，表明模型能够有效地排除非目标情况，从而减少了错误报警的风险。

低特异性：模型较难将负例正确分类，增加了假阳性率，表明模型在排除非目标情况时可能会犯错，增加了错误报警的风险。

特异性通常与灵敏度（敏感度）一起使用，一起提供完整的性能评估。这两个指标可以帮助你理解模型在识别正例和负例时的表现。

在某些情况下，高特异性可能比高灵敏度更为重要，因为模型错误识别负例（假阳性）可能导致不必要的麻烦或成本。特异性通常与其他性能指标如准确率、精确度、F1分数等一起使用，以提供更全面的模型评估。
'''

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold
    # TN: True Negative, FP: False Positive
    TN = ((SR == 0) & (GT == 0)).sum().item()
    FP = ((SR == 1) & (GT == 0)).sum().item()
    SP = TN / (TN + FP + 1e-6)
    
    return SP

'''
精确度（Precision）是一个用于评估分类模型性能的重要指标，通常用于二元分类问题。它度量了模型在预测为正类别的样本中有多少是真正的正类别。精确度可以用以下公式表示：

精确度的高低告诉您模型在识别正类别时有多么准确。在某些应用中，精确度非常重要，特别是当假正例会引发昂贵的后果时，如医学诊断。高精确度意味着您可以相对可靠地信任模型的正类别预测。
'''

def get_precision(SR,GT,threshold=0.5):
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold
    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) & (GT == 1)).sum().item()
    FP = ((SR == 1) & (GT == 0)).sum().item()
    PC = TP / (TP + FP + 1e-6)

    return PC

'''
get_F1 函数的实现基于召回率（Sensitivity）和精确度（Precision）来计算 F1 分数。F1 分数是一个用于评估分类模型性能的综合指标，它同时考虑了模型的召回率和精确度。

F1 分数的取值范围在 0 到 1 之间，其中 1 表示最佳性能，0 表示最差性能。它是召回率和精确度的调和平均。F1 分数对那些正负样本不平衡的情况特别有用。
'''
def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT)
    PC = get_precision(SR, GT)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

'''
get_JS 函数实现的是 Jaccard 相似度（也称为 Jaccard 指数），通常用于度量两个集合的相似性。

在图像分割等任务中，Jaccard 相似度通常用于度量预测的二进制掩码（如SR和GT）与真实标签之间的重叠程度。
'''
def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold

    Inter = (SR & GT).sum().item()
    Union = (SR | GT).sum().item()
    
    JS = Inter/(Union + 1e-6)
    
    return JS

'''
get_DC 函数实现的是 Dice 系数（Dice Coefficient），它也常用于图像分割任务的性能评估。

Dice 系数度量了模型预测的二进制掩码（如SR）与真实标签（如GT）之间的重叠程度。
'''
def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > torch.max(SR)*threshold
    GT = GT > torch.max(GT)*threshold

    Inter = (SR & GT).sum().item()
    DC = 2*Inter/(SR.numel() + GT.numel() + 1e-6)

    return DC



