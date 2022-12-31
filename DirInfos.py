from CustomDataset import CustomDataset, EvalDataset

train_dir = r"D:/Library/Desktop/mnist/train"
test_dir = r"D:/Library/Desktop/mnist/test"
train_labels = r"D:/Library/Desktop/mnist/train labels.csv"
test_labels = r"D:/Library/Desktop/mnist/test labels.csv"
train_dataset = CustomDataset(train_dir, train_labels)
test_dataset = CustomDataset(test_dir, test_labels, '.png')
