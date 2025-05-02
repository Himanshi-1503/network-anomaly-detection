def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)


def load_model(filename):
    import joblib
    return joblib.load(filename)


def log_message(message, log_file='logs/training.log'):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")