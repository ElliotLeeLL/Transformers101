from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true,
        y_pred,
        target_names=['Negative Review', 'Positive Review']
    )
    print(performance)

