def calculate_fps(latency_ms):
    if latency_ms <= 0:
        return 0

    return 1000 / latency_ms


def calculate_precision(true_positives, false_positives):
    total = true_positives + false_positives

    if total == 0:
        return 0

    return true_positives / total


def calculate_recall(true_positives, false_negatives):
    total = true_positives + false_negatives

    if total == 0:
        return 0

    return true_positives / total